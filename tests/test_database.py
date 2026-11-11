"""Synthetic database: schema constraints, integrity, scenarios, reproducibility."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timedelta

import pytest

from backend.database.db import connect_readonly, connect_writable
from backend.database.schema import CLINICAL_TABLES
from backend.database.synthetic_generator import (
    SCENARIOS,
    TS_FMT,
    GeneratorConfig,
    generate_dataset,
    write_sqlite,
)
from backend.database.validation import run_integrity_checks, table_counts


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "test.db"
    write_sqlite(generate_dataset(GeneratorConfig()), path)
    return path


@pytest.fixture()
def conn(db_path):
    c = connect_readonly(db_path)
    yield c
    c.close()


@pytest.fixture()
def wconn(tmp_path):
    """Writable copy for constraint tests (each test gets a fresh DB)."""
    path = tmp_path / "w.db"
    write_sqlite(generate_dataset(GeneratorConfig(patients_per_scenario_per_type=1)), path)
    c = connect_writable(path)
    yield c
    c.close()


def _ts(s: str) -> datetime:
    return datetime.strptime(s, TS_FMT)


def _gt(conn) -> dict[str, dict]:
    return {
        r["BookingNumber"]: {"scenario": r["Scenario"], "trajectory": r["Trajectory"], "details": json.loads(r["Details"])}
        for r in conn.execute("SELECT * FROM synthetic_ground_truth")
    }


def _scores(conn, booking: str) -> list[int]:
    return [r[0] for r in conn.execute(
        "SELECT Score FROM withdrawal_assessments WHERE BookingNumber = ? ORDER BY AssessmentDateTime", (booking,))]


# --------------------------------------------------------------------------- #
# Size & distribution
# --------------------------------------------------------------------------- #


def test_dataset_size_and_distribution(conn):
    counts = table_counts(conn)
    assert counts["patient_bookings"] == 100
    assert 1000 <= counts["withdrawal_assessments"] <= 1200
    assert 1000 <= counts["vital_signs"] <= 1200
    assert counts["medication_orders"] == 200
    assert 700 <= counts["medication_administrations"] <= 800
    assert 300 <= counts["nursing_notes"] <= 330
    types = Counter(r[0] for r in conn.execute("SELECT WithdrawalType FROM patient_bookings"))
    assert types == {"Alcohol": 50, "Opioid": 50}
    scen = Counter(r[0] for r in conn.execute("SELECT Scenario FROM synthetic_ground_truth"))
    assert set(scen) == set(SCENARIOS) and set(scen.values()) == {10}


def test_one_booking_per_patient_and_distinct_identifiers(conn):
    rows = conn.execute("SELECT PatientID, BookingNumber FROM patient_bookings").fetchall()
    assert len({r[0] for r in rows}) == 100 and len({r[1] for r in rows}) == 100
    assert not {r[0] for r in rows} & {r[1] for r in rows}


# --------------------------------------------------------------------------- #
# Integrity
# --------------------------------------------------------------------------- #


def test_all_integrity_checks_pass(conn):
    failures = {k: v for k, v in run_integrity_checks(conn).items() if v}
    assert failures == {}


def test_score_ranges_and_instrument_consistency(conn):
    for atype, lo, hi in conn.execute(
        "SELECT AssessmentType, MIN(Score), MAX(Score) FROM withdrawal_assessments GROUP BY AssessmentType"
    ):
        assert lo >= 0 and hi <= (67 if atype == "CIWA-Ar" else 48)
    mixed = conn.execute("""SELECT BookingNumber FROM withdrawal_assessments
                            GROUP BY BookingNumber HAVING COUNT(DISTINCT AssessmentType) > 1""").fetchall()
    assert mixed == []


def test_timestamps_valid_and_chronological(conn):
    for table, col in [("withdrawal_assessments", "AssessmentDateTime"), ("vital_signs", "RecordedDateTime"),
                       ("nursing_notes", "NoteDateTime"), ("medication_administrations", "ScheduledDateTime")]:
        for (v,) in conn.execute(f"SELECT {col} FROM {table}"):
            _ts(v)  # raises if malformed
    # Within each booking, assessments are at least a few minutes apart.
    rows = conn.execute("SELECT BookingNumber, AssessmentDateTime FROM withdrawal_assessments ORDER BY BookingNumber, AssessmentDateTime").fetchall()
    for (b1, t1), (b2, t2) in zip(rows, rows[1:]):
        if b1 == b2:
            assert _ts(t2) - _ts(t1) >= timedelta(minutes=30)


def test_records_within_monitoring_window(conn):
    for b, adm in conn.execute("SELECT BookingNumber, AdmissionDateTime FROM patient_bookings"):
        last = conn.execute("SELECT MAX(AssessmentDateTime) FROM withdrawal_assessments WHERE BookingNumber = ?", (b,)).fetchone()[0]
        assert _ts(last) - _ts(adm) <= timedelta(hours=49)


def test_administration_status_and_time_consistency(conn):
    rows = conn.execute("SELECT AdministrationStatus, AdministrationDateTime FROM medication_administrations").fetchall()
    statuses = Counter(r[0] for r in rows)
    assert set(statuses) == {"Administered", "Not administered", "Refused", "Held", "Not documented"}
    for status, when in rows:
        assert (when is None) == (status == "Not documented")


def test_nulls_only_where_scenario_injects_them(conn):
    gt = _gt(conn)
    rows = conn.execute("""SELECT BookingNumber FROM vital_signs WHERE SBP IS NULL OR DBP IS NULL OR Pulse IS NULL
                           OR Temperature IS NULL OR RespiratoryRate IS NULL OR OxygenSaturation IS NULL""").fetchall()
    assert rows, "expected some NULL measurements"
    assert {gt[r[0]]["scenario"] for r in rows} == {"missing_vitals"}


# --------------------------------------------------------------------------- #
# Constraints reject bad data
# --------------------------------------------------------------------------- #


def _booking(c, wtype):
    return c.execute("SELECT PatientID, BookingNumber FROM patient_bookings WHERE WithdrawalType = ? LIMIT 1", (wtype,)).fetchone()


def test_rejects_out_of_range_score(wconn):
    pid, bk = _booking(wconn, "Opioid")
    with pytest.raises(sqlite3.IntegrityError):
        wconn.execute("INSERT INTO withdrawal_assessments VALUES ('ASM-X', ?, ?, '2026-03-05T10:00:00', 'COWS', 49)", (pid, bk))


def test_rejects_wrong_instrument_for_booking(wconn):
    pid, bk = _booking(wconn, "Opioid")
    with pytest.raises(sqlite3.IntegrityError, match="does not match"):
        wconn.execute("INSERT INTO withdrawal_assessments VALUES ('ASM-X', ?, ?, '2026-03-05T10:00:00', 'CIWA-Ar', 5)", (pid, bk))


def test_rejects_cross_patient_booking(wconn):
    (p1, _), (_, b2) = wconn.execute("SELECT PatientID, BookingNumber FROM patient_bookings LIMIT 2").fetchall()
    with pytest.raises(sqlite3.IntegrityError):
        wconn.execute("INSERT INTO nursing_notes VALUES ('NOTE-X', ?, ?, '2026-03-05T10:00:00', 'text')", (p1, b2))


def test_rejects_malformed_timestamp_and_bad_status(wconn):
    pid, bk = _booking(wconn, "Alcohol")
    with pytest.raises(sqlite3.IntegrityError):
        wconn.execute("INSERT INTO nursing_notes VALUES ('NOTE-X', ?, ?, '03/05/2026 10:00', 'text')", (pid, bk))
    oid = wconn.execute("SELECT OrderID FROM medication_orders WHERE BookingNumber = ?", (bk,)).fetchone()[0]
    with pytest.raises(sqlite3.IntegrityError):  # 'Not documented' must have no administration time
        wconn.execute("INSERT INTO medication_administrations VALUES ('MAR-X', ?, ?, ?, '2026-03-05T10:00:00', "
                      "'2026-03-05T10:05:00', 'Not documented')", (oid, pid, bk))


def test_rejects_implausible_vitals(wconn):
    pid, bk = _booking(wconn, "Alcohol")
    with pytest.raises(sqlite3.IntegrityError):
        wconn.execute("INSERT INTO vital_signs VALUES ('VIT-X', ?, ?, '2026-03-05T10:00:00', 80, 120, 70, 98.6, 16, 97)", (pid, bk))


def test_readonly_connection_blocks_writes(conn):
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM nursing_notes")


# --------------------------------------------------------------------------- #
# Scenario ground truth is reflected in the data
# --------------------------------------------------------------------------- #


def test_trajectory_shapes(conn):
    for b, g in _gt(conn).items():
        s = _scores(conn, b)
        if g["scenario"] in ("increasing", "conflicting_documentation"):
            assert s[-1] - s[0] >= 5, (b, s)
        elif g["scenario"] in ("decreasing", "missing_assessments"):
            assert s[0] - s[-1] >= 5, (b, s)
        elif g["scenario"] == "stable":
            assert max(s) - min(s) <= 8, (b, s)
        elif g["scenario"] == "fluctuating":
            diffs = [y - x for x, y in zip(s, s[1:])]
            assert any(d > 0 for d in diffs) and any(d < 0 for d in diffs)


def test_missing_assessments_have_no_rows_at_expected_times(conn):
    for b, g in _gt(conn).items():
        if g["scenario"] != "missing_assessments":
            continue
        times = [_ts(r[0]) for r in conn.execute("SELECT AssessmentDateTime FROM withdrawal_assessments WHERE BookingNumber = ?", (b,))]
        assert len(times) == 9
        for missing in g["details"]["expected_but_unrecorded_times"]:
            assert all(abs(t - _ts(missing)) > timedelta(hours=1) for t in times)


def test_irregular_intervals_contain_long_gap(conn):
    for b, g in _gt(conn).items():
        if g["scenario"] == "irregular_intervals":
            times = [_ts(r[0]) for r in conn.execute("SELECT AssessmentDateTime FROM withdrawal_assessments WHERE BookingNumber = ? ORDER BY 1", (b,))]
            gaps = [(y - x).total_seconds() / 3600 for x, y in zip(times, times[1:])]
            assert max(gaps) >= 8.5 and min(gaps) < 7


def test_incomplete_monitoring_stops_early(conn):
    for b, g in _gt(conn).items():
        if g["scenario"] != "incomplete_monitoring":
            continue
        adm = _ts(conn.execute("SELECT AdmissionDateTime FROM patient_bookings WHERE BookingNumber = ?", (b,)).fetchone()[0])
        last = _ts(conn.execute("SELECT MAX(AssessmentDateTime) FROM withdrawal_assessments WHERE BookingNumber = ?", (b,)).fetchone()[0])
        assert last - adm <= timedelta(hours=24.5)
        if not g["details"]["released_early"]:
            assert g["details"]["undocumented_after_records_end"]


def test_ground_truth_references_existing_records(conn):
    lookup = {
        "assessment": ("withdrawal_assessments", "AssessmentID"),
        "administration": ("medication_administrations", "AdministrationID"),
        "note": ("nursing_notes", "NoteID"),
    }
    for b, g in _gt(conn).items():
        for c in g["details"].get("conflicts", []):
            for key, (table, col) in lookup.items():
                if key in c:
                    row = conn.execute(f"SELECT BookingNumber FROM {table} WHERE {col} = ?", (c[key],)).fetchone()
                    assert row and row[0] == b
        for ch in g["details"].get("medication_status_changes", []):
            st = conn.execute("SELECT AdministrationStatus FROM medication_administrations WHERE AdministrationID = ?",
                              (ch["administration"],)).fetchone()[0]
            assert st == ch["status"]


def test_medication_names_are_explicitly_fictional(conn):
    names = {r[0] for r in conn.execute("SELECT DISTINCT MedicationName FROM medication_orders")}
    assert names and all("Fictional" in n and "demo only" in n for n in names)


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #


def _dump(path) -> dict:
    c = connect_readonly(path)
    out = {t: [tuple(r) for r in c.execute(f"SELECT * FROM {t} ORDER BY 1")] for t in CLINICAL_TABLES + ["synthetic_ground_truth"]}
    c.close()
    return out


def test_same_seed_reproduces_identical_data(tmp_path, db_path):
    p = tmp_path / "again.db"
    write_sqlite(generate_dataset(GeneratorConfig()), p)
    assert _dump(p) == _dump(db_path)


def test_different_seed_changes_data(tmp_path, db_path):
    p = tmp_path / "other.db"
    write_sqlite(generate_dataset(GeneratorConfig(seed=1)), p)
    assert _dump(p)["withdrawal_assessments"] != _dump(db_path)["withdrawal_assessments"]


def test_assessment_interval_is_configurable(tmp_path):
    p = tmp_path / "six.db"
    write_sqlite(generate_dataset(GeneratorConfig(assessment_interval_hours=6, n_assessments=8, patients_per_scenario_per_type=1)), p)
    c = connect_readonly(p)
    b = c.execute("SELECT g.BookingNumber FROM synthetic_ground_truth g WHERE Scenario = 'stable' LIMIT 1").fetchone()[0]
    times = [_ts(r[0]) for r in c.execute("SELECT AssessmentDateTime FROM withdrawal_assessments WHERE BookingNumber = ? ORDER BY 1", (b,))]
    gaps = [(y - x).total_seconds() / 3600 for x, y in zip(times, times[1:])]
    assert len(times) == 8 and all(5.4 <= g <= 6.6 for g in gaps)
    c.close()


def test_refuses_to_overwrite_without_flag(db_path):
    with pytest.raises(FileExistsError):
        write_sqlite(generate_dataset(GeneratorConfig(patients_per_scenario_per_type=1)), db_path)
