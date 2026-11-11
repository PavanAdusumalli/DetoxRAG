"""Timeline analytics: unit tests on hand-built records + ground-truth checks
against a freshly generated synthetic database."""

from __future__ import annotations

import pytest

from backend.database import queries as q
from backend.database.db import connect_readonly
from backend.database.synthetic_generator import GeneratorConfig, generate_dataset, write_sqlite
from backend.services.schemas import Assessment, Booking, MedicationAdministration, NursingNote, VitalSign
from backend.services.timeline import (
    AnalyticsConfig,
    build_patient_timeline,
    calculate_assessment_intervals,
    calculate_score_changes,
    identify_documentation_conflicts,
    identify_documentation_gaps,
    resolve_window,
    summarize_medications,
    summarize_vital_signs,
)
from evaluation.ground_truth import evaluate_all, summarize

CFG = AnalyticsConfig(review_interval_hours=4, gap_tolerance=1.5, pairing_minutes=30, conflict_minutes=60)
BOOKING = Booking(PatientID="P900", BookingNumber="B9000", AdmissionDateTime="2026-03-01T07:00:00",
                  ReleaseDateTime=None, WithdrawalType="Alcohol")


def A(i, t, s, typ="CIWA-Ar"):
    return Assessment(AssessmentID=f"ASM-{i:05d}", AssessmentDateTime=f"2026-03-01T{t}:00", AssessmentType=typ, Score=s)


def V(i, t, **kw):
    base = dict(SBP=120, DBP=80, Pulse=80, Temperature=98.2, RespiratoryRate=16, OxygenSaturation=98)
    base.update(kw)
    return VitalSign(VitalID=f"VIT-{i:05d}", RecordedDateTime=f"2026-03-01T{t}:00", **base)


def window(start="2026-03-01T07:00:00", end="2026-03-01T20:00:00"):
    return resolve_window(BOOKING, None, start=start, end=end)


# --------------------------------------------------------------------------- #
# Score changes — the worked example from the project specification
# --------------------------------------------------------------------------- #


def test_spec_example_score_changes():
    assessments = [A(1, "08:00", 6), A(2, "12:00", 9), A(3, "16:00", 13), A(4, "20:00", 17)]
    s = calculate_score_changes(assessments, "CIWA-Ar")
    assert (s.earliest.score, s.latest.score, s.change) == (6, 17, 11)
    assert s.observation_span_hours == 12.0
    assert s.n_assessments == 4
    assert s.net_direction == "increasing"
    assert (s.minimum.score, s.maximum.score) == (6, 17)
    assert s.direction_changes == 0


def test_score_sorting_is_chronological_regardless_of_input_order():
    s = calculate_score_changes([A(3, "16:00", 13), A(1, "08:00", 6), A(2, "12:00", 9)], "CIWA-Ar")
    assert (s.earliest.assessment_id, s.latest.assessment_id) == ("ASM-00001", "ASM-00003")


def test_score_direction_variants():
    dec = calculate_score_changes([A(1, "08:00", 20), A(2, "12:00", 12)], "CIWA-Ar")
    assert dec.net_direction == "decreasing" and dec.change == -8
    flat = calculate_score_changes([A(1, "08:00", 5), A(2, "12:00", 9), A(3, "16:00", 5)], "CIWA-Ar")
    assert flat.net_direction == "unchanged" and flat.change == 0 and flat.direction_changes == 1
    one = calculate_score_changes([A(1, "08:00", 5)], "CIWA-Ar")
    assert one.net_direction == "single assessment" and one.change is None
    none = calculate_score_changes([], "COWS")
    assert none.net_direction == "no assessments recorded" and none.earliest is None and none.scale_max == 48


def test_min_max_ties_use_first_occurrence():
    s = calculate_score_changes([A(1, "08:00", 9), A(2, "12:00", 4), A(3, "16:00", 9), A(4, "18:00", 4)], "CIWA-Ar")
    assert s.maximum.assessment_id == "ASM-00001" and s.minimum.assessment_id == "ASM-00002"


# --------------------------------------------------------------------------- #
# Intervals & window
# --------------------------------------------------------------------------- #


def test_interval_statistics_and_long_gap():
    assessments = [A(1, "08:00", 6), A(2, "12:00", 9), A(3, "19:30", 13)]
    iv = calculate_assessment_intervals(assessments, window(), CFG)
    assert [i.hours for i in iv.intervals] == [4.0, 7.5]
    assert (iv.min_hours, iv.max_hours, iv.mean_hours, iv.median_hours) == (4.0, 7.5, 5.75, 5.75)
    assert iv.flag_threshold_hours == 6.0
    assert [(i.from_id, i.to_id) for i in iv.long_intervals] == [("ASM-00002", "ASM-00003")]
    assert iv.leading_gap is None and iv.trailing_gap is None


def test_leading_and_trailing_gaps():
    iv = calculate_assessment_intervals([A(1, "14:00", 6)], window("2026-03-01T07:00:00", "2026-03-01T22:00:00"), CFG)
    assert iv.leading_gap.hours == 7.0 and iv.trailing_gap.hours == 8.0


def test_window_last_n_hours_and_clamping():
    w = resolve_window(BOOKING, "2026-03-02T19:00:00", hours=12)
    assert (w.start, w.end, w.hours) == ("2026-03-02T07:00:00", "2026-03-02T19:00:00", 12.0)
    clamped = resolve_window(BOOKING, "2026-03-01T15:00:00", hours=12)
    assert clamped.start == BOOKING.AdmissionDateTime and clamped.hours == 8.0
    assert "record begins at admission" in clamped.label
    full = resolve_window(BOOKING, "2026-03-02T19:00:00")
    assert full.start == BOOKING.AdmissionDateTime and full.label == "Entire recorded booking"
    with pytest.raises(q.InvalidIdentifier):
        resolve_window(BOOKING, "2026-03-02T19:00:00", hours=-1)
    with pytest.raises(q.InvalidIdentifier):
        resolve_window(BOOKING, None, start="2026-03-02T10:00:00", end="2026-03-02T09:00:00")


# --------------------------------------------------------------------------- #
# Vitals, medications, gaps, conflicts
# --------------------------------------------------------------------------- #


def test_vitals_summary_handles_missing_values():
    vs = summarize_vital_signs([V(1, "08:00", Pulse=80, Temperature=None), V(2, "12:00", Pulse=96, Temperature=98.9)])
    m = {x.measure: x for x in vs.measures}
    assert vs.n_sets == 2
    assert (m["Pulse"].earliest.value, m["Pulse"].latest.value, m["Pulse"].change) == (80, 96, 16)
    assert m["Temperature"].n_recorded == 1 and m["Temperature"].n_not_recorded == 1
    assert m["Temperature"].change is None  # only one recorded value
    assert m["SBP"].unit == "mmHg"


def MAR(i, status, sched="08:00", admin="08:10"):
    return MedicationAdministration(
        AdministrationID=f"MAR-{i:05d}", OrderID="ORD-0001", MedicationName="Fictional Med AW-1 (demo only)",
        ScheduledDateTime=f"2026-03-01T{sched}:00",
        AdministrationDateTime=None if status == "Not documented" else f"2026-03-01T{admin}:00",
        AdministrationStatus=status)


def test_medication_status_handling():
    rows = [MAR(1, "Administered"), MAR(2, "Refused"), MAR(3, "Held"), MAR(4, "Not documented"), MAR(5, "Not administered")]
    s = summarize_medications([], rows)
    assert s.status_counts == {"Administered": 1, "Held": 1, "Not administered": 1, "Not documented": 1, "Refused": 1}
    assert [m.AdministrationID for m in s.not_administered] == ["MAR-00002", "MAR-00003", "MAR-00004", "MAR-00005"]
    gaps = identify_documentation_gaps([], [], rows, calculate_assessment_intervals([], window(), CFG), window(), CFG)
    undocumented = [g for g in gaps if g.gap_type == "medication_not_documented"]
    assert [g.record_ids for g in undocumented] == [["MAR-00004"]]  # only 'Not documented' is a gap
    assert "not the same as 'Not administered'" in undocumented[0].description


def test_gap_detection_pairs_assessments_and_vitals():
    assessments = [A(1, "08:00", 6), A(2, "12:00", 9)]
    vitals = [V(1, "08:05"), V(2, "16:00", Temperature=None)]  # none near 12:00; 16:00 has no assessment
    w = window()
    gaps = identify_documentation_gaps(assessments, vitals, [], calculate_assessment_intervals(assessments, w, CFG), w, CFG)
    types = {(g.gap_type, tuple(g.record_ids)) for g in gaps}
    assert ("assessment_without_vital_signs", ("ASM-00002",)) in types
    assert ("vital_signs_without_assessment", ("VIT-00002",)) in types
    assert ("missing_vital_measurement", ("VIT-00002",)) in types
    assert ("no_assessment_at_window_end", ("ASM-00002",)) in types


def test_no_assessments_in_window_is_reported_without_claiming_none_were_done():
    gaps = identify_documentation_gaps([], [], [], calculate_assessment_intervals([], window(), CFG), window(), CFG)
    assert gaps[0].gap_type == "no_assessments_in_window"
    assert "recorded in the retrieved dataset" in gaps[0].description


def test_conflict_rules():
    assessments = [A(1, "08:00", 3), A(2, "12:00", 10), A(3, "16:00", 24)]
    notes = [
        NursingNote(NoteID="NOTE-0001", NoteDateTime="2026-03-01T16:10:00", NoteText="Patient resting comfortably. No signs of withdrawal observed."),
        NursingNote(NoteID="NOTE-0002", NoteDateTime="2026-03-01T08:30:00", NoteText="Patient refused scheduled dose."),
        NursingNote(NoteID="NOTE-0003", NoteDateTime="2026-03-01T12:20:00", NoteText="No signs of withdrawal observed."),  # mid-range score: no flag
    ]
    conflicts = identify_documentation_conflicts(assessments, [MAR(1, "Administered", "08:00", "08:15")], notes, CFG)
    pairs = {(c.conflict_type, tuple(c.record_ids)) for c in conflicts}
    assert pairs == {
        ("note_vs_assessment_score", ("NOTE-0001", "ASM-00003")),
        ("note_vs_administration_record", ("NOTE-0002", "MAR-00001")),
    }


# --------------------------------------------------------------------------- #
# Against the generated database
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def conn(tmp_path_factory):
    path = tmp_path_factory.mktemp("tl") / "tl.db"
    write_sqlite(generate_dataset(GeneratorConfig()), path)
    c = connect_readonly(path)
    yield c
    c.close()


def test_ground_truth_defects_all_detected_without_false_flags(conn):
    s = summarize(evaluate_all(conn))
    assert s["missed"] == []
    assert s["recall"] == 1.0
    assert s["flags_on_clean_bookings"] == 0
    assert s["spurious_conflicts"] == 0


def test_time_window_filtering(conn):
    full = build_patient_timeline(conn, "P015")
    last12 = build_patient_timeline(conn, "P015", hours=12)
    assert last12.window.hours == 12.0 and last12.window.end == full.window.end
    for a in last12.assessments:
        assert last12.window.start <= a.AssessmentDateTime <= last12.window.end
    assert {a.AssessmentID for a in last12.assessments} < {a.AssessmentID for a in full.assessments}
    assert all(last12.window.start <= m.ScheduledDateTime <= last12.window.end for m in last12.administrations)
    assert [e.time for e in last12.events] == sorted(e.time for e in last12.events)


def test_patient_isolation_every_record_belongs_to_booking(conn):
    """No query returns another booking's records, for any patient."""
    for p in q.list_patients(conn):
        t = build_patient_timeline(conn, p.PatientID)
        raw = {
            "withdrawal_assessments": ("AssessmentID", {a.AssessmentID for a in t.assessments}),
            "vital_signs": ("VitalID", {v.VitalID for v in t.vitals}),
            "medication_orders": ("OrderID", {o.OrderID for o in t.orders}),
            "medication_administrations": ("AdministrationID", {m.AdministrationID for m in t.administrations}),
            "nursing_notes": ("NoteID", {n.NoteID for n in t.notes}),
        }
        for table, (col, ids) in raw.items():
            expected = {r[0] for r in conn.execute(
                f"SELECT {col} FROM {table} WHERE PatientID = ? AND BookingNumber = ?", (p.PatientID, p.BookingNumber))}
            assert ids == expected, (p.PatientID, table)


def test_booking_of_other_patient_is_not_found(conn):
    with pytest.raises(q.RecordNotFound):
        q.get_patient_booking(conn, "P001", "B1002")
    with pytest.raises(q.RecordNotFound):
        q.get_patient_booking(conn, "P999")


@pytest.mark.parametrize("bad", ["1", "P1", "P001; DROP TABLE x", "' OR 1=1 --", "p001", ""])
def test_invalid_patient_ids_rejected(conn, bad):
    with pytest.raises(q.InvalidIdentifier):
        q.get_patient_booking(conn, bad)


def test_invalid_window_timestamps_rejected(conn):
    b = q.get_patient_booking(conn, "P001")
    with pytest.raises(q.InvalidIdentifier):
        q.get_withdrawal_assessments(conn, b, "2026-03-01", "2026-03-02T00:00:00")


def test_list_patients(conn):
    items = q.list_patients(conn)
    assert len(items) == 100 and items[0].PatientID == "P001"
    assert {i.AssessmentType for i in items if i.WithdrawalType == "Opioid"} == {"COWS"}
    assert len(q.list_patients(conn, "Alcohol")) == 50
