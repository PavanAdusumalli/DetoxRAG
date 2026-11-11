"""Score the deterministic analytics against the generator's injected defects.

`synthetic_ground_truth` records exactly which documentation problems were
planted for each booking. This module turns those into expected findings and
checks whether `build_patient_timeline` (full booking window) reports them.

Used by tests (tests/test_timeline.py) and by the Phase 8 evaluation.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

from backend.services.schemas import PatientTimeline
from backend.services.timeline import _near, build_patient_timeline

CLEAN_SCENARIOS = {"increasing", "decreasing", "fluctuating", "stable"}


@dataclass
class BookingResult:
    patient_id: str
    booking_number: str
    scenario: str
    expected: list[tuple[str, str]] = field(default_factory=list)  # (finding_type, key)
    detected: list[tuple[str, str]] = field(default_factory=list)
    missed: list[tuple[str, str]] = field(default_factory=list)
    flags_raised: int = 0  # gaps + conflicts reported by the analytics
    spurious_conflicts: int = 0


def _gap_ids(t: PatientTimeline, gap_type: str) -> set[str]:
    return {rid for g in t.documentation_gaps if g.gap_type == gap_type for rid in g.record_ids}


def evaluate_booking(t: PatientTimeline, scenario: str, details: dict) -> BookingResult:
    r = BookingResult(t.booking.PatientID, t.booking.BookingNumber, scenario)
    gaps, conflicts = t.documentation_gaps, t.documentation_conflicts
    r.flags_raised = len(gaps) + len(conflicts)

    def expect(kind: str, key: str, found: bool) -> None:
        r.expected.append((kind, key))
        (r.detected if found else r.missed).append((kind, key))

    if scenario == "missing_assessments":
        for ts in details["expected_but_unrecorded_times"]:
            found = any(g.gap_type == "vital_signs_without_assessment" and _near(g.start, ts, 30) for g in gaps) or any(
                g.gap_type == "long_assessment_interval" and g.start < ts < g.end for g in gaps)
            expect("unrecorded_assessment", ts, found)
    elif scenario == "missing_vitals":
        for ts in details["unrecorded_vital_sets_near"]:
            expect("unrecorded_vital_set", ts, any(
                g.gap_type == "assessment_without_vital_signs" and _near(g.start, ts, 30) for g in gaps))
        null_ids = _gap_ids(t, "missing_vital_measurement")
        for item in details["null_measurements"]:
            expect("null_measurement", item["vital"], item["vital"] in null_ids)
    elif scenario == "irregular_intervals":
        need = details["long_gap_hours"] - 0.5
        expect("long_interval", f"{details['long_gap_hours']}h",
               any(i.hours >= need for i in t.intervals.long_intervals))
    elif scenario == "medication_documentation_gaps":
        undocumented = _gap_ids(t, "medication_not_documented")
        non_admin = {m.AdministrationID for m in t.medication_summary.not_administered}
        for ch in details["medication_status_changes"]:
            if ch["status"] == "Not documented":
                expect("undocumented_dose", ch["administration"], ch["administration"] in undocumented)
            else:
                expect(f"dose_{ch['status'].lower().replace(' ', '_')}", ch["administration"],
                       ch["administration"] in non_admin)
    elif scenario == "conflicting_documentation":
        found_pairs = {frozenset(c.record_ids) for c in conflicts}
        for c in details["conflicts"]:
            ids = frozenset(v for k, v in c.items() if k in ("note", "assessment", "administration"))
            expect(c["type"], "+".join(sorted(ids)), ids in found_pairs)
    elif scenario == "incomplete_monitoring":
        if not details["released_early"]:
            expect("records_stop_early", t.booking.BookingNumber, t.intervals.trailing_gap is not None)
        undocumented = _gap_ids(t, "medication_not_documented")
        for rid in details.get("undocumented_after_records_end", []):
            expect("undocumented_dose", rid, rid in undocumented)

    if scenario != "conflicting_documentation":
        r.spurious_conflicts = len(conflicts)
    return r


def evaluate_all(conn: sqlite3.Connection) -> list[BookingResult]:
    rows = conn.execute(
        "SELECT PatientID, BookingNumber, Scenario, Details FROM synthetic_ground_truth ORDER BY PatientID"
    ).fetchall()
    out = []
    for pid, bk, scenario, details in rows:
        t = build_patient_timeline(conn, pid, bk)
        out.append(evaluate_booking(t, scenario, json.loads(details)))
    return out


def summarize(results: list[BookingResult]) -> dict:
    by_type: dict[str, dict[str, int]] = {}
    for r in results:
        for kind, _ in r.expected:
            by_type.setdefault(kind, {"expected": 0, "detected": 0})["expected"] += 1
        for kind, _ in r.detected:
            by_type[kind]["detected"] += 1
    total_e = sum(v["expected"] for v in by_type.values())
    total_d = sum(v["detected"] for v in by_type.values())
    clean = [r for r in results if r.scenario in CLEAN_SCENARIOS]
    return {
        "injected_defects": total_e,
        "detected": total_d,
        "recall": round(total_d / total_e, 4) if total_e else None,
        "by_type": {k: {**v, "recall": round(v["detected"] / v["expected"], 4)} for k, v in sorted(by_type.items())},
        "clean_bookings": len(clean),
        "flags_on_clean_bookings": sum(r.flags_raised for r in clean),
        "spurious_conflicts": sum(r.spurious_conflicts for r in results),
        "missed": [(r.patient_id, kind, key) for r in results for kind, key in r.missed],
    }
