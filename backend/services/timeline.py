"""Deterministic patient timeline analytics.

All numbers the assistant reports about a patient (score changes, intervals,
counts, gaps) are computed here in Python — never by the language model.
Outputs are descriptive observations about what was *recorded*, not clinical
classifications. Thresholds used to flag long intervals or pair records are
configurable software settings, not clinical standards.
"""

from __future__ import annotations

import re
import sqlite3
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta

from backend.config import get_settings
from backend.database import queries as q
from backend.services.schemas import (
    SCALE_MAX,
    Assessment,
    Booking,
    DocumentationConflict,
    DocumentationGap,
    Interval,
    IntervalSummary,
    MedicationAdministration,
    MedicationOrder,
    MedicationSummary,
    NursingNote,
    OrderSummary,
    PatientTimeline,
    ScorePoint,
    ScoreSummary,
    TimelineEvent,
    TimeWindow,
    ValuePoint,
    VitalMeasureSummary,
    VitalsSummary,
    VitalSign,
)

TS_FMT = "%Y-%m-%dT%H:%M:%S"
VITAL_MEASURES = ["SBP", "DBP", "Pulse", "Temperature", "RespiratoryRate", "OxygenSaturation"]
VITAL_LABELS = {
    "SBP": "systolic BP", "DBP": "diastolic BP", "Pulse": "pulse", "Temperature": "temperature",
    "RespiratoryRate": "respiratory rate", "OxygenSaturation": "oxygen saturation",
}
DEFAULT_UNITS = {"SBP": "mmHg", "DBP": "mmHg", "Pulse": "beats/min", "Temperature": "°F",
                 "RespiratoryRate": "breaths/min", "OxygenSaturation": "%", "Score": "points"}

REFUSAL_RE = re.compile(r"\brefus(ed|es|al)\b", re.I)
NO_SIGNS_RE = re.compile(r"\bno (signs|symptoms) of withdrawal\b|\bno withdrawal (signs|symptoms)\b", re.I)


def parse_ts(ts: str) -> datetime:
    return datetime.strptime(ts, TS_FMT)


def fmt_ts(dt: datetime) -> str:
    return dt.strftime(TS_FMT)


def hours_between(a: str, b: str) -> float:
    return round((parse_ts(b) - parse_ts(a)).total_seconds() / 3600.0, 2)


def display_ts(ts: str) -> str:
    return ts.replace("T", " ")[:16]


@dataclass(frozen=True)
class AnalyticsConfig:
    review_interval_hours: float = 4.0
    gap_tolerance: float = 1.5
    pairing_minutes: float = 30.0
    conflict_minutes: float = 60.0

    @property
    def flag_threshold_hours(self) -> float:
        return round(self.review_interval_hours * self.gap_tolerance, 2)

    @classmethod
    def from_settings(cls) -> "AnalyticsConfig":
        s = get_settings()
        return cls(s.review_interval_hours, s.gap_tolerance, s.pairing_minutes, s.conflict_minutes)


# --------------------------------------------------------------------------- #
# Time window
# --------------------------------------------------------------------------- #


def resolve_window(
    booking: Booking,
    latest_event: str | None,
    hours: float | None = None,
    start: str | None = None,
    end: str | None = None,
) -> TimeWindow:
    """Resolve the analysis window.

    Synthetic data has no meaningful "now", so by default the window ends at the
    booking's latest recorded event. `hours` gives "last N hours" ending there;
    explicit `start`/`end` override. The start never precedes admission.
    """
    admission = booking.AdmissionDateTime
    if end is not None:
        q.validate_timestamp(end)
        end_ref = "requested end time"
    else:
        end = latest_event or admission
        end_ref = "latest recorded event for this booking"
    if start is not None:
        q.validate_timestamp(start)
    elif hours is not None:
        if hours <= 0:
            raise q.InvalidIdentifier("hours must be positive")
        start = fmt_ts(parse_ts(end) - timedelta(hours=hours))
    else:
        start = admission
    start = max(start, admission)
    if start > end:
        raise q.InvalidIdentifier(f"Window start {start} is after end {end}")
    span = hours_between(start, end)
    if hours is not None and start == admission and span < hours:
        label = f"Last {hours:g} hours requested; record begins at admission ({span:g} h available)"
    elif hours is not None:
        label = f"Last {hours:g} hours of recorded monitoring"
    else:
        label = "Entire recorded booking" if start == admission and end_ref.startswith("latest") else "Requested window"
    return TimeWindow(start=start, end=end, hours=span, label=label, end_reference=end_ref)


# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #


def calculate_score_changes(assessments: list[Assessment], assessment_type: str) -> ScoreSummary:
    pts = sorted(assessments, key=lambda a: (a.AssessmentDateTime, a.AssessmentID))
    sp = [ScorePoint(assessment_id=a.AssessmentID, time=a.AssessmentDateTime, score=a.Score) for a in pts]
    scale_max = SCALE_MAX[assessment_type]
    if not sp:
        return ScoreSummary(
            assessment_type=assessment_type, scale_max=scale_max, n_assessments=0, earliest=None, latest=None,
            change=None, minimum=None, maximum=None, net_direction="no assessments recorded",
            n_increases=0, n_decreases=0, direction_changes=0, observation_span_hours=None,
        )
    diffs = [b.score - a.score for a, b in zip(sp, sp[1:])]
    signs = [d > 0 for d in diffs if d != 0]
    change = sp[-1].score - sp[0].score
    if len(sp) == 1:
        direction = "single assessment"
    else:
        direction = "increasing" if change > 0 else "decreasing" if change < 0 else "unchanged"
    return ScoreSummary(
        assessment_type=assessment_type,
        scale_max=scale_max,
        n_assessments=len(sp),
        earliest=sp[0],
        latest=sp[-1],
        change=change if len(sp) > 1 else None,
        minimum=min(sp, key=lambda p: p.score),  # first occurrence on ties
        maximum=max(sp, key=lambda p: p.score),
        net_direction=direction,
        n_increases=sum(d > 0 for d in diffs),
        n_decreases=sum(d < 0 for d in diffs),
        direction_changes=sum(a != b for a, b in zip(signs, signs[1:])),
        observation_span_hours=hours_between(sp[0].time, sp[-1].time) if len(sp) > 1 else 0.0,
    )


# --------------------------------------------------------------------------- #
# Intervals
# --------------------------------------------------------------------------- #


def calculate_assessment_intervals(
    assessments: list[Assessment], window: TimeWindow, cfg: AnalyticsConfig
) -> IntervalSummary:
    pts = sorted(assessments, key=lambda a: a.AssessmentDateTime)
    intervals = [
        Interval(from_id=a.AssessmentID, to_id=b.AssessmentID, start=a.AssessmentDateTime,
                 end=b.AssessmentDateTime, hours=hours_between(a.AssessmentDateTime, b.AssessmentDateTime))
        for a, b in zip(pts, pts[1:])
    ]
    hrs = [i.hours for i in intervals]
    thr = cfg.flag_threshold_hours
    leading = trailing = None
    if pts:
        lead_h = hours_between(window.start, pts[0].AssessmentDateTime)
        if lead_h > thr:
            leading = Interval(from_id=None, to_id=pts[0].AssessmentID, start=window.start,
                               end=pts[0].AssessmentDateTime, hours=lead_h)
        trail_h = hours_between(pts[-1].AssessmentDateTime, window.end)
        if trail_h > thr:
            trailing = Interval(from_id=pts[-1].AssessmentID, to_id=None, start=pts[-1].AssessmentDateTime,
                                end=window.end, hours=trail_h)
    return IntervalSummary(
        n_intervals=len(intervals),
        intervals=intervals,
        min_hours=min(hrs) if hrs else None,
        max_hours=max(hrs) if hrs else None,
        mean_hours=round(statistics.mean(hrs), 2) if hrs else None,
        median_hours=round(statistics.median(hrs), 2) if hrs else None,
        configured_review_interval_hours=cfg.review_interval_hours,
        flag_threshold_hours=thr,
        long_intervals=[i for i in intervals if i.hours > thr],
        leading_gap=leading,
        trailing_gap=trailing,
    )


# --------------------------------------------------------------------------- #
# Vital signs
# --------------------------------------------------------------------------- #


def summarize_vital_signs(vitals: list[VitalSign], units: dict[str, str] | None = None) -> VitalsSummary:
    units = units or DEFAULT_UNITS
    vs = sorted(vitals, key=lambda v: (v.RecordedDateTime, v.VitalID))
    measures = []
    for m in VITAL_MEASURES:
        pts = [ValuePoint(record_id=v.VitalID, time=v.RecordedDateTime, value=float(getattr(v, m)))
               for v in vs if getattr(v, m) is not None]
        measures.append(VitalMeasureSummary(
            measure=m,
            unit=units.get(m, DEFAULT_UNITS[m]),
            n_recorded=len(pts),
            n_not_recorded=len(vs) - len(pts),
            earliest=pts[0] if pts else None,
            latest=pts[-1] if pts else None,
            minimum=min(pts, key=lambda p: p.value) if pts else None,
            maximum=max(pts, key=lambda p: p.value) if pts else None,
            change=round(pts[-1].value - pts[0].value, 1) if len(pts) > 1 else None,
        ))
    return VitalsSummary(n_sets=len(vs), measures=measures)


# --------------------------------------------------------------------------- #
# Medications
# --------------------------------------------------------------------------- #


def summarize_medications(
    orders: list[MedicationOrder], administrations: list[MedicationAdministration]
) -> MedicationSummary:
    def counts(rows):
        c: dict[str, int] = {}
        for r in rows:
            c[r.AdministrationStatus] = c.get(r.AdministrationStatus, 0) + 1
        return dict(sorted(c.items()))

    per_order = [
        OrderSummary(order=o, n_scheduled_in_window=len(rows := [a for a in administrations if a.OrderID == o.OrderID]),
                     status_counts=counts(rows))
        for o in orders
    ]
    return MedicationSummary(
        n_scheduled_in_window=len(administrations),
        status_counts=counts(administrations),
        orders=per_order,
        not_administered=[a for a in administrations if a.AdministrationStatus != "Administered"],
    )


# --------------------------------------------------------------------------- #
# Documentation gaps & conflicts
# --------------------------------------------------------------------------- #


def _near(a: str, b: str, minutes: float) -> bool:
    return abs((parse_ts(a) - parse_ts(b)).total_seconds()) <= minutes * 60


def identify_documentation_gaps(
    assessments: list[Assessment],
    vitals: list[VitalSign],
    administrations: list[MedicationAdministration],
    intervals: IntervalSummary,
    window: TimeWindow,
    cfg: AnalyticsConfig,
) -> list[DocumentationGap]:
    gaps: list[DocumentationGap] = []
    thr = intervals.flag_threshold_hours
    basis = (f"longer than {thr:g} h ({cfg.gap_tolerance:g} × the configured review interval of "
             f"{cfg.review_interval_hours:g} h)")

    if not assessments:
        gaps.append(DocumentationGap(
            gap_type="no_assessments_in_window",
            description=f"No withdrawal assessments are recorded in the retrieved dataset between "
                        f"{display_ts(window.start)} and {display_ts(window.end)}.",
            start=window.start, end=window.end))
    if intervals.leading_gap:
        g = intervals.leading_gap
        gaps.append(DocumentationGap(
            gap_type="no_assessment_at_window_start",
            description=f"No assessment recorded in the first {g.hours:g} h of the window "
                        f"({display_ts(g.start)} to {display_ts(g.end)}), {basis}.",
            start=g.start, end=g.end, record_ids=[g.to_id]))
    for i in intervals.long_intervals:
        gaps.append(DocumentationGap(
            gap_type="long_assessment_interval",
            description=f"{i.hours:g} h between recorded assessments {i.from_id} ({display_ts(i.start)}) and "
                        f"{i.to_id} ({display_ts(i.end)}), {basis}.",
            start=i.start, end=i.end, record_ids=[i.from_id, i.to_id]))
    if intervals.trailing_gap:
        g = intervals.trailing_gap
        gaps.append(DocumentationGap(
            gap_type="no_assessment_at_window_end",
            description=f"No assessment recorded in the last {g.hours:g} h of the window: the latest recorded "
                        f"assessment is {g.from_id} at {display_ts(g.start)}, while other records continue to "
                        f"{display_ts(g.end)}.",
            start=g.start, end=g.end, record_ids=[g.from_id]))

    # Assessments and vital-sign sets are usually documented together; flag unpaired ones.
    for a in assessments:
        if not any(_near(a.AssessmentDateTime, v.RecordedDateTime, cfg.pairing_minutes) for v in vitals):
            gaps.append(DocumentationGap(
                gap_type="assessment_without_vital_signs",
                description=f"Assessment {a.AssessmentID} at {display_ts(a.AssessmentDateTime)} has no vital-sign "
                            f"set recorded within {cfg.pairing_minutes:g} minutes.",
                start=a.AssessmentDateTime, end=a.AssessmentDateTime, record_ids=[a.AssessmentID]))
    for v in vitals:
        if not any(_near(v.RecordedDateTime, a.AssessmentDateTime, cfg.pairing_minutes) for a in assessments):
            gaps.append(DocumentationGap(
                gap_type="vital_signs_without_assessment",
                description=f"Vital signs {v.VitalID} recorded at {display_ts(v.RecordedDateTime)} with no withdrawal "
                            f"assessment recorded within {cfg.pairing_minutes:g} minutes.",
                start=v.RecordedDateTime, end=v.RecordedDateTime, record_ids=[v.VitalID]))
        missing = [VITAL_LABELS[m] for m in VITAL_MEASURES if getattr(v, m) is None]
        if missing:
            gaps.append(DocumentationGap(
                gap_type="missing_vital_measurement",
                description=f"Vital-sign set {v.VitalID} at {display_ts(v.RecordedDateTime)}: "
                            f"{', '.join(missing)} not recorded in the retrieved dataset.",
                start=v.RecordedDateTime, end=v.RecordedDateTime, record_ids=[v.VitalID]))

    for m in administrations:
        if m.AdministrationStatus == "Not documented":
            gaps.append(DocumentationGap(
                gap_type="medication_not_documented",
                description=f"Scheduled dose {m.AdministrationID} of {m.MedicationName} at "
                            f"{display_ts(m.ScheduledDateTime)} has no administration documentation "
                            f"(status unknown; not the same as 'Not administered').",
                start=m.ScheduledDateTime, end=m.ScheduledDateTime, record_ids=[m.AdministrationID]))

    return sorted(gaps, key=lambda g: (g.start or "", g.gap_type))


def identify_documentation_conflicts(
    assessments: list[Assessment],
    administrations: list[MedicationAdministration],
    notes: list[NursingNote],
    cfg: AnalyticsConfig,
) -> list[DocumentationConflict]:
    """Rule-based flags for records that may contradict each other.

    These are prompts for human review, not determinations of which record is
    correct. Rules:
      * a note mentioning a refusal near a dose documented as 'Administered';
      * a note stating no withdrawal signs near an assessment in the upper third
        of this patient's recorded score range in the window.
    """
    out: list[DocumentationConflict] = []
    for n in notes:
        if REFUSAL_RE.search(n.NoteText):
            for m in administrations:
                if (m.AdministrationStatus == "Administered" and m.AdministrationDateTime
                        and _near(n.NoteDateTime, m.AdministrationDateTime, cfg.conflict_minutes)):
                    out.append(DocumentationConflict(
                        conflict_type="note_vs_administration_record",
                        description=f"Note {n.NoteID} ({display_ts(n.NoteDateTime)}) mentions a refusal, but dose "
                                    f"{m.AdministrationID} of {m.MedicationName} is documented as Administered at "
                                    f"{display_ts(m.AdministrationDateTime)}.",
                        time=n.NoteDateTime, record_ids=[n.NoteID, m.AdministrationID]))
    if assessments:
        scores = [a.Score for a in assessments]
        lo, hi = min(scores), max(scores)
        upper = lo + (hi - lo) * 2 / 3
        for n in notes:
            if NO_SIGNS_RE.search(n.NoteText):
                for a in assessments:
                    if hi > lo and a.Score >= upper and _near(n.NoteDateTime, a.AssessmentDateTime, cfg.conflict_minutes):
                        out.append(DocumentationConflict(
                            conflict_type="note_vs_assessment_score",
                            description=f"Note {n.NoteID} ({display_ts(n.NoteDateTime)}) states no signs of withdrawal, "
                                        f"while assessment {a.AssessmentID} at {display_ts(a.AssessmentDateTime)} "
                                        f"recorded {a.AssessmentType} {a.Score}, in the upper third of this "
                                        f"patient's recorded range in the window ({lo}–{hi}).",
                            time=n.NoteDateTime, record_ids=[n.NoteID, a.AssessmentID]))
    return sorted(out, key=lambda c: c.time)


# --------------------------------------------------------------------------- #
# Chronology
# --------------------------------------------------------------------------- #


def _fmt_vitals(v: VitalSign) -> str:
    def f(x, suffix=""):
        return "not recorded" if x is None else f"{x}{suffix}"
    bp = "not recorded" if v.SBP is None or v.DBP is None else f"{v.SBP}/{v.DBP} mmHg"
    return (f"BP {bp}, pulse {f(v.Pulse, ' /min')}, temp {f(v.Temperature, ' °F')}, "
            f"RR {f(v.RespiratoryRate, ' /min')}, SpO2 {f(v.OxygenSaturation, '%')}")


def build_events(assessments, vitals, administrations, notes) -> list[TimelineEvent]:
    ev = [TimelineEvent(time=a.AssessmentDateTime, kind="assessment", record_id=a.AssessmentID,
                        summary=f"{a.AssessmentType} score {a.Score}") for a in assessments]
    ev += [TimelineEvent(time=v.RecordedDateTime, kind="vitals", record_id=v.VitalID, summary=_fmt_vitals(v))
           for v in vitals]
    ev += [TimelineEvent(
        time=m.ScheduledDateTime, kind="medication", record_id=m.AdministrationID,
        summary=f"{m.MedicationName} scheduled; status {m.AdministrationStatus}"
                + (f" (documented {display_ts(m.AdministrationDateTime)})" if m.AdministrationDateTime else ""))
        for m in administrations]
    ev += [TimelineEvent(time=n.NoteDateTime, kind="note", record_id=n.NoteID, summary=n.NoteText) for n in notes]
    order = {"assessment": 0, "vitals": 1, "medication": 2, "note": 3}
    return sorted(ev, key=lambda e: (e.time, order[e.kind], e.record_id))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def build_patient_timeline(
    conn: sqlite3.Connection,
    patient_id: str,
    booking_number: str | None = None,
    hours: float | None = None,
    start: str | None = None,
    end: str | None = None,
    cfg: AnalyticsConfig | None = None,
) -> PatientTimeline:
    cfg = cfg or AnalyticsConfig.from_settings()
    booking = q.get_patient_booking(conn, patient_id, booking_number)
    window = resolve_window(booking, q.get_latest_event_time(conn, booking), hours, start, end)
    s, e = window.start, window.end

    assessments = q.get_withdrawal_assessments(conn, booking, s, e)
    vitals = q.get_vital_signs(conn, booking, s, e)
    orders = q.get_medication_orders(conn, booking, s, e)
    administrations = q.get_medication_administrations(conn, booking, s, e)
    notes = q.get_nursing_notes(conn, booking, s, e)
    units = q.get_measurement_units(conn) or DEFAULT_UNITS

    intervals = calculate_assessment_intervals(assessments, window, cfg)
    return PatientTimeline(
        booking=booking,
        window=window,
        units=units,
        assessments=assessments,
        vitals=vitals,
        orders=orders,
        administrations=administrations,
        notes=notes,
        score_summary=calculate_score_changes(assessments, booking.assessment_type),
        intervals=intervals,
        vitals_summary=summarize_vital_signs(vitals, units),
        medication_summary=summarize_medications(orders, administrations),
        documentation_gaps=identify_documentation_gaps(assessments, vitals, administrations, intervals, window, cfg),
        documentation_conflicts=identify_documentation_conflicts(assessments, administrations, notes, cfg),
        events=build_events(assessments, vitals, administrations, notes),
    )
