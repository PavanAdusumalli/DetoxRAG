"""Build the evidence package handed to the language model.

Two separate kinds of evidence:

* **Patient evidence** — records retrieved from SQLite (cited by their record
  IDs, e.g. ASM-00148) plus deterministic computed facts from
  `timeline.py` (cited as CALC-*, GAP-nn, CONFLICT-nn). The model is told to
  use CALC facts for any number, so it never has to calculate.
* **Clinical evidence** — guideline passages from hybrid RAG (cited by chunk
  ID, e.g. BJA-2023-0071).

Nursing notes and passages are rendered inside tags and treated as untrusted
data; tag-like text inside them is neutralized.
"""

from __future__ import annotations

import re
import sqlite3

from pydantic import BaseModel, Field

from backend.rag.models import RetrievedPassage
from backend.rag.retrieval import Retriever
from backend.services.schemas import PatientTimeline
from backend.services.timeline import VITAL_LABELS, build_patient_timeline, display_ts


class EvidenceItem(BaseModel):
    evidence_id: str
    kind: str  # computed | gap | conflict | assessment | vitals | medication_order | medication_administration | note
    time: str | None = None
    text: str


class EvidencePackage(BaseModel):
    patient_id: str
    booking_number: str
    withdrawal_type: str
    monitoring_type: str
    question: str
    timeline: PatientTimeline
    patient_evidence: list[EvidenceItem]
    guideline_passages: list[RetrievedPassage] = Field(default_factory=list)
    guideline_query: str | None = None
    retrieval_mode: str = "hybrid"

    @property
    def patient_evidence_ids(self) -> set[str]:
        return {e.evidence_id for e in self.patient_evidence}

    @property
    def guideline_ids(self) -> set[str]:
        return {p.citation_id for p in self.guideline_passages}


# --------------------------------------------------------------------------- #
# Computed facts
# --------------------------------------------------------------------------- #


def _signed(x: float | int | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:+g}"


def computed_facts(t: PatientTimeline) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    s = t.score_summary
    w = t.window
    items.append(EvidenceItem(
        evidence_id="CALC-WINDOW", kind="computed",
        text=f"Review window: {display_ts(w.start)} to {display_ts(w.end)} ({w.hours:g} h; {w.label}). "
             f"Admission {display_ts(t.booking.AdmissionDateTime)}; release "
             f"{display_ts(t.booking.ReleaseDateTime) if t.booking.ReleaseDateTime else 'not recorded (in custody)'}."))
    if s.n_assessments == 0:
        score_text = f"No {s.assessment_type} assessments are recorded in the retrieved dataset for this window."
    else:
        score_text = (
            f"{s.n_assessments} {s.assessment_type} assessment(s) recorded (scale 0–{s.scale_max}). "
            f"Earliest: {s.earliest.score} at {display_ts(s.earliest.time)} ({s.earliest.assessment_id}). "
            f"Latest: {s.latest.score} at {display_ts(s.latest.time)} ({s.latest.assessment_id}). "
        )
        if s.n_assessments > 1:
            score_text += (
                f"Change earliest to latest: {_signed(s.change)} points over {s.observation_span_hours:g} h. "
                f"Lowest {s.minimum.score} ({s.minimum.assessment_id}); highest {s.maximum.score} ({s.maximum.assessment_id}). "
                f"Recorded net direction: {s.net_direction}; {s.n_increases} step increase(s), "
                f"{s.n_decreases} step decrease(s), {s.direction_changes} reversal(s) of direction."
            )
    items.append(EvidenceItem(evidence_id="CALC-SCORE", kind="computed", text=score_text))

    iv = t.intervals
    if iv.n_intervals:
        items.append(EvidenceItem(
            evidence_id="CALC-INTERVALS", kind="computed",
            text=f"Intervals between recorded assessments: median {iv.median_hours:g} h, shortest {iv.min_hours:g} h, "
                 f"longest {iv.max_hours:g} h. {len(iv.long_intervals)} interval(s) exceed the {iv.flag_threshold_hours:g} h "
                 f"review flag (a software setting, not a clinical standard)."))

    vs = t.vitals_summary
    parts = []
    for m in vs.measures:
        label = VITAL_LABELS[m.measure]
        if m.n_recorded == 0:
            parts.append(f"{label}: not recorded")
            continue
        p = f"{label} ({m.unit}): earliest {m.earliest.value:g}, latest {m.latest.value:g}"
        if m.change is not None:
            p += f", change {_signed(m.change)}, range {m.minimum.value:g}–{m.maximum.value:g}"
        if m.n_not_recorded:
            p += f", not recorded in {m.n_not_recorded} set(s)"
        parts.append(p)
    items.append(EvidenceItem(
        evidence_id="CALC-VITALS", kind="computed",
        text=f"{vs.n_sets} vital-sign set(s) recorded. " + "; ".join(parts) + "." if vs.n_sets
        else "No vital-sign sets are recorded in the retrieved dataset for this window."))

    ms = t.medication_summary
    if ms.n_scheduled_in_window:
        status = ", ".join(f"{k}: {v}" for k, v in ms.status_counts.items())
        orders = "; ".join(f"{o.order.MedicationName} ({o.order.OrderID}, order status {o.order.OrderStatus}, "
                           f"{o.n_scheduled_in_window} scheduled)" for o in ms.orders)
        items.append(EvidenceItem(
            evidence_id="CALC-MEDS", kind="computed",
            text=f"{ms.n_scheduled_in_window} scheduled dose(s) in the window. Documented status counts — {status}. "
                 f"Orders: {orders}. Medication names are fictional demonstration examples."))
    else:
        items.append(EvidenceItem(evidence_id="CALC-MEDS", kind="computed",
                                  text="No scheduled medication doses are recorded in the retrieved dataset for this window."))

    for i, g in enumerate(t.documentation_gaps, 1):
        items.append(EvidenceItem(evidence_id=f"GAP-{i:02d}", kind="gap", time=g.start, text=g.description))
    for i, c in enumerate(t.documentation_conflicts, 1):
        items.append(EvidenceItem(evidence_id=f"CONFLICT-{i:02d}", kind="conflict", time=c.time,
                                  text=f"Possible documentation conflict (for review, not resolved): {c.description}"))
    return items


def record_items(t: PatientTimeline) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for a in t.assessments:
        items.append(EvidenceItem(evidence_id=a.AssessmentID, kind="assessment", time=a.AssessmentDateTime,
                                  text=f"{display_ts(a.AssessmentDateTime)} {a.AssessmentType} total score {a.Score}"))
    for e in t.events:
        if e.kind == "vitals":
            items.append(EvidenceItem(evidence_id=e.record_id, kind="vitals", time=e.time,
                                      text=f"{display_ts(e.time)} vital signs: {e.summary}"))
    for o in t.orders:
        items.append(EvidenceItem(
            evidence_id=o.OrderID, kind="medication_order", time=o.OrderStartDateTime,
            text=f"Order {o.MedicationName}: {display_ts(o.OrderStartDateTime)} to "
                 f"{display_ts(o.OrderStopDateTime) if o.OrderStopDateTime else 'no stop time'}, status {o.OrderStatus}"))
    for m in t.administrations:
        doc = f"documented {display_ts(m.AdministrationDateTime)}" if m.AdministrationDateTime else "no documentation"
        items.append(EvidenceItem(
            evidence_id=m.AdministrationID, kind="medication_administration", time=m.ScheduledDateTime,
            text=f"{m.MedicationName} scheduled {display_ts(m.ScheduledDateTime)}: {m.AdministrationStatus} ({doc})"))
    for n in t.notes:
        items.append(EvidenceItem(evidence_id=n.NoteID, kind="note", time=n.NoteDateTime, text=n.NoteText))
    return sorted(items, key=lambda x: (x.time or "", x.evidence_id))


# --------------------------------------------------------------------------- #
# Guideline query
# --------------------------------------------------------------------------- #


def build_guideline_query(question: str, withdrawal_type: str, monitoring_type: str) -> str:
    """Deterministic retrieval query: the user's question plus the patient's
    monitoring context (no patient values, so nothing identifying is embedded)."""
    return f"{question.strip()} ({withdrawal_type.lower()} withdrawal; {monitoring_type} monitoring and reassessment)"


# --------------------------------------------------------------------------- #
# Assembly & rendering
# --------------------------------------------------------------------------- #


def build_evidence_package(
    conn: sqlite3.Connection,
    patient_id: str,
    question: str,
    booking_number: str | None = None,
    hours: float | None = None,
    retriever: Retriever | None = None,
    retrieval_mode: str = "hybrid",
    k: int = 4,
) -> EvidencePackage:
    t = build_patient_timeline(conn, patient_id, booking_number, hours=hours)
    b = t.booking
    passages: list[RetrievedPassage] = []
    gq = None
    if retriever is not None and retrieval_mode != "none":
        gq = build_guideline_query(question, b.WithdrawalType, b.assessment_type)
        passages = retriever.search(gq, k)
    return EvidencePackage(
        patient_id=b.PatientID,
        booking_number=b.BookingNumber,
        withdrawal_type=b.WithdrawalType,
        monitoring_type=b.assessment_type,
        question=question,
        timeline=t,
        patient_evidence=computed_facts(t) + record_items(t),
        guideline_passages=passages,
        guideline_query=gq,
        retrieval_mode=retrieval_mode if retriever is not None else "none",
    )


def _neutralize(text: str) -> str:
    return re.sub(r"</?\s*(note|passage|patient_evidence|guideline_passages)\b[^>]*>", "[tag removed]", text, flags=re.I)


def _redact_other_identifiers(text: str, pkg: EvidencePackage) -> str:
    """Free-text notes could mention someone else's identifier; never pass it on."""
    keep = {pkg.patient_id, pkg.booking_number}
    return re.sub(r"\b(P\d{3,6}|B\d{4,8})\b", lambda m: m.group(0) if m.group(0) in keep else "[identifier removed]", text)


def render_patient_evidence(pkg: EvidencePackage) -> str:
    lines = [
        f'<patient_evidence patient_id="{pkg.patient_id}" booking_number="{pkg.booking_number}" '
        f'withdrawal_type="{pkg.withdrawal_type}" monitoring_scale="{pkg.monitoring_type}">',
        "Computed facts (deterministic; use these for all numbers):",
    ]
    for e in pkg.patient_evidence:
        if e.kind in ("computed", "gap", "conflict"):
            lines.append(f"[{e.evidence_id}] {e.text}")
    lines.append("Records (chronological):")
    for e in pkg.patient_evidence:
        if e.kind == "note":
            text = _redact_other_identifiers(_neutralize(e.text), pkg)
            lines.append(f'<note id="{e.evidence_id}" time="{display_ts(e.time)}">{text}</note>')
        elif e.kind not in ("computed", "gap", "conflict"):
            lines.append(f"[{e.evidence_id}] {e.text}")
    lines.append("</patient_evidence>")
    return "\n".join(lines)


def render_guideline_passages(pkg: EvidencePackage) -> str:
    if not pkg.guideline_passages:
        return "<guideline_passages>\nNo guideline passages were retrieved for this request.\n</guideline_passages>"
    blocks = ["<guideline_passages>"]
    for p in pkg.guideline_passages:
        c = p.chunk
        blocks.append(f'<passage id="{c.chunk_id}" source="{c.document_id}" section="{c.section_path}" '
                      f'pages="{c.page_display()}">\n{_neutralize(c.text)}\n</passage>')
    blocks.append("</guideline_passages>")
    return "\n".join(blocks)
