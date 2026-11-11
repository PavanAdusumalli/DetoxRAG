"""Pure metric functions for the DetoxRAG benchmark (unit-tested)."""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timedelta

from backend.rag.models import Chunk

# --------------------------------------------------------------------------- #
# Gold anchors
# --------------------------------------------------------------------------- #


def anchor_matches(anchor: dict, chunk: Chunk) -> bool:
    if "doc" in anchor and chunk.document_id != anchor["doc"]:
        return False
    if "rec" in anchor:
        return anchor["rec"] in chunk.recommendation_ids
    if "text" in anchor:
        return re.search(anchor["text"], chunk.text, re.I) is not None
    if "section" in anchor:
        return chunk.section_heading == anchor["section"]
    if "section_chapter" in anchor:
        return chunk.chapter == anchor["section_chapter"]
    raise ValueError(f"Unknown anchor {anchor}")


def is_relevant(chunk: Chunk, anchors: list[dict]) -> bool:
    return any(anchor_matches(a, chunk) for a in anchors)


def retrieval_metrics(retrieved: list[Chunk], anchors: list[dict], k: int) -> dict:
    """Recall@K over anchors, Precision@K over chunks, reciprocal rank, hit@1."""
    top = retrieved[:k]
    rel = [is_relevant(c, anchors) for c in top]
    covered = sum(any(anchor_matches(a, c) for c in top) for a in anchors)
    first = next((i + 1 for i, r in enumerate(rel) if r), None)
    return {
        "recall_at_k": covered / len(anchors) if anchors else None,
        "precision_at_k": sum(rel) / k,
        "rr": 1.0 / first if first else 0.0,
        "hit_at_1": 1.0 if rel and rel[0] else 0.0,
        "first_relevant_rank": first,
    }


def mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


# --------------------------------------------------------------------------- #
# Independent gold facts (deliberately NOT using backend/services/timeline.py)
# --------------------------------------------------------------------------- #

FMT = "%Y-%m-%dT%H:%M:%S"


def gold_window(conn: sqlite3.Connection, booking: str, hours: float | None) -> tuple[str, str]:
    adm = conn.execute("SELECT AdmissionDateTime FROM patient_bookings WHERE BookingNumber = ?", (booking,)).fetchone()[0]
    times = []
    for sql in (
        "SELECT AssessmentDateTime FROM withdrawal_assessments WHERE BookingNumber = ?",
        "SELECT RecordedDateTime FROM vital_signs WHERE BookingNumber = ?",
        "SELECT NoteDateTime FROM nursing_notes WHERE BookingNumber = ?",
        "SELECT ScheduledDateTime FROM medication_administrations WHERE BookingNumber = ?",
        "SELECT AdministrationDateTime FROM medication_administrations WHERE BookingNumber = ? AND AdministrationDateTime IS NOT NULL",
    ):
        times += [r[0] for r in conn.execute(sql, (booking,))]
    end = max(times)
    start = adm if hours is None else max(adm, (datetime.strptime(end, FMT) - timedelta(hours=hours)).strftime(FMT))
    return start, end


def gold_score_facts(conn: sqlite3.Connection, booking: str, hours: float | None) -> dict:
    start, end = gold_window(conn, booking, hours)
    rows = conn.execute(
        "SELECT AssessmentDateTime, Score FROM withdrawal_assessments WHERE BookingNumber = ? "
        "AND AssessmentDateTime >= ? AND AssessmentDateTime <= ? ORDER BY AssessmentDateTime",
        (booking, start, end),
    ).fetchall()
    scores = [r[1] for r in rows]
    gaps = [
        (datetime.strptime(b[0], FMT) - datetime.strptime(a[0], FMT)).total_seconds() / 3600
        for a, b in zip(rows, rows[1:])
    ]
    return {
        "window": (start, end),
        "n": len(scores),
        "earliest": scores[0] if scores else None,
        "latest": scores[-1] if scores else None,
        "change": scores[-1] - scores[0] if len(scores) > 1 else None,
        "min": min(scores) if scores else None,
        "max": max(scores) if scores else None,
        "max_interval_h": round(max(gaps), 2) if gaps else None,
    }


# --------------------------------------------------------------------------- #
# Fact extraction from raw model text
# --------------------------------------------------------------------------- #

VITAL_WORDS = re.compile(r"\b(bp|blood pressure|systolic|diastolic|pulse|heart rate|temperature|respiratory|oxygen|spo2|saturation)\b", re.I)
SENT = re.compile(r"(?<=[.!?])\s+")


def _score_sentences(text: str) -> list[str]:
    return [s for s in SENT.split(text) if re.search(r"\bscores?\b|ciwa|cows", s, re.I) and not VITAL_WORDS.search(s)]


def stated_score_facts(text: str) -> dict:
    """Extract earliest/latest/change claims about the withdrawal score."""
    out: dict = {}
    for s in _score_sentences(re.sub(r"\[[^\]]*\]", " ", text)):
        m = re.search(r"from (\d+) to (\d+)", s)
        if m and "earliest" not in out:
            out["earliest"], out["latest"] = int(m.group(1)), int(m.group(2))
        m = re.search(r"\b(latest|most recent|last)\b[^.]{0,40}?\b(\d+)\b", s, re.I)
        if m and "latest_explicit" not in out:
            out["latest_explicit"] = int(m.group(2))
        m = re.search(r"\(?([+-]\d+)\s*points?|\b(increase|decrease|change|rise|drop)d?\s+(?:of|by)\s+(\d+)", s, re.I)
        if m and "change" not in out:
            if m.group(1):
                out["change"] = int(m.group(1))
            else:
                sign = -1 if m.group(2).lower() in ("decrease", "drop") else 1
                out["change"] = sign * int(m.group(3))
    if "latest" not in out and "latest_explicit" in out:
        out["latest"] = out["latest_explicit"]
    out.pop("latest_explicit", None)
    return out


def score_fact_accuracy(stated: dict, gold: dict) -> tuple[int, int, int]:
    """Return (correct, stated, total) over the facts earliest/latest/change."""
    keys = ["earliest", "latest"] + (["change"] if gold.get("change") is not None else [])
    if not gold.get("n"):
        return 0, 0, 0
    n_stated = sum(k in stated for k in keys)
    correct = sum(stated[k] == gold[k] for k in keys if k in stated)  # extracted changes are signed
    return correct, n_stated, len(keys)


GAP_WORDS = re.compile(r"not recorded|no assessment|missing|gap|no documentation|not documented|incomplete|conflict|inconsisten|contradict", re.I)


def acknowledges_gaps(raw_fields: dict) -> bool:
    text = " ".join(raw_fields.values())
    return bool(GAP_WORDS.search(text) or re.search(r"\b(GAP|CONFLICT)-\d{2}\b", text))
