"""Deterministic verification of numeric claims in generated patient text.

A small local model can write fluent sentences with wrong numbers (e.g.
copying counts from a prompt example). This module extracts specific,
checkable claims from each sentence and compares them with the computed
timeline analytics:

  * "score ... from X to Y"          -> earliest / latest recorded score
  * "latest ... score was N"         -> latest recorded score
  * "N ... assessments"              -> number of recorded assessments
  * "N ... doses ... administered"   -> count with status Administered
  * "N ... no documentation"         -> count with status Not documented
  * "<vital> ... from A to B"         -> earliest / latest recorded value

A sentence with a failed claim is replaced by a deterministic statement
(prefixed "Computed:") so the response never shows a contradicted number.
Each failure is recorded so evaluation can measure raw model accuracy.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from backend.services.schemas import PatientTimeline
from backend.services.timeline import VITAL_LABELS, display_ts

NUM_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
NUM = r"(\d+(?:\.\d+)?|" + "|".join(NUM_WORDS) + r")"
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
CITE_RE = re.compile(r"\[[^\]]*\]")

VITAL_KEYS = [
    (re.compile(r"\bsystolic\b", re.I), "SBP"),
    (re.compile(r"\bdiastolic\b", re.I), "DBP"),
    (re.compile(r"\b(pulse|heart rate)\b", re.I), "Pulse"),
    (re.compile(r"\btemperature\b", re.I), "Temperature"),
    (re.compile(r"\brespiratory rate\b|\bRR\b", re.I), "RespiratoryRate"),
    (re.compile(r"\b(oxygen saturation|spo2|o2 sat)\b", re.I), "OxygenSaturation"),
]


class FactCheckResult(BaseModel):
    section: str
    claim: str
    stated: str
    expected: str
    sentence: str


def _val(tok: str) -> float:
    tok = tok.lower()
    return float(NUM_WORDS[tok]) if tok in NUM_WORDS else float(tok)


def _close(a: float, b: float) -> bool:
    return abs(a - b) < 0.05


# --------------------------------------------------------------------------- #
# Deterministic replacement sentences
# --------------------------------------------------------------------------- #


def score_sentence(t: PatientTimeline) -> str:
    s = t.score_summary
    if s.n_assessments == 0:
        return f"Computed: no {s.assessment_type} assessments are recorded in the retrieved record for this window [CALC-SCORE]."
    if s.n_assessments == 1:
        return (f"Computed: one {s.assessment_type} assessment is recorded, score {s.latest.score} at "
                f"{display_ts(s.latest.time)} [CALC-SCORE].")
    return (f"Computed: {s.n_assessments} {s.assessment_type} assessments are recorded; the score changed from "
            f"{s.earliest.score} to {s.latest.score} ({s.change:+d} points) between {display_ts(s.earliest.time)} "
            f"and {display_ts(s.latest.time)} [CALC-SCORE].")


def medication_sentence(t: PatientTimeline) -> str:
    m = t.medication_summary
    if not m.n_scheduled_in_window:
        return "Computed: no scheduled doses are recorded in the retrieved record for this window [CALC-MEDS]."
    parts = ", ".join(f"{v} {k}" for k, v in m.status_counts.items())
    per = "; ".join(
        f"{o.order.MedicationName.replace(' (demo only)', '')}: {o.n_scheduled_in_window} scheduled, "
        f"{o.status_counts.get('Administered', 0)} Administered"
        for o in m.orders if o.n_scheduled_in_window
    )
    return f"Computed: {m.n_scheduled_in_window} scheduled dose(s) in the window — {parts} ({per}) [CALC-MEDS]."


MED_CODE_RE = re.compile(r"\b([A-Z]{1,2}-\d)\b")
RELEASE_RE = re.compile(r"\b(released|by release|at release|upon release|until release|discharg\w*)\b", re.I)


def vital_sentence(t: PatientTimeline, measure: str) -> str:
    vm = next(v for v in t.vitals_summary.measures if v.measure == measure)
    label = VITAL_LABELS[measure]
    if vm.n_recorded == 0:
        return f"Computed: {label} is not recorded in the retrieved record for this window [CALC-VITALS]."
    if vm.n_recorded == 1:
        return f"Computed: {label} was recorded once, {vm.latest.value:g} {vm.unit} [CALC-VITALS]."
    return (f"Computed: {label} changed from {vm.earliest.value:g} to {vm.latest.value:g} {vm.unit} "
            f"(range {vm.minimum.value:g}–{vm.maximum.value:g}) [CALC-VITALS].")


# --------------------------------------------------------------------------- #
# Claim checks
# --------------------------------------------------------------------------- #


def check_sentence(section: str, sentence: str, t: PatientTimeline) -> tuple[list[FactCheckResult], list[str]]:
    """Return (failures, replacement_sentences) for one sentence."""
    text = CITE_RE.sub(" ", sentence)
    low = text.lower()
    fails: list[FactCheckResult] = []
    repl: list[str] = []
    s = t.score_summary
    ms = t.medication_summary

    def fail(claim, stated, expected, replacement):
        fails.append(FactCheckResult(section=section, claim=claim, stated=str(stated), expected=str(expected), sentence=sentence))
        if replacement and replacement not in repl:
            repl.append(replacement)

    # --- scores (not in the vitals section) ---------------------------------
    if section != "vital_sign_observations" and re.search(r"\bscores?\b", low) and s.n_assessments:
        m = re.search(rf"from {NUM} to {NUM}", low)
        if m and s.n_assessments > 1 and not any(k.search(text) for k, _ in VITAL_KEYS):
            a, b = _val(m.group(1)), _val(m.group(2))
            if not (_close(a, s.earliest.score) and _close(b, s.latest.score)):
                fail("score_range", f"{a:g}→{b:g}", f"{s.earliest.score}→{s.latest.score}", score_sentence(t))
        m = re.search(rf"latest (?:recorded )?(?:[\w-]+ )?score (?:was|of|is|=) {NUM}", low)
        if m and not _close(_val(m.group(1)), s.latest.score):
            fail("latest_score", m.group(1), s.latest.score, score_sentence(t))
        m = re.search(rf"(?:earliest|first|initial) (?:recorded )?(?:[\w-]+ )?score (?:was|of|is|=) {NUM}", low)
        if m and not _close(_val(m.group(1)), s.earliest.score):
            fail("earliest_score", m.group(1), s.earliest.score, score_sentence(t))
        m = re.search(rf"\(?([+-]\d+) points?", low)
        if m and s.change is not None and int(m.group(1)) != s.change:
            fail("score_change", m.group(1), f"{s.change:+d}", score_sentence(t))
    m = re.search(rf"\b{NUM} (?:recorded )?(?:ciwa-ar |cows )?(?:withdrawal )?assessments?\b", low)
    if m and section != "medication_documentation" and not _close(_val(m.group(1)), s.n_assessments):
        fail("assessment_count", m.group(1), s.n_assessments, score_sentence(t))

    # --- medications ---------------------------------------------------------
    if section == "medication_documentation":
        m = re.search(rf"\b{NUM} (?:of \w+ )?(?:scheduled )?(?:fictional )?(?:medication )?doses? (?:are |were |was |is )?(?:documented |recorded )?(?:as )?administered", low)
        if m and not _close(_val(m.group(1)), ms.status_counts.get("Administered", 0)):
            fail("doses_administered", m.group(1), ms.status_counts.get("Administered", 0), medication_sentence(t))
        m = re.search(rf"\b{NUM} (?:scheduled )?(?:doses? )?(?:have|has|had|with) no (?:administration )?documentation", low)
        if m and not _close(_val(m.group(1)), ms.status_counts.get("Not documented", 0)):
            fail("doses_not_documented", m.group(1), ms.status_counts.get("Not documented", 0), medication_sentence(t))
        m = re.search(rf"\b{NUM} (?:scheduled )?doses? (?:in the window|scheduled|were scheduled|are scheduled)", low)
        if m and not _close(_val(m.group(1)), ms.n_scheduled_in_window):
            fail("doses_scheduled", m.group(1), ms.n_scheduled_in_window, medication_sentence(t))

    # --- per-medication counts ("AW-1 was administered 8 times", "6 doses of ... OW-1") ---
    if section in ("medication_documentation", "patient_summary"):
        orders = {MED_CODE_RE.search(o.order.MedicationName).group(1): o
                  for o in ms.orders if MED_CODE_RE.search(o.order.MedicationName)}
        for clause in re.split(r"[,;]", text):
            cl = clause.lower()
            # The verb usually governs the whole clause: "4 doses of X and 1 dose of Y were documented as administered".
            counts_administered = "administered" in cl and "scheduled" not in cl
            parts = [clause]
            if len([c for c in MED_CODE_RE.findall(clause) if c in orders]) > 1:
                split = re.split(r" and (?=\w)", clause)
                # "8 doses of X and Y" is a combined total, not 8 of X: only split when every part has its own number.
                if all(re.search(rf"\b{NUM}\b", MED_CODE_RE.sub(" ", p).lower()) for p in split):
                    parts = split
                else:
                    continue
            for part in parts:
                codes = [c for c in MED_CODE_RE.findall(part) if c in orders]
                if len(codes) != 1:
                    continue
                o = orders[codes[0]]
                pl = part.lower()
                n_admin = o.status_counts.get("Administered", 0)
                m = re.search(rf"\b{NUM} times\b", pl)
                if m and "administered" in cl and not _close(_val(m.group(1)), n_admin):
                    fail(f"administered_{codes[0]}", m.group(1), n_admin, medication_sentence(t))
                m = re.search(rf"\b{NUM} (?:scheduled )?doses? of\b", pl)
                if m:
                    expected = n_admin if counts_administered else o.n_scheduled_in_window
                    kind = "administered" if counts_administered else "scheduled"
                    if not _close(_val(m.group(1)), expected):
                        fail(f"{kind}_{codes[0]}", m.group(1), expected, medication_sentence(t))

    # --- release (a common invented detail) --------------------------------------
    released_in_window = t.booking.ReleaseDateTime is not None and t.booking.ReleaseDateTime <= t.window.end
    if RELEASE_RE.search(text) and not released_in_window and not re.search(r"\bnot (been )?released|no release", low):
        fail("release", "release mentioned", "no release recorded in the window",
             score_sentence(t) if re.search(r"\bscores?\b", low) else "")

    # --- vital signs ----------------------------------------------------------
    if section in ("vital_sign_observations", "patient_summary"):
        for clause in re.split(r"[,;]| and (?=[a-z]+ (?:bp|blood|pulse|temp|resp|oxygen))", text):
            m = re.search(rf"from {NUM} to {NUM}", clause.lower())
            if not m:
                continue
            for key, measure in VITAL_KEYS:
                if key.search(clause):
                    vm = next(v for v in t.vitals_summary.measures if v.measure == measure)
                    if vm.n_recorded < 2:
                        fail(f"vital_{measure}", f"{m.group(1)}→{m.group(2)}", "fewer than 2 recorded values",
                             vital_sentence(t, measure))
                    elif not (_close(_val(m.group(1)), vm.earliest.value) and _close(_val(m.group(2)), vm.latest.value)):
                        fail(f"vital_{measure}", f"{m.group(1)}→{m.group(2)}",
                             f"{vm.earliest.value:g}→{vm.latest.value:g}", vital_sentence(t, measure))
                    break
    return fails, repl


def fact_check_section(section: str, text: str, t: PatientTimeline) -> tuple[str, list[FactCheckResult]]:
    """Check every sentence; replace failing sentences with computed ones."""
    out: list[str] = []
    failures: list[FactCheckResult] = []
    for sent in SENTENCE_RE.split(text.strip()):
        if not sent:
            continue
        fails, repl = check_sentence(section, sent, t)
        if fails:
            failures.extend(fails)
            out.extend(r for r in repl if r not in out)
        else:
            out.append(sent)
    return " ".join(out), failures
