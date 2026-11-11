"""Evaluation metric functions and gold-label integrity."""

from __future__ import annotations

import json

import pytest

from backend.config import PROJECT_ROOT, get_settings
from evaluation.metrics import (
    acknowledges_gaps,
    anchor_matches,
    gold_score_facts,
    retrieval_metrics,
    score_fact_accuracy,
    stated_score_facts,
)
from tests.conftest import make_chunk, requires_index

A13 = make_chunk("BJA-2023-0071", "A-13. Assess at least every 8 hours.", recommendation_ids=["A-13"])
APPX = make_chunk("BJA-2023-0151", "Re-assessment ... at least every 8 hours during alcohol withdrawal management.",
                  chapter="Appendix E: Monitoring Patients During Withdrawal Management")
OTHER = make_chunk("BJA-2023-0001", "Unrelated introduction text.")
BOP = make_chunk("BOP-2020-0020", "Thiamine text", section_heading="THIAMINE REPLACEMENT")


def test_anchor_types():
    assert anchor_matches({"rec": "A-13"}, A13)
    assert not anchor_matches({"rec": "A-13"}, OTHER)
    assert anchor_matches({"text": "every 8 hours during alcohol"}, APPX)
    assert anchor_matches({"section": "THIAMINE REPLACEMENT", "doc": "BOP-2020"}, BOP)
    assert not anchor_matches({"section": "THIAMINE REPLACEMENT", "doc": "BJA-2023"}, BOP)
    assert anchor_matches({"section_chapter": "Appendix E: Monitoring Patients During Withdrawal Management"}, APPX)
    with pytest.raises(ValueError):
        anchor_matches({"bogus": 1}, A13)


def test_retrieval_metrics():
    anchors = [{"rec": "A-13"}, {"text": "during alcohol withdrawal management"}]
    m = retrieval_metrics([OTHER, A13, APPX, OTHER, OTHER], anchors, k=5)
    assert m["recall_at_k"] == 1.0 and m["precision_at_k"] == 0.4
    assert m["rr"] == 0.5 and m["hit_at_1"] == 0.0 and m["first_relevant_rank"] == 2
    none = retrieval_metrics([OTHER], anchors, k=5)
    assert none["recall_at_k"] == 0.0 and none["rr"] == 0.0 and none["first_relevant_rank"] is None
    half = retrieval_metrics([A13], anchors, k=1)
    assert half["recall_at_k"] == 0.5 and half["hit_at_1"] == 1.0


def test_stated_score_facts_extraction():
    text = ("The recorded score changed from 6 to 17 (+11 points) over 12 h [CALC-SCORE]. "
            "Pulse changed from 80 to 96 beats/min [CALC-VITALS].")
    assert stated_score_facts(text) == {"earliest": 6, "latest": 17, "change": 11}
    assert stated_score_facts("The latest CIWA-Ar score was 27 [ASM-00159].") == {"latest": 27}
    assert stated_score_facts("COWS scores showed a decrease of 9 points.") == {"change": -9}
    assert stated_score_facts("Vital signs were stable.") == {}


def test_score_fact_accuracy():
    gold = {"n": 4, "earliest": 6, "latest": 17, "change": 11}
    assert score_fact_accuracy({"earliest": 6, "latest": 17, "change": 11}, gold) == (3, 3, 3)
    assert score_fact_accuracy({"earliest": 6, "latest": 15}, gold) == (1, 2, 3)
    assert score_fact_accuracy({}, {"n": 0, "earliest": None, "latest": None, "change": None}) == (0, 0, 0)
    assert score_fact_accuracy({"latest": 5}, {"n": 1, "earliest": 5, "latest": 5, "change": None}) == (1, 1, 2)


def test_acknowledges_gaps():
    assert acknowledges_gaps({"a": "Temperature was not recorded in one set [GAP-01]."})
    assert acknowledges_gaps({"a": "See [CONFLICT-01]."})
    assert not acknowledges_gaps({"a": "Scores increased [CALC-SCORE]."})


def test_question_files_are_well_formed():
    gq = json.loads((PROJECT_ROOT / "evaluation/questions/guideline_questions.json").read_text())["questions"]
    assert len(gq) >= 20 and len({q["id"] for q in gq}) == len(gq)
    assert all(q["anchors"] for q in gq)
    cases = json.loads((PROJECT_ROOT / "evaluation/questions/patient_cases.json").read_text())
    assert len(cases["cases"]) >= 10
    for c in cases["cases"]:
        assert c["guideline_gold"] is None or c["guideline_gold"] in cases["guideline_gold_sets"]


@requires_index
def test_every_gold_anchor_matches_at_least_one_chunk():
    """Guards against gold labels that silently point at nothing (e.g. after re-chunking)."""
    from backend.rag.indexing import load_chunks

    chunks = load_chunks(get_settings().chunks_path)
    gq = json.loads((PROJECT_ROOT / "evaluation/questions/guideline_questions.json").read_text())["questions"]
    sets = json.loads((PROJECT_ROOT / "evaluation/questions/patient_cases.json").read_text())["guideline_gold_sets"]
    anchors = [a for q in gq for a in q["anchors"]] + [a for s in sets.values() for a in s]
    dead = [a for a in anchors if not any(anchor_matches(a, c) for c in chunks)]
    assert dead == []


def test_gold_score_facts_independent_window(tmp_path):
    from backend.database.db import connect_readonly
    from backend.database.synthetic_generator import GeneratorConfig, generate_dataset, write_sqlite

    p = tmp_path / "g.db"
    write_sqlite(generate_dataset(GeneratorConfig(patients_per_scenario_per_type=1)), p)
    conn = connect_readonly(p)
    bk = conn.execute("SELECT BookingNumber FROM patient_bookings LIMIT 1").fetchone()[0]
    full = gold_score_facts(conn, bk, None)
    last12 = gold_score_facts(conn, bk, 12)
    assert full["n"] >= last12["n"] and full["window"][1] == last12["window"][1]
    conn.close()
