"""DetoxRAG benchmark runner.

    python -m evaluation.run_evaluation                 # full run (~15 min on M-series, 16 GB)
    python -m evaluation.run_evaluation --skip-generation --skip-qa   # retrieval + analytics only (fast)
    python -m evaluation.run_evaluation --cases C01,C10 # subset of patient cases
    python -m evaluation.run_evaluation --attach-review # merge evaluation/manual_review.json into latest.json

Experiments
  Retrieval     dense vs BM25 vs hybrid on manually labelled guideline questions
  A             patient question + SQL facts + timeline analytics, no guideline retrieval
  B             A + dense-retrieved guideline passages
  C             A + hybrid (dense + BM25, RRF) guideline passages
  Guideline QA  cited answers to the guideline questions (dense vs hybrid)
  Analytics     deterministic gap detection vs injected defects; temporal calculations
                vs an independent SQL implementation
  Isolation     every retrieved record belongs to the requested booking

All numbers in the output come from this run. Generation uses temperature 0.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

from backend.config import PROJECT_ROOT, get_settings
from backend.database import queries as q
from backend.database.db import connect_readonly
from backend.rag.generation import answer_question
from backend.services.clinical_summary import generate_clinical_summary
from backend.services.timeline import build_patient_timeline
from evaluation.ground_truth import evaluate_all, summarize
from evaluation.metrics import (
    acknowledges_gaps,
    gold_score_facts,
    is_relevant,
    mean,
    retrieval_metrics,
    score_fact_accuracy,
    stated_score_facts,
)

EVAL_DIR = PROJECT_ROOT / "evaluation"
RESULTS_DIR = EVAL_DIR / "results"
EXPERIMENTS = {
    "A_structured_only": ("A · Structured data only", "none"),
    "B_dense_rag": ("B · + Dense RAG", "dense"),
    "C_hybrid_rag": ("C · + Hybrid RAG", "hybrid"),
}
RECORD_ID_RE = re.compile(r"^(ASM|VIT|ORD|MAR|NOTE)-\d+$")
TABLE_FOR = {"ASM": ("withdrawal_assessments", "AssessmentID"), "VIT": ("vital_signs", "VitalID"),
             "ORD": ("medication_orders", "OrderID"), "MAR": ("medication_administrations", "AdministrationID"),
             "NOTE": ("nursing_notes", "NoteID")}


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #


def run_retrieval(retrievers: dict, questions: list[dict], k: int) -> dict:
    per_q = []
    agg: dict[str, dict[str, list]] = {m: {"recall_at_k": [], "precision_at_k": [], "rr": [], "hit_at_1": [], "ms": []} for m in retrievers}
    for qd in questions:
        row = {"id": qd["id"], "question": qd["question"]}
        for m, r in retrievers.items():
            t0 = time.perf_counter()
            hits = r.search(qd["question"], k)
            ms = (time.perf_counter() - t0) * 1000
            met = retrieval_metrics([h.chunk for h in hits], qd["anchors"], k)
            for key in ("recall_at_k", "precision_at_k", "rr", "hit_at_1"):
                agg[m][key].append(met[key])
            agg[m]["ms"].append(ms)
            row[m] = {**met, "retrieved": [h.citation_id for h in hits],
                      "relevant": [h.citation_id for h in hits if is_relevant(h.chunk, qd["anchors"])]}
        per_q.append(row)
    labels = {"dense": "Dense (Qwen3-Embedding + Chroma)", "bm25": "BM25 (BM25S)", "hybrid": "Hybrid (RRF)"}
    methods = {
        m: {"label": labels[m], "metrics": {
            "recall_at_k": mean(a["recall_at_k"]), "precision_at_k": mean(a["precision_at_k"]),
            "mrr": mean(a["rr"]), "hit_at_1": mean(a["hit_at_1"]),
            "mean_latency_ms": round(sum(a["ms"][1:]) / max(len(a["ms"]) - 1, 1), 1),  # first query is warm-up
        }}
        for m, a in agg.items()
    }
    return {"k": k, "n_questions": len(questions), "methods": methods, "per_question": per_q}


# --------------------------------------------------------------------------- #
# Analytics + isolation
# --------------------------------------------------------------------------- #


def run_analytics(conn) -> dict:
    gap = summarize(evaluate_all(conn))
    checks = mismatches = 0
    examples = []
    for p in q.list_patients(conn):
        for hours in (None, 24, 12):
            gold = gold_score_facts(conn, p.BookingNumber, hours)
            t = build_patient_timeline(conn, p.PatientID, p.BookingNumber, hours=hours)
            s = t.score_summary
            got = {
                "window": (t.window.start, t.window.end), "n": s.n_assessments,
                "earliest": s.earliest.score if s.earliest else None, "latest": s.latest.score if s.latest else None,
                "change": s.change, "min": s.minimum.score if s.minimum else None,
                "max": s.maximum.score if s.maximum else None, "max_interval_h": t.intervals.max_hours,
            }
            for k, v in gold.items():
                checks += 1
                if got[k] != v:
                    mismatches += 1
                    if len(examples) < 10:
                        examples.append({"patient": p.PatientID, "hours": hours, "field": k, "gold": v, "got": got[k]})
    return {
        "gap_detection": gap,
        "temporal_accuracy": {"n_checks": checks, "mismatches": mismatches,
                              "accuracy": round((checks - mismatches) / checks, 4), "examples": examples},
    }


def run_isolation(conn) -> dict:
    checks = violations = 0
    for p in q.list_patients(conn):
        t = build_patient_timeline(conn, p.PatientID, p.BookingNumber)
        ids = ([("ASM", a.AssessmentID) for a in t.assessments] + [("VIT", v.VitalID) for v in t.vitals]
               + [("ORD", o.OrderID) for o in t.orders] + [("MAR", m.AdministrationID) for m in t.administrations]
               + [("NOTE", n.NoteID) for n in t.notes])
        for prefix, rid in ids:
            table, col = TABLE_FOR[prefix]
            owner = conn.execute(f"SELECT BookingNumber FROM {table} WHERE {col} = ?", (rid,)).fetchone()[0]
            checks += 1
            violations += owner != p.BookingNumber
        # And nothing from this booking is missing (no silent truncation)
        for table, col in TABLE_FOR.values():
            n_db = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE BookingNumber = ?", (p.BookingNumber,)).fetchone()[0]
            n_tl = sum(1 for pre, _ in ids if TABLE_FOR[pre][0] == table)
            checks += 1
            violations += n_db != n_tl
    return {"n_checks": checks, "violations": violations, "generation_checks": 0, "generation_violations": 0}


# --------------------------------------------------------------------------- #
# Generation experiments
# --------------------------------------------------------------------------- #


def resolve_case_patient(conn, case: dict) -> tuple[str, str]:
    row = conn.execute(
        "SELECT g.PatientID, g.BookingNumber FROM synthetic_ground_truth g JOIN patient_bookings b USING (BookingNumber) "
        "WHERE g.Scenario = ? AND b.WithdrawalType = ? "
        + ("AND json_extract(g.Details, '$.released_early') = 0 " if case["scenario"] == "incomplete_monitoring" else "")
        + "ORDER BY g.PatientID LIMIT 1",
        (case["scenario"], case["withdrawal_type"]),
    ).fetchone()
    return row[0], row[1]


def run_generation(conn, llm, retrievers, cases_doc: dict, only: set[str] | None, isolation: dict) -> dict:
    gold_sets = cases_doc["guideline_gold_sets"]
    cases = [c for c in cases_doc["cases"] if not only or c["id"] in only]
    per_case = []
    for case in cases:
        pid, bk = resolve_case_patient(conn, case)
        gold = gold_score_facts(conn, bk, case["hours"])
        anchors = gold_sets.get(case["guideline_gold"]) if case["guideline_gold"] else None
        for exp, (label, mode) in EXPERIMENTS.items():
            log(f"  {case['id']} {exp} {pid}")
            res = generate_clinical_summary(conn, pid, case["question"], llm,
                                            retriever=retrievers.get(mode), booking_number=bk,
                                            hours=case["hours"], retrieval_mode=mode)
            raw = res.raw_model_output or {}
            raw_patient_text = " ".join(raw.get(f, "") for f in
                                        ("patient_summary", "assessment_trend", "vital_sign_observations", "medication_documentation"))
            stated = stated_score_facts(raw_patient_text)
            correct, n_stated, total = score_fact_accuracy(stated, gold)
            v = res.validation
            valid = len(res.patient_evidence) + len(res.guideline_citations)
            # Isolation: every cited record ID belongs to this booking; no other patient IDs in output.
            for e in res.patient_evidence:
                if RECORD_ID_RE.match(e.evidence_id):
                    table, col = TABLE_FOR[e.evidence_id.split("-")[0]]
                    owner = conn.execute(f"SELECT BookingNumber FROM {table} WHERE {col} = ?", (e.evidence_id,)).fetchone()
                    isolation["generation_checks"] += 1
                    isolation["generation_violations"] += (owner is None or owner[0] != bk)
            out_text = json.dumps(res.model_dump(include={"patient_summary", "assessment_trend", "vital_sign_observations",
                                                          "medication_documentation", "guideline_context"}))
            isolation["generation_checks"] += 1
            isolation["generation_violations"] += bool(set(re.findall(r"\bP\d{3}\b", out_text)) - {pid})
            has_gaps = bool(res.documentation_gaps or res.documentation_conflicts)
            retrieved_rel = [p.citation_id for p in res.retrieved_passages if anchors and is_relevant(p.chunk, anchors)]
            cited_rel = [c.citation_id for c in res.guideline_citations
                         if anchors and is_relevant(next(p.chunk for p in res.retrieved_passages if p.citation_id == c.citation_id), anchors)]
            per_case.append({
                "case_id": case["id"], "experiment": exp, "patient_id": pid, "booking_number": bk,
                "scenario": case["scenario"], "question": case["question"], "hours": case["hours"],
                "gold_facts": {k: gold[k] for k in ("n", "earliest", "latest", "change")}, "stated_facts": stated,
                "facts_correct": correct, "facts_stated": n_stated, "facts_total": total,
                "valid_citations": valid, "invalid_citations": len(v.invalid_citations),
                "fact_check_failures": len(v.fact_check_failures),
                "fact_check_detail": [f.model_dump() for f in v.fact_check_failures],
                "sentences": v.n_sentences, "uncited_sentences": v.n_uncited_sentences,
                "removed_sentences": len(v.removed_sentences), "json_ok": v.json_valid_first_try,
                "generated": res.generated, "has_gaps": has_gaps,
                "gap_acknowledged": acknowledges_gaps(raw) if has_gaps else None,
                "guideline_gold": case["guideline_gold"] if mode != "none" else None,
                "n_retrieved": len(res.retrieved_passages), "gold_retrieved": bool(retrieved_rel) if anchors and mode != "none" else None,
                "guideline_cited": len(res.guideline_citations), "guideline_cited_relevant": len(cited_rel),
                "retrieval_ms": res.timings_ms.get("evidence_and_retrieval"), "generation_ms": res.timings_ms.get("generation"),
                "total_ms": res.timings_ms.get("total"), "prompt_tokens": res.llm_tokens.get("prompt"),
                "raw_output": raw,
                "final_output": res.model_dump(include={"patient_summary", "assessment_trend", "vital_sign_observations",
                                                        "medication_documentation", "guideline_context"}),
                "warnings": res.warnings,
            })

    experiments = {}
    for exp, (label, mode) in EXPERIMENTS.items():
        rows = [r for r in per_case if r["experiment"] == exp]
        if not rows:
            continue
        sent = sum(r["sentences"] + r["removed_sentences"] for r in rows)
        unsupported = sum(r["uncited_sentences"] + r["fact_check_failures"] + r["removed_sentences"] for r in rows)
        stated = sum(r["facts_stated"] for r in rows)
        total = sum(r["facts_total"] for r in rows)
        cites = sum(r["valid_citations"] + r["invalid_citations"] for r in rows)
        gap_rows = [r for r in rows if r["has_gaps"]]
        gold_rows = [r for r in rows if r["gold_retrieved"] is not None]
        cited_g = sum(r["guideline_cited"] for r in rows if r["guideline_gold"])
        experiments[exp] = {"label": label, "retrieval_mode": mode, "metrics": {
            "citation_validity": round(sum(r["valid_citations"] for r in rows) / cites, 4) if cites else None,
            "patient_fact_accuracy": round(sum(r["facts_correct"] for r in rows) / stated, 4) if stated else None,
            "patient_fact_coverage": round(stated / total, 4) if total else None,
            "contradiction_free_rate": round(sum(r["fact_check_failures"] == 0 for r in rows) / len(rows), 4),
            "unsupported_sentence_rate": round(unsupported / sent, 4) if sent else None,
            "gap_acknowledgement_rate": round(sum(bool(r["gap_acknowledged"]) for r in gap_rows) / len(gap_rows), 4) if gap_rows else None,
            "guideline_evidence_success": round(sum(r["gold_retrieved"] for r in gold_rows) / len(gold_rows), 4) if gold_rows else None,
            "guideline_citation_relevance": round(sum(r["guideline_cited_relevant"] for r in rows if r["guideline_gold"]) / cited_g, 4) if cited_g else None,
            "safety_removal_rate": round(sum(r["removed_sentences"] > 0 for r in rows) / len(rows), 4),
            "json_valid_rate": round(sum(r["json_ok"] for r in rows) / len(rows), 4),
            "mean_retrieval_ms": mean([r["retrieval_ms"] for r in rows]),
            "mean_generation_ms": mean([r["generation_ms"] for r in rows]),
            "mean_total_ms": mean([r["total_ms"] for r in rows]),
            "mean_prompt_tokens": mean([r["prompt_tokens"] for r in rows]),
        }}
    return {"n_cases": len(cases), "experiments": experiments, "per_case": per_case}


# --------------------------------------------------------------------------- #
# Guideline QA
# --------------------------------------------------------------------------- #


def run_guideline_qa(llm, retrievers, questions: list[dict]) -> dict:
    per_q = []
    for qd in questions:
        for m in ("dense", "hybrid"):
            log(f"  QA {qd['id']} {m}")
            t0 = time.perf_counter()
            a = answer_question(qd["question"], retrievers[m], llm, k=5)
            ms = (time.perf_counter() - t0) * 1000
            by_id = {p.citation_id: p.chunk for p in a.passages}
            per_q.append({
                "id": qd["id"], "method": m, "answered": not a.insufficient_evidence,
                "valid_citations": len(a.citations), "invalid_citations": len(a.invalid_citations),
                "cited_gold": any(is_relevant(by_id[c.citation_id], qd["anchors"]) for c in a.citations),
                "gold_retrieved": any(is_relevant(p.chunk, qd["anchors"]) for p in a.passages),
                "total_ms": round(ms, 1), "answer": a.answer, "citations": [c.citation_id for c in a.citations],
            })
    exps = {}
    for m, label in (("dense", "Dense RAG"), ("hybrid", "Hybrid RAG")):
        rows = [r for r in per_q if r["method"] == m]
        cites = sum(r["valid_citations"] + r["invalid_citations"] for r in rows)
        exps[m] = {"label": label, "metrics": {
            "answered_rate": round(sum(r["answered"] for r in rows) / len(rows), 4),
            "citation_validity": round(sum(r["valid_citations"] for r in rows) / cites, 4) if cites else None,
            "cited_gold_rate": round(sum(r["cited_gold"] for r in rows) / len(rows), 4),
            "gold_retrieved_rate": round(sum(r["gold_retrieved"] for r in rows) / len(rows), 4),
            "mean_total_ms": mean([r["total_ms"] for r in rows]),
        }}
    return {"n_questions": len(questions), "experiments": exps, "per_question": per_q}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def attach_review() -> int:
    latest = RESULTS_DIR / "latest.json"
    review = EVAL_DIR / "manual_review.json"
    data = json.loads(latest.read_text())
    rv = json.loads(review.read_text())
    if rv.get("run_id") != data["run_id"]:
        print(f"manual_review.json is for run {rv.get('run_id')}, latest is {data['run_id']}; not attaching.")
        return 1
    data["manual_review"] = rv["notes"]
    latest.write_text(json.dumps(data, indent=2))
    (RESULTS_DIR / f"run_{data['run_id']}.json").write_text(json.dumps(data, indent=2))
    print(f"Attached {len(rv['notes'])} review notes to run {data['run_id']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-generation", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--cases", help="Comma-separated case IDs (default: all)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--attach-review", action="store_true")
    args = ap.parse_args()
    if args.attach_review:
        return attach_review()

    t_start = time.perf_counter()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    s = get_settings()
    conn = connect_readonly()
    gq = json.loads((EVAL_DIR / "questions" / "guideline_questions.json").read_text())["questions"]
    cases_doc = json.loads((EVAL_DIR / "questions" / "patient_cases.json").read_text())

    from backend.rag.retrieval import get_retriever

    t0 = time.perf_counter()
    retrievers = {m: get_retriever(m) for m in ("dense", "bm25", "hybrid")}
    retrievers["dense"].search("warm-up", 1)
    load_embed_s = time.perf_counter() - t0
    log(f"Retrievers ready in {load_embed_s:.1f}s")

    results: dict = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "machine": f"{platform.machine()} · {platform.mac_ver()[0] and 'macOS ' + platform.mac_ver()[0]}",
            "python": platform.python_version(),
            "llm": s.llm_path.name, "embedding": s.embedding_path.name,
            "temperature": s.llm_temperature, "chunk_max_words": s.chunk_max_words,
            "retrieval_candidates": s.retrieval_candidates, "rrf_k": s.rrf_k,
            "relevance_gate": s.retrieval_min_dense_similarity,
        },
    }

    log("Retrieval benchmark")
    results["retrieval"] = run_retrieval(retrievers, gq, args.k)
    log("Analytics benchmark")
    results["analytics"] = run_analytics(conn)
    log("Isolation checks")
    isolation = run_isolation(conn)

    llm = None
    load_llm_s = None
    if not (args.skip_generation and args.skip_qa):
        from backend.rag.llm import get_llm

        t0 = time.perf_counter()
        llm = get_llm()
        load_llm_s = time.perf_counter() - t0
        log(f"LLM loaded in {load_llm_s:.1f}s")
    if not args.skip_generation:
        log("Generation experiments A/B/C")
        only = set(args.cases.split(",")) if args.cases else None
        results["generation"] = run_generation(conn, llm, {"none": None, **retrievers}, cases_doc, only, isolation)
    if not args.skip_qa:
        log("Guideline QA")
        results["guideline_qa"] = run_guideline_qa(llm, retrievers, gq)
    results["isolation"] = {**isolation, "violations": isolation["violations"] + isolation["generation_violations"],
                            "n_checks": isolation["n_checks"] + isolation["generation_checks"]}

    sysinfo = {
        "embedding_and_index_load_s": round(load_embed_s, 2),
        "peak_process_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),
    }
    if load_llm_s is not None:
        import mlx.core as mx

        sysinfo["llm_load_s"] = round(load_llm_s, 2)
        sysinfo["peak_mlx_memory_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    for m, v in results["retrieval"]["methods"].items():
        sysinfo[f"retrieval_{m}_mean_ms"] = v["metrics"]["mean_latency_ms"]
    if "generation" in results:
        for exp, v in results["generation"]["experiments"].items():
            sysinfo[f"{exp}_mean_total_ms"] = v["metrics"]["mean_total_ms"]
    results["system"] = sysinfo
    results["not_run"] = [{
        "experiment": "D · Hybrid RAG + reranking",
        "reason": "Optional in the project plan. Not run: it needs an additional local reranker model (e.g. "
                  "Qwen3-Reranker-0.6B, ~1.2 GB) and adds a third model to a 16 GB machine. See the retrieval "
                  "results for the remaining headroom a reranker could address.",
    }]
    results["caveats"] = [
        "Small benchmark: 21 guideline questions and 12 patient cases × 3 configurations, single run at temperature 0. Differences of a few points are within noise.",
        "All patient data is synthetic and the gap-detection rules were written knowing how the generator injects defects; 100% detection shows the code works as designed, not real-world sensitivity.",
        "Gold guideline labels were assigned by the developer by reading the extracted chunks (no independent clinical review).",
        "Patient fact accuracy uses regex extraction of score claims from the raw model output; claims phrased differently are not counted as stated.",
        "Nothing here demonstrates clinical effectiveness or patient safety.",
    ]
    results["duration_s"] = round(time.perf_counter() - t_start, 1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    review = EVAL_DIR / "manual_review.json"
    if review.exists():
        rv = json.loads(review.read_text())
        if rv.get("run_id") == run_id:
            results["manual_review"] = rv["notes"]
    out = json.dumps(results, indent=2, default=str)
    (RESULTS_DIR / f"run_{run_id}.json").write_text(out)
    (RESULTS_DIR / "latest.json").write_text(out)
    log(f"Done in {results['duration_s']:.0f}s -> evaluation/results/run_{run_id}.json")
    print(json.dumps({k: results[k] for k in ("retrieval",) if k in results}["retrieval"]["methods"], indent=1))
    if "generation" in results:
        print(json.dumps({k: v["metrics"] for k, v in results["generation"]["experiments"].items()}, indent=1))
    if "guideline_qa" in results:
        print(json.dumps({k: v["metrics"] for k, v in results["guideline_qa"]["experiments"].items()}, indent=1))
    print(json.dumps({"analytics": {"gap_recall": results["analytics"]["gap_detection"]["recall"],
                                    "temporal": results["analytics"]["temporal_accuracy"]},
                      "isolation": results["isolation"], "system": sysinfo}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
