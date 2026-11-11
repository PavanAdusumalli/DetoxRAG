"""Command-line guideline RAG assistant.

Usage:
    python scripts/ask_guidelines.py "How often should CIWA-Ar be repeated?"
    python scripts/ask_guidelines.py --method bm25 --k 3 "COWS frequency for fentanyl"
    python scripts/ask_guidelines.py --retrieve-only "signs of dehydration"
    python scripts/ask_guidelines.py --json "..."      # machine-readable output
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import get_settings  # noqa: E402
from backend.rag.retrieval import get_retriever  # noqa: E402


def print_passages(passages) -> None:
    for p in passages:
        c = p.chunk
        diag = []
        if p.dense_rank:
            diag.append(f"dense#{p.dense_rank} sim={p.dense_score:.3f}")
        if p.bm25_rank:
            diag.append(f"bm25#{p.bm25_rank} score={p.bm25_score:.2f}")
        recs = f" | recs {', '.join(c.recommendation_ids)}" if c.recommendation_ids else ""
        print(f"\n[{p.rank}] {c.chunk_id}  ({'; '.join(diag)})")
        print(f"    {c.section_path}")
        print(f"    {c.page_display()} (PDF p. {c.pdf_page_start}){recs}")
        print(textwrap.indent(textwrap.shorten(c.text, 400, placeholder=" …"), "    "))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--method", choices=["dense", "bm25", "hybrid"], default="hybrid")
    ap.add_argument("--k", type=int, default=None, help="Passages to retrieve")
    ap.add_argument("--retrieve-only", action="store_true", help="Skip LLM generation")
    ap.add_argument("--json", action="store_true", help="Print the full answer object as JSON")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    s = get_settings()
    k = args.k or s.retrieval_top_k
    retriever = get_retriever(args.method)

    if args.retrieve_only:
        t0 = time.perf_counter()
        passages = retriever.search(args.question, k)
        print(f"{len(passages)} passages via {args.method} in {(time.perf_counter() - t0) * 1000:.0f} ms")
        print_passages(passages)
        return 0

    from backend.rag.generation import answer_question
    from backend.rag.llm import get_llm

    result = answer_question(args.question, retriever, get_llm(), k=k)
    if args.json:
        print(result.model_dump_json(indent=2))
        return 0

    print("=" * 78)
    print("ANSWER" + ("  (insufficient evidence)" if result.insufficient_evidence else ""))
    print("=" * 78)
    print(textwrap.fill(result.answer, 78))
    if result.citations:
        print("\nCITATIONS")
        for c in result.citations:
            recs = f", recs {', '.join(c.recommendation_ids)}" if c.recommendation_ids else ""
            print(f"  [{c.citation_id}] {c.publisher.split('(')[0].strip()} ({c.publication_date}). "
                  f"{c.section_path}. {c.pages}, PDF p. {c.pdf_page_start}{recs}.")
            print(f"      {c.source_url}")
    for w in result.warnings:
        print(f"\n⚠ {w}")
    print(f"\nRetrieved passages ({args.method}):")
    print_passages(result.passages)
    print(f"\nTimings (ms): {result.timings_ms}  tokens: {result.llm_tokens}")
    print(f"\n{result.disclaimer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
