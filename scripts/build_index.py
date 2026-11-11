"""Ingest guideline PDFs, chunk them, and build the dense + BM25 indexes.

Usage:
    python scripts/build_index.py              # full build
    python scripts/build_index.py --dry-run    # ingest + chunk only, print stats
    python scripts/build_index.py --show 5     # also print 5 sample chunks
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import get_settings  # noqa: E402
from backend.rag.chunking import chunk_document  # noqa: E402
from backend.rag.indexing import build_all, save_chunks  # noqa: E402
from backend.rag.ingestion import ingest_document, load_manifest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Skip embedding/indexing")
    ap.add_argument("--show", type=int, default=0, help="Print N random sample chunks")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    s = get_settings()
    all_chunks = []
    report = {}
    for doc in load_manifest(s.sources_manifest):
        ing = ingest_document(doc, s.guidelines_dir)
        chunks = chunk_document(ing, s.chunk_max_words, s.chunk_overlap_words)
        wc = [c.word_count for c in chunks]
        report[doc.document_id] = {
            "pages": ing.n_pages,
            "toc_entries": len(ing.toc),
            "body_lines": len(ing.lines),
            "chunks": len(chunks),
            "words_min_median_max": [min(wc), statistics.median(wc), max(wc)],
            "chunks_with_recommendation_ids": sum(bool(c.recommendation_ids) for c in chunks),
            "kerning_repairs": ing.kerning_merges,
            "warnings": ing.warnings,
        }
        all_chunks.extend(chunks)

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.show:
        for c in random.Random(0).sample(all_chunks, min(args.show, len(all_chunks))):
            print(f"\n--- {c.chunk_id} | {c.section_path} | {c.page_display()} (PDF {c.pdf_page_start}) ---")
            print(c.text)

    if args.dry_run:
        save_chunks(all_chunks, s.chunks_path)
        print(f"\nDry run: wrote {len(all_chunks)} chunks to {s.chunks_path}; indexes not rebuilt.")
        return 0

    info = build_all(all_chunks, s)
    print("\nIndex built:")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
