"""Verify both local models load and run offline; report latency and memory.

Usage:
    python scripts/check_models.py
"""

from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import get_settings  # noqa: E402


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS


def main() -> int:
    s = get_settings()
    ok = True

    print(f"Embedding model: {s.embedding_path}")
    try:
        from backend.rag.embeddings import get_embedder

        t0 = time.perf_counter()
        emb = get_embedder()
        load_s = time.perf_counter() - t0
        docs = [
            "The CIWA-Ar is used to assess alcohol withdrawal severity.",
            "The COWS rates common opioid withdrawal signs.",
            "Breakfast is served at 6 a.m.",
        ]
        t0 = time.perf_counter()
        d = emb.embed_documents(docs)
        q = emb.embed_query("How is opioid withdrawal severity measured?")
        enc_ms = (time.perf_counter() - t0) * 1000
        sims = d @ q
        best = int(sims.argmax())
        print(f"  device={emb.device} dim={emb.dim} load={load_s:.1f}s encode={enc_ms:.0f}ms")
        print(f"  similarities={[round(float(x), 3) for x in sims]} -> best: {docs[best]!r}")
        if best != 1:
            ok = False
            print("  [FAIL] expected the COWS sentence to be most similar")
        else:
            print("  [ ok ]")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  [FAIL] {e}")

    print(f"\nLLM: {s.llm_path}")
    try:
        from backend.rag.llm import get_llm

        t0 = time.perf_counter()
        llm = get_llm()
        load_s = time.perf_counter() - t0
        res = llm.generate(
            [{"role": "user", "content": "Reply with exactly: DetoxRAG local model OK"}], max_tokens=20
        )
        print(f"  load={load_s:.1f}s gen={res.seconds:.2f}s tokens={res.generation_tokens} "
              f"({res.generation_tps:.0f} tok/s) peak_mlx_mem={res.peak_memory_gb:.2f} GB")
        print(f"  output={res.text!r}")
        print("  [ ok ]" if "OK" in res.text else "  [WARN] unexpected output")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  [FAIL] {e}")

    print(f"\nProcess max RSS: {rss_gb():.2f} GB")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
