"""Build and load the guideline indexes.

Three artifacts are written, all keyed by `chunk_id`:

  * `data/processed/chunks.jsonl` — the canonical chunk store (text + metadata).
    Both retrievers return chunk IDs; full metadata is always looked up here,
    so citations come from a single source of truth.
  * ChromaDB collection (`data/vector_store/chroma`) — dense vectors from
    Qwen3-Embedding, cosine space. Embeddings are computed by us; Chroma's
    default embedding function is never used.
  * BM25S index (`data/vector_store/bm25`) — lexical index over the same text.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import bm25s
import Stemmer

from backend.config import Settings, get_settings
from backend.rag.models import Chunk

log = logging.getLogger(__name__)

COLLECTION_NAME = "guidelines"
_STEMMER_LANG = "english"


# --------------------------------------------------------------------------- #
# Chunk store
# --------------------------------------------------------------------------- #


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    ids = [c.chunk_id for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/build_index.py`.")
    with path.open(encoding="utf-8") as f:
        return [Chunk.model_validate_json(line) for line in f if line.strip()]


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #


def get_stemmer():
    return Stemmer.Stemmer(_STEMMER_LANG)


def bm25_tokenize(texts: list[str]):
    return bm25s.tokenize(texts, stopwords="en", stemmer=get_stemmer(), show_progress=False)


def build_bm25(chunks: list[Chunk], out_dir: Path) -> None:
    corpus = [c.embedding_text() for c in chunks]
    retriever = bm25s.BM25()
    retriever.index(bm25_tokenize(corpus), show_progress=False)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    retriever.save(str(out_dir))
    # Row i of the BM25 index corresponds to chunk_ids[i].
    (out_dir / "chunk_ids.json").write_text(json.dumps([c.chunk_id for c in chunks]))


def load_bm25(out_dir: Path) -> tuple[bm25s.BM25, list[str]]:
    if not out_dir.exists():
        raise FileNotFoundError(f"BM25 index not found at {out_dir}. Run `python scripts/build_index.py`.")
    retriever = bm25s.BM25.load(str(out_dir))
    ids = json.loads((out_dir / "chunk_ids.json").read_text())
    return retriever, ids


# --------------------------------------------------------------------------- #
# Chroma
# --------------------------------------------------------------------------- #


def get_chroma_client(chroma_dir: Path):
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(chroma_dir), settings=ChromaSettings(anonymized_telemetry=False)
    )


def chunk_metadata(c: Chunk) -> dict:
    """Flat metadata for Chroma (scalars only; None is not allowed)."""
    return {
        "document_id": c.document_id,
        "chapter": c.chapter or "",
        "section_heading": c.section_heading or "",
        "page_start": c.page_start or "",
        "page_end": c.page_end or "",
        "pdf_page_start": c.pdf_page_start,
        "pdf_page_end": c.pdf_page_end,
        "recommendation_ids": ",".join(c.recommendation_ids),
    }


def build_chroma(chunks: list[Chunk], embeddings, chroma_dir: Path) -> None:
    client = get_chroma_client(chroma_dir)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    col = client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}, embedding_function=None
    )
    batch = 64
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        col.add(
            ids=[c.chunk_id for c in part],
            embeddings=[e.tolist() for e in embeddings[i : i + batch]],
            documents=[c.embedding_text() for c in part],
            metadatas=[chunk_metadata(c) for c in part],
        )


def load_chroma_collection(chroma_dir: Path):
    client = get_chroma_client(chroma_dir)
    try:
        return client.get_collection(COLLECTION_NAME, embedding_function=None)
    except Exception as e:  # chromadb raises NotFoundError / ValueError by version
        raise FileNotFoundError(
            f"Chroma collection '{COLLECTION_NAME}' not found in {chroma_dir}. "
            "Run `python scripts/build_index.py`."
        ) from e


# --------------------------------------------------------------------------- #
# Full build
# --------------------------------------------------------------------------- #


def build_all(chunks: list[Chunk], settings: Settings | None = None, embedder=None) -> dict:
    """Write chunk store, BM25 index and Chroma collection. Returns build info."""
    s = settings or get_settings()
    if embedder is None:
        from backend.rag.embeddings import get_embedder

        embedder = get_embedder()

    save_chunks(chunks, s.chunks_path)

    t0 = time.perf_counter()
    build_bm25(chunks, s.bm25_dir)
    bm25_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    vecs = embedder.embed_documents([c.embedding_text() for c in chunks], show_progress=True)
    embed_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    build_chroma(chunks, vecs, s.chroma_dir)
    chroma_s = time.perf_counter() - t0

    docs: dict[str, int] = {}
    for c in chunks:
        docs[c.document_id] = docs.get(c.document_id, 0) + 1
    info = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_chunks": len(chunks),
        "chunks_per_document": docs,
        "embedding_model": s.embedding_path.name,
        "embedding_dim": int(vecs.shape[1]),
        "embedding_device": getattr(embedder, "device", "unknown"),
        "chunk_max_words": s.chunk_max_words,
        "chunk_overlap_words": s.chunk_overlap_words,
        "timings_s": {
            "bm25": round(bm25_s, 2),
            "embedding": round(embed_s, 2),
            "chroma": round(chroma_s, 2),
        },
    }
    s.index_info_path.parent.mkdir(parents=True, exist_ok=True)
    s.index_info_path.write_text(json.dumps(info, indent=2))
    return info
