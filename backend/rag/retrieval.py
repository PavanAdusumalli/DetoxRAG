"""Dense, BM25, and hybrid (Reciprocal Rank Fusion) guideline retrieval.

Each retriever is independently usable and testable. All of them return
`RetrievedPassage` objects whose metadata comes from the canonical chunk store,
so a citation can only ever refer to a chunk that actually exists.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import numpy as np

from backend.config import get_settings
from backend.rag.indexing import bm25_tokenize, load_bm25, load_chroma_collection, load_chunks
from backend.rag.models import Chunk, RetrievedPassage

log = logging.getLogger(__name__)


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[RetrievedPassage]: ...


@dataclass
class ChunkStore:
    chunks: dict[str, Chunk]

    @classmethod
    def load(cls) -> "ChunkStore":
        return cls({c.chunk_id: c for c in load_chunks(get_settings().chunks_path)})

    def get(self, chunk_id: str) -> Chunk:
        return self.chunks[chunk_id]

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self.chunks

    def __len__(self) -> int:
        return len(self.chunks)


# --------------------------------------------------------------------------- #
# Dense
# --------------------------------------------------------------------------- #


class DenseRetriever:
    def __init__(self, store: ChunkStore, collection, embedder):
        self.store = store
        self.collection = collection
        self.embedder = embedder

    def search_vector(self, qvec: np.ndarray, k: int) -> list[RetrievedPassage]:
        n = min(k, self.collection.count())
        if n == 0:
            return []
        res = self.collection.query(
            query_embeddings=[qvec.tolist()], n_results=n, include=["distances"]
        )
        out: list[RetrievedPassage] = []
        for rank, (cid, dist) in enumerate(zip(res["ids"][0], res["distances"][0]), start=1):
            if cid not in self.store:
                log.warning("Dense hit %s missing from chunk store; skipping", cid)
                continue
            sim = 1.0 - float(dist)  # cosine distance -> cosine similarity
            out.append(
                RetrievedPassage(
                    chunk=self.store.get(cid), score=sim, rank=rank, method="dense",
                    dense_rank=rank, dense_score=sim,
                )
            )
        return out

    def search(self, query: str, k: int) -> list[RetrievedPassage]:
        if not query.strip():
            return []
        return self.search_vector(self.embedder.embed_query(query), k)


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #


class BM25Retriever:
    def __init__(self, store: ChunkStore, bm25, chunk_ids: list[str]):
        self.store = store
        self.bm25 = bm25
        self.chunk_ids = chunk_ids

    def search(self, query: str, k: int) -> list[RetrievedPassage]:
        if not query.strip():
            return []
        tokens = bm25_tokenize([query])
        if not tokens.vocab:  # query is only stopwords / punctuation
            return []
        n = min(k, len(self.chunk_ids))
        docs, scores = self.bm25.retrieve(tokens, k=n, show_progress=False)
        out: list[RetrievedPassage] = []
        rank = 0
        for idx, score in zip(docs[0], scores[0]):
            if score <= 0:  # no query term matched this document
                continue
            rank += 1
            cid = self.chunk_ids[int(idx)]
            out.append(
                RetrievedPassage(
                    chunk=self.store.get(cid), score=float(score), rank=rank, method="bm25",
                    bm25_rank=rank, bm25_score=float(score),
                )
            )
        return out


# --------------------------------------------------------------------------- #
# Hybrid (RRF)
# --------------------------------------------------------------------------- #


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[str]], k: int = 60, weights: dict[str, float] | None = None
) -> list[tuple[str, float]]:
    """Fuse ranked ID lists: score(d) = sum_i w_i / (k + rank_i(d)).

    Duplicate IDs within one list count only at their best rank. Ties are
    broken by the best rank achieved in any list, then by ID for determinism.
    """
    weights = weights or {}
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for name, ids in ranked_lists.items():
        w = weights.get(name, 1.0)
        seen: set[str] = set()
        rank = 0
        for cid in ids:
            if cid in seen:
                continue
            seen.add(cid)
            rank += 1
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
            best_rank[cid] = min(best_rank.get(cid, rank), rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], best_rank[kv[0]], kv[0]))


def _normalized_text(text: str) -> str:
    return re.sub(r"\W+", " ", text).strip().casefold()


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        candidates: int = 20,
        rrf_k: int = 60,
        weights: dict[str, float] | None = None,
        min_dense_similarity: float | None = None,
    ):
        self.dense = dense
        self.bm25 = bm25
        self.candidates = candidates
        self.rrf_k = rrf_k
        self.weights = weights or {"dense": 1.0, "bm25": 1.0}
        # Query-level relevance gate: if no dense candidate reaches this cosine
        # similarity, the query is treated as off-topic and nothing is returned.
        # BM25 alone can't make that call (it matches common words like "what").
        self.min_dense_similarity = min_dense_similarity

    def search(self, query: str, k: int) -> list[RetrievedPassage]:
        if not query.strip():
            return []
        dense_hits = self.dense.search(query, self.candidates)
        if self.min_dense_similarity is not None:
            best = max((p.dense_score or 0.0 for p in dense_hits), default=0.0)
            if best < self.min_dense_similarity:
                log.info("Relevance gate: best dense similarity %.3f < %.3f; no passages", best, self.min_dense_similarity)
                return []
        bm25_hits = self.bm25.search(query, self.candidates)
        by_id_dense = {p.citation_id: p for p in dense_hits}
        by_id_bm25 = {p.citation_id: p for p in bm25_hits}

        fused = reciprocal_rank_fusion(
            {"dense": [p.citation_id for p in dense_hits], "bm25": [p.citation_id for p in bm25_hits]},
            k=self.rrf_k,
            weights=self.weights,
        )

        out: list[RetrievedPassage] = []
        seen_text: set[str] = set()
        for cid, score in fused:
            d, b = by_id_dense.get(cid), by_id_bm25.get(cid)
            chunk = (d or b).chunk
            # Drop near-verbatim duplicate passages (e.g. repeated boilerplate).
            norm = _normalized_text(chunk.text)
            if norm in seen_text:
                continue
            seen_text.add(norm)
            out.append(
                RetrievedPassage(
                    chunk=chunk, score=score, rank=len(out) + 1, method="hybrid",
                    dense_rank=d.dense_rank if d else None, dense_score=d.dense_score if d else None,
                    bm25_rank=b.bm25_rank if b else None, bm25_score=b.bm25_score if b else None,
                )
            )
            if len(out) >= k:
                break
        return out


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def get_chunk_store() -> ChunkStore:
    return ChunkStore.load()


def get_retriever(method: str = "hybrid") -> Retriever:
    """Construct a retriever over the on-disk indexes ('dense' | 'bm25' | 'hybrid')."""
    s = get_settings()
    store = get_chunk_store()
    if method in ("dense", "hybrid"):
        from backend.rag.embeddings import get_embedder

        dense = DenseRetriever(store, load_chroma_collection(s.chroma_dir), get_embedder())
        if method == "dense":
            return dense
    bm25_index, ids = load_bm25(s.bm25_dir)
    bm25 = BM25Retriever(store, bm25_index, ids)
    if method == "bm25":
        return bm25
    if method == "hybrid":
        return HybridRetriever(
            dense, bm25, candidates=s.retrieval_candidates, rrf_k=s.rrf_k,
            min_dense_similarity=s.retrieval_min_dense_similarity,
        )
    raise ValueError(f"Unknown retrieval method: {method!r}")
