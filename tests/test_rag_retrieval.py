"""Dense, BM25 and hybrid retrieval — unit tests with fakes, plus a small
gold-question check against the real index."""

from __future__ import annotations

import bm25s
import numpy as np
import pytest

from backend.rag.indexing import bm25_tokenize
from backend.rag.models import RetrievedPassage
from backend.rag.retrieval import (
    BM25Retriever,
    ChunkStore,
    DenseRetriever,
    HybridRetriever,
    reciprocal_rank_fusion,
)
from tests.conftest import make_chunk, requires_index

# --------------------------------------------------------------------------- #
# Reciprocal Rank Fusion
# --------------------------------------------------------------------------- #


def test_rrf_scores_match_formula():
    fused = dict(reciprocal_rank_fusion({"a": ["x", "y"], "b": ["y", "z"]}, k=60))
    assert fused["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["x"] == pytest.approx(1 / 61)
    assert fused["z"] == pytest.approx(1 / 62)


def test_rrf_rewards_agreement():
    ranked = [cid for cid, _ in reciprocal_rank_fusion({"a": ["x", "y"], "b": ["y", "z"]})]
    assert ranked[0] == "y"


def test_rrf_ignores_duplicates_within_a_list():
    fused = dict(reciprocal_rank_fusion({"a": ["x", "x", "y"]}, k=60))
    assert fused["x"] == pytest.approx(1 / 61)
    assert fused["y"] == pytest.approx(1 / 62)  # duplicate didn't consume a rank


def test_rrf_weights_and_deterministic_ties():
    fused = reciprocal_rank_fusion({"a": ["x"], "b": ["y"]}, k=60, weights={"a": 2.0})
    assert [c for c, _ in fused] == ["x", "y"]
    tie = reciprocal_rank_fusion({"a": ["p"], "b": ["q"]}, k=60)
    assert [c for c, _ in tie] == ["p", "q"]  # equal score + rank -> by ID


def test_rrf_empty():
    assert reciprocal_rank_fusion({"a": [], "b": []}) == []


# --------------------------------------------------------------------------- #
# BM25 over a tiny in-memory corpus
# --------------------------------------------------------------------------- #

CORPUS = {
    "TST-2024-0001": "Alcohol withdrawal severity should be monitored with the CIWA-Ar every 8 hours.",
    "TST-2024-0002": "Opioid withdrawal is monitored with the COWS at least every 4 hours for fentanyl.",
    "TST-2024-0003": "Patients should have access to fluids to prevent dehydration.",
}


@pytest.fixture()
def store():
    return ChunkStore({cid: make_chunk(cid, text) for cid, text in CORPUS.items()})


@pytest.fixture()
def bm25_retriever(store):
    ids = list(CORPUS)
    index = bm25s.BM25()
    index.index(bm25_tokenize([store.get(i).embedding_text() for i in ids]), show_progress=False)
    return BM25Retriever(store, index, ids)


def test_bm25_exact_terminology(bm25_retriever):
    assert bm25_retriever.search("CIWA-Ar", 3)[0].citation_id == "TST-2024-0001"
    assert bm25_retriever.search("COWS fentanyl", 3)[0].citation_id == "TST-2024-0002"


def test_bm25_only_returns_matching_documents(bm25_retriever):
    hits = bm25_retriever.search("dehydration", 3)
    assert [h.citation_id for h in hits] == ["TST-2024-0003"]
    assert hits[0].method == "bm25" and hits[0].bm25_rank == 1


def test_bm25_empty_and_stopword_queries(bm25_retriever):
    assert bm25_retriever.search("", 3) == []
    assert bm25_retriever.search("   ", 3) == []
    assert bm25_retriever.search("the and of", 3) == []


def test_bm25_unknown_terms(bm25_retriever):
    assert bm25_retriever.search("xylophone quasar", 3) == []


# --------------------------------------------------------------------------- #
# Dense retriever with a fake Chroma collection
# --------------------------------------------------------------------------- #


class FakeCollection:
    def __init__(self, ids, distances):
        self.ids, self.distances = ids, distances

    def count(self):
        return len(self.ids)

    def query(self, query_embeddings, n_results, include):
        return {"ids": [self.ids[:n_results]], "distances": [self.distances[:n_results]]}


class FakeEmbedder:
    def embed_query(self, q):
        return np.zeros(4, dtype=np.float32)


def test_dense_converts_distance_and_skips_unknown_ids(store):
    col = FakeCollection(["TST-2024-0002", "MISSING-0001-0001", "TST-2024-0001"], [0.2, 0.3, 0.4])
    hits = DenseRetriever(store, col, FakeEmbedder()).search("q", 3)
    assert [h.citation_id for h in hits] == ["TST-2024-0002", "TST-2024-0001"]
    assert hits[0].dense_score == pytest.approx(0.8)


def test_dense_empty_query_and_empty_collection(store):
    assert DenseRetriever(store, FakeCollection([], []), FakeEmbedder()).search("q", 3) == []
    assert DenseRetriever(store, FakeCollection(["TST-2024-0001"], [0.1]), FakeEmbedder()).search("", 3) == []


# --------------------------------------------------------------------------- #
# Hybrid with fake component retrievers
# --------------------------------------------------------------------------- #


class StaticRetriever:
    def __init__(self, passages):
        self.passages = passages

    def search(self, query, k):
        return self.passages[:k]


def _p(chunk, rank, method, score):
    kw = {"dense_rank": rank, "dense_score": score} if method == "dense" else {"bm25_rank": rank, "bm25_score": score}
    return RetrievedPassage(chunk=chunk, score=score, rank=rank, method=method, **kw)


def test_hybrid_fuses_and_records_component_ranks(store):
    c1, c2, c3 = (store.get(i) for i in CORPUS)
    dense = StaticRetriever([_p(c2, 1, "dense", 0.9), _p(c1, 2, "dense", 0.8)])
    bm25 = StaticRetriever([_p(c1, 1, "bm25", 5.0), _p(c3, 2, "bm25", 2.0)])
    hits = HybridRetriever(dense, bm25, candidates=10).search("q", 5)
    assert hits[0].citation_id == "TST-2024-0001"  # in both lists
    assert hits[0].dense_rank == 2 and hits[0].bm25_rank == 1
    assert [h.rank for h in hits] == [1, 2, 3]
    assert all(h.method == "hybrid" for h in hits)


def test_hybrid_drops_duplicate_text(store):
    a = make_chunk("TST-2024-0001", "Identical passage text.")
    b = make_chunk("TST-2024-0009", "Identical passage text.")
    hits = HybridRetriever(StaticRetriever([_p(a, 1, "dense", 0.9)]), StaticRetriever([_p(b, 1, "bm25", 3.0)])).search("q", 5)
    assert len(hits) == 1


def test_hybrid_relevance_gate(store):
    c1 = store.get("TST-2024-0001")
    dense = StaticRetriever([_p(c1, 1, "dense", 0.40)])
    bm25 = StaticRetriever([_p(c1, 1, "bm25", 3.0)])
    assert HybridRetriever(dense, bm25, min_dense_similarity=0.5).search("q", 5) == []
    assert len(HybridRetriever(dense, bm25, min_dense_similarity=0.3).search("q", 5)) == 1
    assert len(HybridRetriever(dense, bm25, min_dense_similarity=None).search("q", 5)) == 1


def test_hybrid_respects_k(store):
    chunks = [store.get(i) for i in CORPUS]
    dense = StaticRetriever([_p(c, r, "dense", 0.9) for r, c in enumerate(chunks, 1)])
    bm25 = StaticRetriever([])
    assert len(HybridRetriever(dense, bm25).search("q", 2)) == 2


# --------------------------------------------------------------------------- #
# Real index: small gold set (expanded in Phase 8 evaluation)
# --------------------------------------------------------------------------- #

GOLD = [
    ("How often should the CIWA-Ar be reassessed during alcohol withdrawal treatment?", "A-13"),
    ("How often should patients using fentanyl be monitored with the COWS?", "O-42"),
    ("Should custody staff be trained to recognize oversedation?", "A-15"),
    ("Can alcohol withdrawal scales be used as stand-alone diagnostic tools?", "A-7"),
    ("Should oral or IV alcohol be used to treat alcohol withdrawal?", "A-10"),
]


@pytest.fixture(scope="module")
def real_retrievers():
    from backend.rag.retrieval import get_retriever

    return {m: get_retriever(m) for m in ("dense", "bm25", "hybrid")}


@requires_index
@pytest.mark.integration
@pytest.mark.parametrize("question,rec_id", GOLD)
def test_hybrid_gold_recommendation_in_top3(real_retrievers, question, rec_id):
    hits = real_retrievers["hybrid"].search(question, 3)
    found = [rid for h in hits for rid in h.chunk.recommendation_ids]
    assert rec_id in found, f"{rec_id} not in top-3: {[h.citation_id for h in hits]}"


@requires_index
@pytest.mark.integration
@pytest.mark.parametrize("method", ["dense", "bm25", "hybrid"])
def test_each_method_returns_real_chunks(real_retrievers, method):
    from backend.rag.retrieval import get_chunk_store

    hits = real_retrievers[method].search("CIWA-Ar monitoring frequency", 5)
    assert 0 < len(hits) <= 5
    store = get_chunk_store()
    assert all(h.citation_id in store for h in hits)
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


@requires_index
@pytest.mark.integration
def test_off_topic_query_gated(real_retrievers):
    assert real_retrievers["hybrid"].search("What's the weather in Paris?", 5) == []
