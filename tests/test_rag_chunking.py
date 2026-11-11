"""Chunk integrity: boundaries, recommendation grouping, sizes, metadata."""

from __future__ import annotations

import re

import pytest

from backend.config import get_settings
from backend.rag.chunking import build_units, chunk_document, join_lines, split_sentences
from backend.rag.ingestion import IngestedDocument
from backend.rag.models import SourceDocument
from tests.conftest import make_line, requires_index

REC_RE = re.compile(r"^((?:G|A|SH|O|S)-\d+)\.\s")


def _doc(lines, rec_regex=r"^((?:G|A|SH|O|S)-\d+)\.\s") -> IngestedDocument:
    src = SourceDocument(
        document_id="TST-2024", file="x.pdf", title="Test", short_title="Test", publisher="P",
        publication_date="2024", source_url="https://example.org", parsing={"recommendation_id_regex": rec_regex},
    )
    return IngestedDocument(source=src, lines=lines, toc=[], n_pages=1, warnings=[], kerning_merges=[])


def test_join_lines_keeps_items_and_joins_wraps():
    text = join_lines(["A-1. Monitor the patient", "every 8 hours.", "A. If severe, repeat", "more often.", "• bullet"])
    assert text.split("\n") == [
        "A-1. Monitor the patient every 8 hours.",
        "A. If severe, repeat more often.",
        "• bullet",
    ]


def test_recommendation_keeps_sub_items():
    lines = [
        make_line("Intro narrative text."),
        make_line("A-13. Assess at least every 8 hours"),
        make_line("A. If the score is high, repeat every 6 hours."),
        make_line("B. More frequent monitoring if indicated."),
        make_line("A-14. Staff should be trained."),
    ]
    units = build_units(lines, REC_RE)
    assert [u.kind for u in units] == ["narrative", "recommendation", "recommendation"]
    assert units[1].recommendation_id == "A-13"
    assert "B. More frequent" in units[1].text


def test_headings_excluded_from_unit_text():
    lines = [make_line("Monitoring", heading=True), make_line("Body text here.")]
    units = build_units(lines, REC_RE)
    assert len(units) == 1 and "Monitoring" not in units[0].text


def test_chunks_never_cross_sections():
    lines = [make_line(f"Sentence {i} in section one.", section="One") for i in range(5)] + [
        make_line(f"Sentence {i} in section two.", section="Two") for i in range(5)
    ]
    chunks = chunk_document(_doc(lines), max_words=1000, overlap_words=0)
    assert len(chunks) == 2
    assert {c.section_heading for c in chunks} == {"One", "Two"}
    assert all("section two" not in c.text for c in chunks if c.section_heading == "One")


def test_long_section_split_with_size_limit_and_overlap():
    sentences = [f"This is narrative sentence number {i} about monitoring." for i in range(60)]
    lines = [make_line(s, page=1 + i // 20) for i, s in enumerate(sentences)]
    chunks = chunk_document(_doc(lines), max_words=80, overlap_words=20)
    assert len(chunks) > 3
    for c in chunks:
        assert c.word_count <= 80 + 20  # body limit plus carried overlap
    # Overlap: the next chunk begins with the previous chunk's final sentences
    # (as many whole sentences as fit within overlap_words).
    carried = chunks[1].text.split("\n", 1)[0]
    assert 0 < len(carried.split()) <= 20
    assert chunks[0].text.endswith(carried)
    assert split_sentences(carried)[-1] == split_sentences(chunks[0].text)[-1]
    # Page ranges follow the source lines.
    assert chunks[0].page_start == "1" and chunks[-1].page_end == "3"


def test_recommendations_are_not_overlapped():
    lines = [make_line(f"A-{i}. " + "word " * 50) for i in range(1, 5)]
    chunks = chunk_document(_doc(lines), max_words=120, overlap_words=40)
    all_ids = [rid for c in chunks for rid in c.recommendation_ids]
    assert all_ids == ["A-1", "A-2", "A-3", "A-4"]  # each exactly once
    for c in chunks:
        assert c.text.count("A-") == len(c.recommendation_ids)


def test_chunk_ids_unique_and_sequential():
    lines = [make_line(f"Sentence {i}.", section=f"S{i}") for i in range(4)]
    ids = [c.chunk_id for c in chunk_document(_doc(lines))]
    assert ids == ["TST-2024-0001", "TST-2024-0002", "TST-2024-0003", "TST-2024-0004"]


# --------------------------------------------------------------------------- #
# Real chunk store
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def real_chunks():
    from backend.rag.indexing import load_chunks

    return load_chunks(get_settings().chunks_path)


@requires_index
def test_real_chunks_metadata_complete(real_chunks):
    assert len(real_chunks) > 100
    ids = [c.chunk_id for c in real_chunks]
    assert len(ids) == len(set(ids))
    for c in real_chunks:
        assert c.text.strip() and c.word_count > 0
        assert c.source_url.startswith("https://")
        assert c.publisher and c.publication_date and c.section_path
        assert c.pdf_page_start <= c.pdf_page_end


@requires_index
def test_real_a13_chunk_intact(real_chunks):
    """A-13 and its qualifier sub-items must be in the same chunk."""
    hits = [c for c in real_chunks if "A-13" in c.recommendation_ids]
    assert len(hits) == 1
    t = hits[0].text
    assert "A-13. A clinical assessment including the CIWA-Ar should be conducted at least every 8 hours" in t
    assert "repeat the CIWA-Ar at least every 6 hours" in t
    assert "More frequent monitoring should be conducted if clinically indicated" in t
    assert hits[0].page_end == "28"


@requires_index
def test_real_recommendation_ids_appear_once(real_chunks):
    from collections import Counter

    counts = Counter(rid for c in real_chunks for rid in c.recommendation_ids)
    dupes = {k: v for k, v in counts.items() if v > 1}
    assert not dupes, f"Recommendations split or duplicated across chunks: {dupes}"
