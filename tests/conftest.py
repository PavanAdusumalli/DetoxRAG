"""Shared fixtures. Tests that need real artifacts skip cleanly when absent."""

from __future__ import annotations

import pytest

from backend.config import get_settings
from backend.rag.models import Chunk, DocLine, RetrievedPassage


def _have_pdfs() -> bool:
    s = get_settings()
    return (s.guidelines_dir / "bja_withdrawal_jails_2023.pdf").exists() and (
        s.guidelines_dir / "bop_medically_supervised_withdrawal.pdf"
    ).exists()


def _have_index() -> bool:
    s = get_settings()
    return s.chunks_path.exists() and s.bm25_dir.exists() and s.chroma_dir.exists()


def _have_embedder() -> bool:
    return (get_settings().embedding_path / "config.json").exists()


def _have_llm() -> bool:
    return (get_settings().llm_path / "config.json").exists()


requires_pdfs = pytest.mark.skipif(not _have_pdfs(), reason="guideline PDFs not downloaded")
requires_index = pytest.mark.skipif(
    not (_have_index() and _have_embedder()), reason="index or embedding model not available"
)
requires_llm = pytest.mark.skipif(not _have_llm(), reason="local LLM not downloaded")


def make_chunk(chunk_id: str = "TST-2024-0001", text: str = "Sample text.", **kw) -> Chunk:
    defaults = dict(
        chunk_id=chunk_id,
        document_id=chunk_id.rsplit("-", 1)[0],
        document_title="Test Guideline",
        publisher="Test Publisher",
        publication_date="2024-01",
        document_version=None,
        source_url="https://example.org/test.pdf",
        chapter="Chapter",
        section_heading="Section",
        section_path="Test > Chapter > Section",
        page_start="1",
        page_end="1",
        pdf_page_start=5,
        pdf_page_end=5,
        recommendation_ids=[],
        text=text,
        word_count=len(text.split()),
    )
    defaults.update(kw)
    return Chunk(**defaults)


def make_passage(chunk: Chunk, rank: int = 1, method: str = "hybrid") -> RetrievedPassage:
    return RetrievedPassage(chunk=chunk, score=1.0 / rank, rank=rank, method=method)


def make_line(text: str, page: int = 1, chapter: str = "Ch", section: str | None = "Sec", heading=False) -> DocLine:
    return DocLine(
        text=text, pdf_page=page, page_label=str(page), chapter=chapter, section=section, is_heading=heading
    )
