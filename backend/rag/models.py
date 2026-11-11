"""Data models shared across the guideline RAG pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    """Bibliographic metadata for one guideline document (from sources.json)."""

    document_id: str
    file: str
    title: str
    short_title: str
    publisher: str
    publication_date: str
    document_version: str | None = None
    source_url: str
    sha256: str | None = None
    verified_on: str | None = None
    usage_note: str | None = None
    parsing: dict


class TocEntry(BaseModel):
    title: str
    printed_page: str
    is_chapter: bool


class DocLine(BaseModel):
    """One cleaned line of body text with its provenance."""

    text: str
    pdf_page: int  # 1-based physical page in the PDF
    page_label: str | None  # page number printed on the page, if detected
    chapter: str | None
    section: str | None
    is_heading: bool = False


class Chunk(BaseModel):
    """A retrievable passage with full citation metadata."""

    chunk_id: str
    document_id: str
    document_title: str
    publisher: str
    publication_date: str
    document_version: str | None
    source_url: str
    chapter: str | None
    section_heading: str | None
    section_path: str
    page_start: str | None
    page_end: str | None
    pdf_page_start: int
    pdf_page_end: int
    recommendation_ids: list[str] = Field(default_factory=list)
    text: str
    word_count: int

    def page_display(self) -> str:
        """Human-readable page reference using printed page numbers."""
        if self.page_start is None:
            return f"PDF p. {self.pdf_page_start}"
        if self.page_start == self.page_end or self.page_end is None:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}–{self.page_end}"

    def embedding_text(self) -> str:
        """Text used for embedding / keyword indexing: section context + body."""
        return f"{self.section_path}\n{self.text}"


RetrievalMethod = Literal["dense", "bm25", "hybrid"]


class RetrievedPassage(BaseModel):
    """A chunk returned by a retriever, with ranking diagnostics."""

    chunk: Chunk
    score: float
    rank: int
    method: RetrievalMethod
    dense_rank: int | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    bm25_score: float | None = None

    @property
    def citation_id(self) -> str:
        return self.chunk.chunk_id
