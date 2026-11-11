"""PDF extraction, cleaning, TOC parsing and metadata preservation."""

from __future__ import annotations

from collections import Counter

import pytest

from backend.config import get_settings
from backend.rag.ingestion import (
    ingest_document,
    load_manifest,
    normalize_line,
    parse_toc,
    repair_kerning,
    sha256_file,
)
from tests.conftest import requires_pdfs

# --------------------------------------------------------------------------- #
# Unit tests (no PDFs needed)
# --------------------------------------------------------------------------- #


def test_normalize_line_glyphs_and_whitespace():
    assert normalize_line("  Criteria\xa0for   SUDs ") == "• Criteria for SUDs"
    assert normalize_line("Signature ________________ Date") == "Signature ___ Date"


def test_repair_kerning_merges_split_words():
    vocab = Counter({"CIWA-Ar": 30, "care": 20, "Withdrawal": 40, "health": 25, "qualified": 10})
    text, merges = repair_kerning("If the CIW A-Ar is high, qualified health car e staff", vocab)
    assert text == "If the CIWA-Ar is high, qualified health care staff"
    assert len(merges) == 2


def test_repair_kerning_handles_hyphen_split():
    vocab = Counter({"CIWA-Ar": 30, "Repeat": 5})
    text, _ = repair_kerning("Repeat CIWA -Ar in one hour", vocab)
    assert text == "Repeat CIWA-Ar in one hour"


def test_repair_kerning_keeps_real_two_word_phrases():
    # "in" and "to" are both words, so "in to" must not become "into".
    vocab = Counter({"into": 10, "in": 100, "to": 100, "per": 10, "day": 10})
    text, merges = repair_kerning("went in to the room twice per day", vocab)
    assert text == "went in to the room twice per day"
    assert merges == []


# --------------------------------------------------------------------------- #
# Real documents
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def manifest():
    return {d.document_id: d for d in load_manifest(get_settings().sources_manifest)}


@pytest.fixture(scope="module")
def ingested(manifest):
    s = get_settings()
    return {doc_id: ingest_document(d, s.guidelines_dir) for doc_id, d in manifest.items()}


def test_manifest_has_required_metadata(manifest):
    for d in manifest.values():
        for field in ("document_id", "title", "publisher", "publication_date", "source_url", "sha256"):
            assert getattr(d, field), f"{d.document_id} missing {field}"


@requires_pdfs
def test_pdf_checksums_match_manifest(manifest):
    s = get_settings()
    for d in manifest.values():
        assert sha256_file(s.guidelines_dir / d.file) == d.sha256


@requires_pdfs
def test_toc_parsed_with_chapters(manifest):
    s = get_settings()
    bja = parse_toc(s.guidelines_dir / manifest["BJA-2023"].file, manifest["BJA-2023"].parsing)
    titles = [e.title for e in bja]
    assert "Alcohol Withdrawal" in titles and "Opioid Withdrawal" in titles
    chapters = [e.title for e in bja if e.is_chapter]
    assert chapters.index("Alcohol Withdrawal") < chapters.index("Opioid Withdrawal")
    assert "Recommendations" not in titles  # skipped via toc_skip_regex
    bop = parse_toc(s.guidelines_dir / manifest["BOP-2020"].file, manifest["BOP-2020"].parsing)
    assert any(e.title.startswith("7. ALCOHOL WITHDRAWAL") for e in bop)
    assert not any(e.title.startswith("TABLE") for e in bop)


@requires_pdfs
def test_extraction_yields_text_for_every_kept_document(ingested):
    for ing in ingested.values():
        assert len(ing.lines) > 500
        assert not any(l.text.strip() == "" for l in ing.lines)


@requires_pdfs
def test_running_headers_and_artifacts_removed(ingested):
    for ing in ingested.values():
        texts = {l.text for l in ing.lines}
        assert "BUREAU OF JUSTICE ASSISTANCE" not in texts
        assert "Guidelines for Managing Substance Withdrawal in Jails" not in texts
        assert "Clinical Guidance February 2020" not in texts
        assert not any(t.startswith("None None") for t in texts)


@requires_pdfs
def test_kerning_artifacts_repaired(ingested):
    body = " ".join(l.text for ing in ingested.values() for l in ing.lines)
    assert "CIW A-Ar" not in body
    assert "CIWA -Ar" not in body
    assert "CIWA-Ar" in body


@requires_pdfs
def test_excluded_chapters_absent(ingested):
    bja_chapters = {l.chapter for l in ingested["BJA-2023"].lines}
    assert "Endnotes" not in bja_chapters
    assert not any(c.startswith(("Appendix A:", "Appendix B:", "Appendix H:", "Appendix M:")) for c in bja_chapters)
    assert "Appendix E: Monitoring Patients During Withdrawal Management" in bja_chapters
    bop_chapters = {l.chapter for l in ingested["BOP-2020"].lines}
    assert "REFERENCES" not in bop_chapters
    assert not any(c.startswith("APPENDIX 5.") for c in bop_chapters)


@requires_pdfs
def test_known_passage_has_correct_page_and_section(ingested):
    """A-13 (CIWA-Ar reassessment interval) is on printed p. 28 / PDF p. 40."""
    lines = [l for l in ingested["BJA-2023"].lines if l.text.startswith("A-13.")]
    assert len(lines) == 1
    a13 = lines[0]
    assert (a13.page_label, a13.pdf_page) == ("28", 40)
    assert a13.chapter == "Alcohol Withdrawal"
    assert a13.section == "Monitoring Patients During Withdrawal Management"


@requires_pdfs
def test_bop_subsections_assigned(ingested):
    sections = {l.section for l in ingested["BOP-2020"].lines if l.chapter == "10. OPIOID WITHDRAWAL"}
    assert "TREATMENT WITH BUPRENORPHINE" in sections


@requires_pdfs
def test_page_labels_monotonic(ingested):
    for ing in ingested.values():
        pdf_pages = [l.pdf_page for l in ing.lines]
        assert pdf_pages == sorted(pdf_pages)
