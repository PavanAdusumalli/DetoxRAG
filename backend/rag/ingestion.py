"""PDF ingestion: extraction, cleaning, and section structure.

Pipeline per document:
  1. Extract text page by page with pypdf.
  2. Detect the printed page label (e.g. "27") and strip running headers.
  3. Clean text: drop table-extraction artifacts, normalize whitespace and
     glyphs, and repair kerning splits such as "CIW A-Ar" -> "CIWA-Ar".
  4. Parse the document's table of contents (dotted-leader lines) to get the
     chapter/section hierarchy and the printed page where each heading starts.
  5. Locate each TOC heading in the body text, in order, and tag every body
     line with its chapter and section.

Layout-specific details (TOC pages, header patterns, chapter detection,
excluded chapters) live in `data/guidelines/sources.json`, not in code.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from backend.rag.models import DocLine, SourceDocument, TocEntry

log = logging.getLogger(__name__)

TOC_LINE_RE = re.compile(r"^(?P<title>.+?)\s*\.{3,}\s*(?P<page>[0-9]+|[ivxlc]+)\s*$")
TABLE_ARTIFACT_RE = re.compile(r"^(None\s*)+$")
PRIVATE_USE_RE = re.compile(r"[-]")
FORM_BLANK_RE = re.compile(r"_{3,}")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-’']*")


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


def load_manifest(manifest_path: Path) -> list[SourceDocument]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [SourceDocument(**d) for d in data["documents"]]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_document(doc: SourceDocument, guidelines_dir: Path) -> Path:
    """Return the PDF path, raising if it is missing or its checksum differs."""
    path = guidelines_dir / doc.file
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/fetch_guidelines.py` first."
        )
    if doc.sha256:
        actual = sha256_file(path)
        if actual != doc.sha256:
            raise ValueError(
                f"Checksum mismatch for {doc.file}: expected {doc.sha256}, got {actual}. "
                "The publisher may have released a new version; review before re-indexing."
            )
    return path


# --------------------------------------------------------------------------- #
# Text cleaning
# --------------------------------------------------------------------------- #


def normalize_line(line: str) -> str:
    line = line.replace("\xa0", " ").replace("​", "")
    line = PRIVATE_USE_RE.sub("•", line)  # Symbol-font bullets (e.g. U+F0A2)
    line = FORM_BLANK_RE.sub("___", line)
    line = re.sub(r"[ \t]+", " ", line)
    return line.strip()


def build_vocabulary(texts: list[str]) -> Counter:
    """Word frequencies across a whole document (case-preserving)."""
    vocab: Counter = Counter()
    for t in texts:
        vocab.update(WORD_RE.findall(t))
    return vocab


def repair_kerning(text: str, vocab: Counter, min_count: int = 2) -> tuple[str, list[str]]:
    """Join words that PDF extraction split apart because of letter spacing.

    Two adjacent tokens `a b` are merged when `ab` occurs at least `min_count`
    times elsewhere in the same document AND at least one of `a`, `b` is not
    itself a word in the document. The second condition protects genuine
    two-word phrases ("in to", "per day") from being merged.

    Returns the repaired text and a list of the merges performed (for audit).
    """
    merges: list[str] = []

    def is_word(tok: str) -> bool:
        return vocab.get(tok, 0) >= min_count or vocab.get(tok.lower(), 0) >= min_count

    tokens = text.split(" ")
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if i + 1 < len(tokens):
            nxt = tokens[i + 1]
            a = re.sub(r"^[^\w]+", "", tok)
            b = re.sub(r"[^\w\-]+$", "", nxt)
            # b may begin with a hyphen: "CIWA -Ar" -> "CIWA-Ar"
            if a and b and a.isalpha() and (b[:1].isalpha() or (b[:1] == "-" and b[1:2].isalpha())):
                joined = a + b
                if vocab.get(joined, 0) >= min_count and (not is_word(a) or not is_word(b)):
                    merges.append(f"{tok} {nxt} -> {tok}{nxt}")
                    out.append(tok + nxt)
                    i += 2
                    continue
        out.append(tok)
        i += 1
    return " ".join(out), merges


# --------------------------------------------------------------------------- #
# Page extraction
# --------------------------------------------------------------------------- #


@dataclass
class PageText:
    pdf_page: int
    label: str | None
    lines: list[str] = field(default_factory=list)  # cleaned body lines (headers removed)


def _detect_label(raw_lines: list[str], patterns: list[re.Pattern], search: int) -> str | None:
    for line in raw_lines[:search]:
        for pat in patterns:
            m = pat.search(line)
            if m:
                return m.group(1)
    return None


def extract_pages(pdf_path: Path, parsing: dict) -> list[PageText]:
    """Extract each page's text, detect its printed label, and strip headers."""
    reader = PdfReader(str(pdf_path))
    label_pats = [re.compile(p) for p in parsing["page_label_patterns"]]
    header_pats = [re.compile(p) for p in parsing["header_line_patterns"]]
    label_search = parsing.get("page_label_search_lines", 1)
    header_search = parsing.get("header_search_lines", 3)
    inline_strip = re.compile(parsing["inline_header_strip_regex"]) if parsing.get("inline_header_strip_regex") else None

    pages: list[PageText] = []
    for idx, page in enumerate(reader.pages, start=1):
        raw = [normalize_line(l) for l in (page.extract_text() or "").splitlines()]
        raw = [l for l in raw if l]
        label = _detect_label(raw, label_pats, label_search)

        body: list[str] = []
        for i, line in enumerate(raw):
            if i < header_search and any(p.search(line) for p in header_pats):
                continue
            # e.g. running header glued to an appendix title: "77Guidelines for ... Jails"
            if inline_strip and i < header_search:
                line = inline_strip.sub("", line).strip()
            # Footer-style headers can appear at the bottom of a page.
            if i >= len(raw) - 2 and any(p.search(line) for p in header_pats[1:]):
                continue
            if TABLE_ARTIFACT_RE.match(line) or not line:
                continue
            body.append(line)
        pages.append(PageText(pdf_page=idx, label=label, lines=body))

    _infer_missing_labels(pages)
    return pages


def _infer_missing_labels(pages: list[PageText]) -> None:
    """Fill in labels for unlabeled pages that sit between numbered pages."""
    for i, p in enumerate(pages):
        if p.label is None and i > 0 and pages[i - 1].label and pages[i - 1].label.isdigit():
            p.label = str(int(pages[i - 1].label) + 1)


# --------------------------------------------------------------------------- #
# Table of contents
# --------------------------------------------------------------------------- #


def parse_toc(pdf_path: Path, parsing: dict) -> list[TocEntry]:
    reader = PdfReader(str(pdf_path))
    chapter_re = re.compile(parsing["chapter_regex"])
    skip_re = re.compile(parsing["toc_skip_regex"]) if parsing.get("toc_skip_regex") else None
    entries: list[TocEntry] = []
    for pno in parsing["toc_pdf_pages"]:
        text = reader.pages[pno - 1].extract_text() or ""
        for raw in text.splitlines():
            line = normalize_line(raw)
            m = TOC_LINE_RE.match(line)
            if not m:
                continue
            title = re.sub(r"\s+", " ", m.group("title")).strip()
            if skip_re and skip_re.match(title):
                continue
            entries.append(
                TocEntry(
                    title=title,
                    printed_page=m.group("page"),
                    is_chapter=bool(chapter_re.match(title)),
                )
            )
    return entries


# --------------------------------------------------------------------------- #
# Section assignment
# --------------------------------------------------------------------------- #


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().casefold()


def _page_index_by_label(pages: list[PageText]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, p in enumerate(pages):
        if p.label and p.label not in out:
            out[p.label] = i
    return out


def _find_heading(
    pages: list[PageText],
    title: str,
    start: tuple[int, int],
    target_page: int,
    tolerance: int = 1,
    max_lines: int = 3,
) -> tuple[int, int, int] | None:
    """Find `title` as a line (or a few consecutive lines) on/near `target_page`.

    Searches forward from `start` (page index, line index). Returns
    (page_index, line_index, n_lines_consumed) or None.
    """
    want = _norm(title)
    first_page = max(start[0], target_page)
    last_page = min(len(pages) - 1, target_page + tolerance)
    for pi in range(first_page, last_page + 1):
        lines = pages[pi].lines
        li0 = start[1] if pi == start[0] else 0
        for li in range(li0, len(lines)):
            cur = _norm(lines[li])
            if cur == want:
                return pi, li, 1
            # Headings that wrap across up to `max_lines` physical lines.
            if want.startswith(cur) and len(cur) > 8:
                joined = lines[li]
                for n in range(2, max_lines + 1):
                    if li + n - 1 >= len(lines):
                        break
                    joined = f"{joined} {lines[li + n - 1]}"
                    if _norm(joined) == want:
                        return pi, li, n
                    if not want.startswith(_norm(joined)):
                        break
    return None


def assign_sections(
    pages: list[PageText], toc: list[TocEntry], parsing: dict
) -> tuple[list[DocLine], list[str]]:
    """Tag body lines with chapter/section using the TOC.

    Returns (lines, warnings). Lines before the first TOC heading (front
    matter) and lines in excluded chapters are dropped.
    """
    by_label = _page_index_by_label(pages)
    exclude_re = re.compile(parsing["exclude_chapter_regex"]) if parsing.get("exclude_chapter_regex") else None
    warnings: list[str] = []

    # Resolve each TOC entry to a (page_index, line_index) boundary.
    boundaries: list[tuple[int, int, int, TocEntry]] = []
    cursor = (0, 0)
    for entry in toc:
        target = by_label.get(entry.printed_page)
        if target is None:
            warnings.append(f"TOC page '{entry.printed_page}' for '{entry.title}' not found in page labels")
            continue
        found = _find_heading(pages, entry.title, cursor, target)
        if found is None:
            # Fall back to the top of the target page (or the cursor if later).
            pi = max(target, cursor[0])
            li = cursor[1] if pi == cursor[0] else 0
            warnings.append(
                f"Heading '{entry.title}' not found as a line near p.{entry.printed_page}; "
                f"using start of PDF page {pages[pi].pdf_page}"
            )
            boundaries.append((pi, li, 0, entry))
            cursor = (pi, li)
        else:
            pi, li, n = found
            boundaries.append((pi, li, n, entry))
            cursor = (pi, li + n)

    # Walk the document and tag lines.
    out: list[DocLine] = []
    if not boundaries:
        return out, warnings + ["No TOC boundaries resolved"]

    heading_positions = {(b[0], b[1] + k) for b in boundaries for k in range(b[2])}
    # Optional explicit in-chapter subheadings (for documents whose TOC lists
    # only top-level sections). Matched as whole lines.
    subsections = {_norm(h): h for h in parsing.get("subsection_headings", [])}

    chapter: str | None = None
    section: str | None = None
    bi = 0
    for pi, page in enumerate(pages):
        for li, text in enumerate(page.lines):
            while bi < len(boundaries) and (boundaries[bi][0], boundaries[bi][1]) <= (pi, li):
                entry = boundaries[bi][3]
                if entry.is_chapter:
                    chapter, section = entry.title, None
                else:
                    section = entry.title
                bi += 1
            is_heading = (pi, li) in heading_positions
            if not is_heading and _norm(text) in subsections:
                section = subsections[_norm(text)]
                is_heading = True
            if chapter is None:
                continue  # front matter
            if exclude_re and exclude_re.match(chapter):
                continue
            out.append(
                DocLine(
                    text=text,
                    pdf_page=page.pdf_page,
                    page_label=page.label,
                    chapter=chapter,
                    section=section,
                    is_heading=is_heading,
                )
            )
    return out, warnings


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


@dataclass
class IngestedDocument:
    source: SourceDocument
    lines: list[DocLine]
    toc: list[TocEntry]
    n_pages: int
    warnings: list[str]
    kerning_merges: list[str]


def ingest_document(doc: SourceDocument, guidelines_dir: Path) -> IngestedDocument:
    pdf_path = verify_document(doc, guidelines_dir)
    pages = extract_pages(pdf_path, doc.parsing)
    toc = parse_toc(pdf_path, doc.parsing)
    if not toc:
        raise ValueError(f"No table-of-contents entries parsed for {doc.document_id}")

    # Kerning repair uses the whole document as its vocabulary.
    vocab = build_vocabulary([l for p in pages for l in p.lines])
    merges: list[str] = []
    for p in pages:
        repaired = []
        for line in p.lines:
            fixed, m = repair_kerning(line, vocab)
            merges.extend(m)
            repaired.append(fixed)
        p.lines = repaired

    lines, warnings = assign_sections(pages, toc, doc.parsing)
    for w in warnings:
        log.warning("[%s] %s", doc.document_id, w)
    log.info(
        "[%s] %d pages, %d TOC entries, %d body lines kept, %d kerning merges",
        doc.document_id, len(pages), len(toc), len(lines), len(merges),
    )
    return IngestedDocument(
        source=doc, lines=lines, toc=toc, n_pages=len(pages), warnings=warnings, kerning_merges=merges
    )
