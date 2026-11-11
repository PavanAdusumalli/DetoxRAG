"""Section-aware chunking.

Chunks never cross a (chapter, section) boundary. Within a section, text is
first grouped into *units* that should stay together:

  * a numbered recommendation (e.g. "A-13. ...") with all of its lettered and
    roman-numeral sub-items, so a statement is never separated from its
    qualifiers;
  * a table (from its "Table X:" caption up to the next unit start);
  * a run of narrative text.

Units are then packed greedily into chunks of at most `max_words`. A unit that
is itself longer than `max_words` is split at sentence boundaries. When a
chunk boundary falls inside narrative text, the last sentences of the previous
chunk (up to `overlap_words`) are repeated at the start of the next one so
context isn't lost. Recommendations are self-contained and are not overlapped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend.rag.ingestion import IngestedDocument
from backend.rag.models import Chunk, DocLine

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9•(\"“])")
TABLE_CAPTION_RE = re.compile(r"^(Table|TABLE)\s+[A-Z]?-?\d+[.:]?")
SUBHEADING_RE = re.compile(r"^Recommendations?$")
LINE_END_HYPHEN_RE = re.compile(r"[A-Za-z]-$")


@dataclass
class Unit:
    kind: str  # "recommendation" | "table" | "narrative"
    lines: list[DocLine] = field(default_factory=list)
    recommendation_id: str | None = None

    @property
    def text(self) -> str:
        return join_lines([l.text for l in self.lines])

    @property
    def words(self) -> int:
        return len(self.text.split())


def join_lines(lines: list[str]) -> str:
    """Join PDF lines into flowing text.

    Bullets and enumerated sub-items keep their own line; ordinary wrapped
    lines are joined with a space.
    """
    out: list[str] = []
    for line in lines:
        starts_item = bool(re.match(r"^(•|o |[A-Z]\.\s|[ivx]+\.\s|\d+\.\s|[A-Z]{1,2}-\d+\.)", line))
        if out and not starts_item:
            out[-1] = f"{out[-1]} {line}"
        else:
            out.append(line)
    return "\n".join(out)


def build_units(lines: list[DocLine], rec_re: re.Pattern | None) -> list[Unit]:
    units: list[Unit] = []
    current: Unit | None = None
    for line in lines:
        if line.is_heading:
            continue  # heading text is carried in metadata/section_path
        text = line.text
        rec_match = rec_re.match(text) if rec_re else None
        if rec_match:
            current = Unit(kind="recommendation", recommendation_id=rec_match.group(1))
            units.append(current)
        elif TABLE_CAPTION_RE.match(text):
            current = Unit(kind="table")
            units.append(current)
        elif SUBHEADING_RE.match(text):
            current = Unit(kind="narrative")
            units.append(current)
        elif current is None:
            current = Unit(kind="narrative")
            units.append(current)
        # pypdf gives no paragraph breaks, so a unit (including a
        # recommendation and its sub-items) runs until the next marker.
        current.lines.append(line)
    return [u for u in units if u.lines]


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for block in text.split("\n"):
        parts.extend(s for s in SENTENCE_SPLIT_RE.split(block) if s.strip())
    return parts


def _tail_sentences(text: str, max_words: int) -> str:
    """Return the last sentences of `text` totalling at most `max_words`."""
    if max_words <= 0:
        return ""
    picked: list[str] = []
    total = 0
    for s in reversed(split_sentences(text)):
        n = len(s.split())
        if total + n > max_words:
            break
        picked.insert(0, s)
        total += n
    return " ".join(picked)


@dataclass
class _Piece:
    text: str
    lines: list[DocLine]
    recommendation_id: str | None
    kind: str


def _unit_pieces(unit: Unit, max_words: int) -> list[_Piece]:
    """Split a unit into pieces of at most `max_words` (sentence boundaries)."""
    if unit.words <= max_words:
        return [_Piece(unit.text, unit.lines, unit.recommendation_id, unit.kind)]
    pieces: list[_Piece] = []
    buf: list[str] = []
    count = 0
    for sent in split_sentences(unit.text):
        n = len(sent.split())
        if buf and count + n > max_words:
            pieces.append(_Piece(" ".join(buf), [], None, unit.kind))
            buf, count = [], 0
        buf.append(sent)
        count += n
    if buf:
        pieces.append(_Piece(" ".join(buf), [], None, unit.kind))
    # Attribute lines (for page ranges) by locating each piece's words in the unit lines.
    _assign_lines(pieces, unit.lines)
    if pieces:
        pieces[0].recommendation_id = unit.recommendation_id
    return pieces


def _assign_lines(pieces: list[_Piece], lines: list[DocLine]) -> None:
    """Map each piece back to the source lines it came from (by word offsets)."""
    line_spans: list[tuple[int, int, DocLine]] = []
    pos = 0
    for l in lines:
        n = len(l.text.split())
        line_spans.append((pos, pos + n, l))
        pos += n
    start = 0
    for p in pieces:
        end = start + len(p.text.split())
        p.lines = [l for (a, b, l) in line_spans if a < end and b > start] or lines[-1:]
        start = end


def chunk_document(
    doc: IngestedDocument, max_words: int = 220, overlap_words: int = 40
) -> list[Chunk]:
    src = doc.source
    rec_pattern = src.parsing.get("recommendation_id_regex")
    rec_re = re.compile(rec_pattern) if rec_pattern else None

    # Group lines by consecutive (chapter, section).
    groups: list[tuple[str | None, str | None, list[DocLine]]] = []
    for line in doc.lines:
        key = (line.chapter, line.section)
        if not groups or (groups[-1][0], groups[-1][1]) != key:
            groups.append((line.chapter, line.section, []))
        groups[-1][2].append(line)

    chunks: list[Chunk] = []
    for chapter, section, lines in groups:
        units = build_units(lines, rec_re)
        pieces = [p for u in units for p in _unit_pieces(u, max_words)]
        if not pieces:
            continue

        section_path = " > ".join(x for x in (src.short_title, chapter, section) if x)
        buf: list[_Piece] = []
        buf_words = 0

        def flush(carry: str = "") -> None:
            nonlocal buf, buf_words
            if not buf:
                return
            body = "\n".join(p.text for p in buf)
            text = f"{carry}\n{body}".strip() if carry else body
            src_lines = [l for p in buf for l in p.lines]
            labels = [l.page_label for l in src_lines if l.page_label]
            rec_ids = [p.recommendation_id for p in buf if p.recommendation_id]
            chunks.append(
                Chunk(
                    chunk_id=f"{src.document_id}-{len(chunks) + 1:04d}",
                    document_id=src.document_id,
                    document_title=src.title,
                    publisher=src.publisher,
                    publication_date=src.publication_date,
                    document_version=src.document_version,
                    source_url=src.source_url,
                    chapter=chapter,
                    section_heading=section,
                    section_path=section_path,
                    page_start=labels[0] if labels else None,
                    page_end=labels[-1] if labels else None,
                    pdf_page_start=min(l.pdf_page for l in src_lines),
                    pdf_page_end=max(l.pdf_page for l in src_lines),
                    recommendation_ids=rec_ids,
                    text=text,
                    word_count=len(text.split()),
                )
            )
            buf, buf_words = [], 0

        carry = ""
        for piece in pieces:
            n = len(piece.text.split())
            if buf and buf_words + n > max_words:
                last = buf[-1]
                flush(carry)
                # Overlap only when we cut through narrative text.
                carry = (
                    _tail_sentences(last.text, overlap_words)
                    if last.kind == "narrative" and piece.kind == "narrative"
                    else ""
                )
                buf_words = len(carry.split())
            buf.append(piece)
            buf_words += n
        flush(carry)

    return chunks
