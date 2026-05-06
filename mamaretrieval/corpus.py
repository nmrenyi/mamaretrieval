"""Parser for the production MAMAI RAG chunk corpus."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Iterator, Sequence


HEADER_RE = re.compile(
    r"^<sep>\[SOURCE:(?P<source>[^|]+)\|PAGE:(?P<page>\d+)\|CID:(?P<chunk_id>[a-f0-9]+)\]$"
)


class CorpusFormatError(ValueError):
    """Raised when a corpus file does not match the expected chunk format."""


@dataclass(frozen=True)
class CorpusChunk:
    """One parsed RAG corpus chunk."""

    chunk_id: str
    source: str
    page: int
    breadcrumb: str
    text: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation."""
        return asdict(self)


@dataclass(frozen=True)
class ChunkHeader:
    """Parsed metadata from a chunk header line."""

    source: str
    page: int
    chunk_id: str


def parse_header(line: str, line_number: int | None = None) -> ChunkHeader:
    """Parse a `<sep>[SOURCE:...|PAGE:...|CID:...]` header line."""
    match = HEADER_RE.match(line)
    if match is None:
        location = f" on line {line_number}" if line_number is not None else ""
        raise CorpusFormatError(f"Malformed chunk header{location}: {line!r}")
    return ChunkHeader(
        source=match.group("source"),
        page=int(match.group("page")),
        chunk_id=match.group("chunk_id"),
    )


def split_breadcrumb(raw_lines: Sequence[str]) -> tuple[str, str]:
    """Split leading breadcrumb line(s) from chunk text.

    Breadcrumbs are optional. If the first nonblank content line starts with
    `>`, all consecutive leading `>` lines are treated as breadcrumb metadata.
    Otherwise, the full content is returned as text with an empty breadcrumb.
    """
    lines = _strip_outer_blank_lines(raw_lines)
    if not lines:
        return "", ""

    breadcrumb_lines: list[str] = []
    index = 0
    while index < len(lines) and lines[index].lstrip().startswith(">"):
        breadcrumb_lines.append(lines[index].lstrip()[1:].strip())
        index += 1

    if not breadcrumb_lines:
        return "", "\n".join(lines)

    text_lines = _strip_outer_blank_lines(lines[index:])
    breadcrumb = " ".join(part for part in breadcrumb_lines if part).strip()
    return breadcrumb, "\n".join(text_lines)


def iter_chunks(path: str | Path) -> Iterator[CorpusChunk]:
    """Stream parsed chunks from a corpus file."""
    corpus_path = Path(path).expanduser()
    current_header: ChunkHeader | None = None
    current_lines: list[str] = []

    with corpus_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if line.startswith("<sep>"):
                if current_header is not None:
                    yield _build_chunk(current_header, current_lines)
                current_header = parse_header(line, line_number)
                current_lines = []
                continue

            if current_header is None:
                if line.strip():
                    raise CorpusFormatError(
                        f"Content before first chunk header on line {line_number}: {line!r}"
                    )
                continue

            current_lines.append(line)

    if current_header is not None:
        yield _build_chunk(current_header, current_lines)


def read_chunks(path: str | Path) -> list[CorpusChunk]:
    """Read all chunks from a corpus file into memory."""
    return list(iter_chunks(path))


def _build_chunk(header: ChunkHeader, raw_lines: Sequence[str]) -> CorpusChunk:
    breadcrumb, text = split_breadcrumb(raw_lines)
    return CorpusChunk(
        chunk_id=header.chunk_id,
        source=header.source,
        page=header.page,
        breadcrumb=breadcrumb,
        text=text,
    )


def _strip_outer_blank_lines(lines: Sequence[str]) -> list[str]:
    start = 0
    end = len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return list(lines[start:end])

