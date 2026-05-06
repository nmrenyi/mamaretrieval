"""Chunk filtering and sampling for mamaretrieval."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
import random
import re
from typing import Any, Iterable, Sequence

from mamaretrieval.corpus import CorpusChunk


ROOT_SECTION = "__root__"
MIN_BODY_CHARS = 100


FRONT_MATTER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("suggested_citation", re.compile(r"\bsuggested citation\b", re.IGNORECASE)),
    ("endorsed_by", re.compile(r"\bendorsed by\b", re.IGNORECASE)),
    ("table_of_contents", re.compile(r"\btable of contents\b", re.IGNORECASE)),
    ("acknowledgements", re.compile(r"\backnowledg(e)?ments?\b", re.IGNORECASE)),
    ("foreword", re.compile(r"\bforeword\b", re.IGNORECASE)),
    ("copyright", re.compile(r"\bcopyright\b|\ball rights reserved\b", re.IGNORECASE)),
    ("license", re.compile(r"\blicen[cs]e\b|\bisbn\b", re.IGNORECASE)),
)

BOILERPLATE_HEADINGS = {
    "references",
    "reference",
    "bibliography",
    "contents",
    "table of contents",
    "in this chapter",
    "activity",
    "resources",
    "aims",
    "objectives",
    "instructions",
    "instructions for students",
    "summary of module",
    "acknowledgements",
    "acknowledgments",
    "thanks",
    "foreword",
    "preface",
    "suggested citation",
    "endorsed by",
    "how to use this book",
    "facilitator's schedule and preparation activities",
    "reflection on the trigger scenario",
}

STRUCTURAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("approved_by", re.compile(r"\bapproved by\b|\bversion no\.?\b", re.IGNORECASE)),
    (
        "cataloguing",
        re.compile(r"\bWHO Library Cataloguing\b|\bCataloguing-in-Publication\b", re.IGNORECASE),
    ),
    (
        "facilitator_schedule",
        re.compile(r"\bfacilitator'?s schedule\b|\bpreparation activities\b", re.IGNORECASE),
    ),
    ("case_number", re.compile(r"\bcase number\s*:", re.IGNORECASE)),
    (
        "instructions_for",
        re.compile(r"\binstructions for\b|\bguidelines for case study\b", re.IGNORECASE),
    ),
    (
        "reflection_trigger",
        re.compile(r"\breflection on the trigger scenario\b", re.IGNORECASE),
    ),
)


@dataclass(frozen=True)
class SampledChunk:
    """A sampled chunk with benchmark metadata."""

    chunk: CorpusChunk
    tier: str
    section: str

    def to_record(self) -> dict[str, object]:
        """Return the JSONL output schema for sampled chunks."""
        return {
            "chunk_id": self.chunk.chunk_id,
            "source": self.chunk.source,
            "tier": self.tier,
            "page": self.chunk.page,
            "breadcrumb": self.chunk.breadcrumb,
            "text": self.chunk.text,
        }


@dataclass(frozen=True)
class SourceTarget:
    """Sampling target for one source."""

    source: str
    tier: str
    queries_per_source: int
    chunks_to_sample: int


@dataclass
class SourceSamplingReport:
    """Summary of sampling work for one source."""

    source: str
    tier: str
    total_chunks: int
    filtered_chunks: int
    usable_chunks: int
    target_chunks: int
    sampled_chunks: int
    shortfall: int
    section_count: int
    filter_reasons: Counter[str] = field(default_factory=Counter)


def build_source_targets(config: dict[str, Any]) -> list[SourceTarget]:
    """Build source-level chunk targets from tier query budgets."""
    source_targets: list[SourceTarget] = []
    questions_per_chunk = int(config["queries"]["questions_per_chunk"])
    if questions_per_chunk <= 0:
        raise ValueError("queries.questions_per_chunk must be positive")

    source_tiers = config["source_tiers"]
    for tier_name, tier_config in source_tiers.items():
        queries_per_source = int(tier_config["queries_per_source"])
        chunks_to_sample = math.ceil(queries_per_source / questions_per_chunk)
        for source in tier_config["sources"]:
            source_targets.append(
                SourceTarget(
                    source=source,
                    tier=tier_name,
                    queries_per_source=queries_per_source,
                    chunks_to_sample=chunks_to_sample,
                )
            )
    return source_targets


def boilerplate_reason(chunk: CorpusChunk) -> str | None:
    """Return why a chunk should be filtered, or None if it is usable."""
    text = chunk.text.strip()
    plain_text = _plain_text(text)
    if len(plain_text) < MIN_BODY_CHARS:
        return "short_text"

    first_line = _first_nonblank_line(text)
    first_heading = _normalise_heading(first_line)
    if first_heading in BOILERPLATE_HEADINGS:
        return f"heading_{first_heading.replace(' ', '_')}"

    breadcrumb_section = top_level_section(chunk.breadcrumb).lower()
    if breadcrumb_section in BOILERPLATE_HEADINGS:
        return f"section_{breadcrumb_section.replace(' ', '_')}"

    leading_text = "\n".join(_nonblank_lines(text)[:5])
    for reason, pattern in FRONT_MATTER_PATTERNS:
        if pattern.search(leading_text):
            return reason

    for reason, pattern in STRUCTURAL_PATTERNS:
        if pattern.search(leading_text) or pattern.search(chunk.breadcrumb):
            return reason

    if _is_glossary_fragment(text):
        return "glossary_fragment"

    if _is_heading_only(text):
        return "heading_only"

    return None


def top_level_section(breadcrumb: str) -> str:
    """Return the first breadcrumb segment, or ROOT_SECTION."""
    if not breadcrumb.strip():
        return ROOT_SECTION
    section = breadcrumb.split(">", maxsplit=1)[0].strip()
    return section or ROOT_SECTION


def filter_usable_chunks(
    chunks: Sequence[CorpusChunk],
) -> tuple[list[CorpusChunk], Counter[str]]:
    """Filter obvious boilerplate chunks from one source."""
    usable: list[CorpusChunk] = []
    reasons: Counter[str] = Counter()
    for chunk in chunks:
        reason = boilerplate_reason(chunk)
        if reason is None:
            usable.append(chunk)
        else:
            reasons[reason] += 1
    return usable, reasons


def sample_source_chunks(
    chunks: Sequence[CorpusChunk],
    target: SourceTarget,
    rng: random.Random,
) -> tuple[list[SampledChunk], SourceSamplingReport]:
    """Filter and sample chunks for one source."""
    usable_chunks, filter_reasons = filter_usable_chunks(chunks)
    sample_count = min(target.chunks_to_sample, len(usable_chunks))
    shortfall = max(target.chunks_to_sample - len(usable_chunks), 0)

    if sample_count == len(usable_chunks):
        selected_pairs = list(enumerate(usable_chunks))
    else:
        selected_pairs = _stratified_sample_pairs(usable_chunks, sample_count, rng)

    selected_pairs.sort(key=lambda pair: pair[0])
    sampled = [
        SampledChunk(
            chunk=chunk,
            tier=target.tier,
            section=top_level_section(chunk.breadcrumb),
        )
        for _, chunk in selected_pairs
    ]

    section_count = len({top_level_section(chunk.breadcrumb) for chunk in usable_chunks})
    report = SourceSamplingReport(
        source=target.source,
        tier=target.tier,
        total_chunks=len(chunks),
        filtered_chunks=sum(filter_reasons.values()),
        usable_chunks=len(usable_chunks),
        target_chunks=target.chunks_to_sample,
        sampled_chunks=len(sampled),
        shortfall=shortfall,
        section_count=section_count,
        filter_reasons=filter_reasons,
    )
    return sampled, report


def sample_all_sources(
    chunks_by_source: dict[str, Sequence[CorpusChunk]],
    targets: Sequence[SourceTarget],
    random_seed: int,
) -> tuple[list[SampledChunk], list[SourceSamplingReport], list[str]]:
    """Sample all configured sources in target order."""
    rng = random.Random(random_seed)
    sampled_chunks: list[SampledChunk] = []
    reports: list[SourceSamplingReport] = []
    missing_sources: list[str] = []

    for target in targets:
        source_chunks = list(chunks_by_source.get(target.source, []))
        if not source_chunks:
            missing_sources.append(target.source)
        sampled, report = sample_source_chunks(source_chunks, target, rng)
        sampled_chunks.extend(sampled)
        reports.append(report)

    return sampled_chunks, reports, missing_sources


def group_chunks_by_source(
    chunks: Iterable[CorpusChunk], sources: Iterable[str]
) -> dict[str, list[CorpusChunk]]:
    """Collect only requested sources from a chunk stream."""
    wanted_sources = set(sources)
    chunks_by_source: dict[str, list[CorpusChunk]] = defaultdict(list)
    for chunk in chunks:
        if chunk.source in wanted_sources:
            chunks_by_source[chunk.source].append(chunk)
    return dict(chunks_by_source)


def _stratified_sample_pairs(
    chunks: Sequence[CorpusChunk], sample_count: int, rng: random.Random
) -> list[tuple[int, CorpusChunk]]:
    if sample_count <= 0:
        return []

    groups: dict[str, list[tuple[int, CorpusChunk]]] = defaultdict(list)
    for index, chunk in enumerate(chunks):
        groups[top_level_section(chunk.breadcrumb)].append((index, chunk))

    allocations = _proportional_allocations(
        {section: len(section_chunks) for section, section_chunks in groups.items()},
        sample_count,
    )

    selected: list[tuple[int, CorpusChunk]] = []
    for section in sorted(groups):
        count = allocations.get(section, 0)
        if count:
            selected.extend(rng.sample(groups[section], count))
    return selected


def _proportional_allocations(group_sizes: dict[str, int], target: int) -> dict[str, int]:
    """Allocate a target count across groups using largest remainders."""
    total = sum(group_sizes.values())
    if target < 0:
        raise ValueError("target must be non-negative")
    if total == 0 or target == 0:
        return {group: 0 for group in group_sizes}
    if target >= total:
        return dict(group_sizes)

    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for group, size in group_sizes.items():
        raw = target * size / total
        count = min(math.floor(raw), size)
        allocations[group] = count
        remainders.append((raw - count, group))

    remaining = target - sum(allocations.values())
    for _, group in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if remaining <= 0:
            break
        if allocations[group] < group_sizes[group]:
            allocations[group] += 1
            remaining -= 1

    while remaining > 0:
        progressed = False
        for group in sorted(group_sizes):
            if allocations[group] < group_sizes[group]:
                allocations[group] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break

    return allocations


def _is_heading_only(text: str) -> bool:
    lines = _nonblank_lines(text)
    if not lines:
        return True
    return all(_is_heading_line(line) for line in lines)


def _is_heading_line(line: str) -> bool:
    stripped = line.strip()
    if stripped.startswith("#"):
        return True
    if stripped.startswith("**") and stripped.endswith("**") and len(stripped) > 4:
        return True
    return False


def _is_glossary_fragment(text: str) -> bool:
    lines = _nonblank_lines(text)
    if not lines:
        return False

    first_heading = _normalise_heading(lines[0])
    if len(first_heading) == 1 and first_heading.isalpha():
        bold_definition_lines = sum(1 for line in lines[1:] if line.startswith("**"))
        return bold_definition_lines >= 2
    return False


def _first_nonblank_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _nonblank_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _normalise_heading(line: str) -> str:
    line = re.sub(r"^#+\s*", "", line.strip())
    line = line.strip("*_ ")
    line = re.sub(r"[:.]+$", "", line)
    return re.sub(r"\s+", " ", line).lower()


def _plain_text(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"\[[^\]]+\]\([^)]+\)", "", text)
    text = re.sub(r"[*_`#>|-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
