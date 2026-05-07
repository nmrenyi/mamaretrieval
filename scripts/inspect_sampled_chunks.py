#!/usr/bin/env python
"""Print sampled chunks for manual quality review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import sys
import textwrap
from typing import Any, Iterable


DEFAULT_INPUT_PATH = Path("data/sampled_chunks.jsonl")
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help=f"Sampled chunks JSONL path. Defaults to {DEFAULT_INPUT_PATH}.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum chunks to print. Defaults to 10.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start index after filtering for non-random review. Defaults to 0.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Print a random sample instead of the first matching chunks.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used with --random. Defaults to 42.",
    )
    parser.add_argument(
        "--source",
        help="Only print chunks from this source.",
    )
    parser.add_argument(
        "--section",
        help="Only print chunks whose section contains this text.",
    )
    parser.add_argument(
        "--contains",
        help="Only print chunks whose text or breadcrumb contains this text.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        help="Only print chunks with at least this many words.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        help="Only print chunks with at most this many words.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=100,
        help="Wrap text to this terminal width. Use 0 to disable wrapping.",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Print metadata only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Sampled chunks file not found: {input_path}", file=sys.stderr)
        return 1
    if args.limit < 0:
        print("--limit must be non-negative", file=sys.stderr)
        return 1
    if args.offset < 0:
        print("--offset must be non-negative", file=sys.stderr)
        return 1

    chunks = list(_matching_chunks(_read_jsonl(input_path), args))
    selected = _select_chunks(chunks, args)

    print(f"input={input_path}")
    print(f"matching_chunks={len(chunks)}")
    print(f"printed_chunks={len(selected)}")
    if args.random:
        print(f"random_seed={args.seed}")

    for display_index, chunk in enumerate(selected, start=1):
        _print_chunk(display_index, chunk, width=args.width, no_text=args.no_text)

    return 0


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def _matching_chunks(
    chunks: Iterable[dict[str, Any]], args: argparse.Namespace
) -> Iterable[dict[str, Any]]:
    contains = args.contains.lower() if args.contains else None
    section = args.section.lower() if args.section else None

    for index, chunk in enumerate(chunks, start=1):
        word_count = _word_count(str(chunk.get("text", "")))
        chunk["_review_index"] = index
        chunk["_word_count"] = word_count

        if args.source and chunk.get("source") != args.source:
            continue
        if section and section not in str(chunk.get("section", "")).lower():
            continue
        if args.min_words is not None and word_count < args.min_words:
            continue
        if args.max_words is not None and word_count > args.max_words:
            continue
        if contains:
            haystack = "\n".join(
                [
                    str(chunk.get("breadcrumb", "")),
                    str(chunk.get("section", "")),
                    str(chunk.get("text", "")),
                ]
            ).lower()
            if contains not in haystack:
                continue

        yield chunk


def _select_chunks(
    chunks: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    if args.limit == 0:
        return []
    if args.random:
        rng = random.Random(args.seed)
        sample_size = min(args.limit, len(chunks))
        return rng.sample(chunks, sample_size)
    return chunks[args.offset : args.offset + args.limit]


def _print_chunk(
    display_index: int, chunk: dict[str, Any], *, width: int, no_text: bool
) -> None:
    print("\n" + "=" * 100)
    print(
        f"{display_index}. row={chunk['_review_index']} "
        f"chunk_id={chunk.get('chunk_id', '')} words={chunk['_word_count']}"
    )
    print(f"source={chunk.get('source', '')}")
    print(f"tier={chunk.get('tier', '')} page={chunk.get('page', '')}")
    print(f"section={chunk.get('section', '')}")
    print(f"breadcrumb={chunk.get('breadcrumb', '')}")
    if no_text:
        return

    text = str(chunk.get("text", "")).strip()
    print("-" * 100)
    if width > 0:
        print(textwrap.fill(text, width=width, replace_whitespace=False))
    else:
        print(text)


def _word_count(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


if __name__ == "__main__":
    raise SystemExit(main())
