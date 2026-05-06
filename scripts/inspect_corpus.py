#!/usr/bin/env python
"""Inspect and validate the configured MAMAI RAG corpus."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mamaretrieval.config import configured_sources, corpus_chunks_path, load_config
from mamaretrieval.corpus import CorpusFormatError, iter_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML. Defaults to config.yaml.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    chunks_path = corpus_chunks_path(config)
    expected_sources = configured_sources(config)

    source_counts: Counter[str] = Counter()
    breadcrumb_counts: Counter[str] = Counter()
    total_chunks = 0
    chunks_with_breadcrumb = 0

    try:
        for chunk in iter_chunks(chunks_path):
            total_chunks += 1
            source_counts[chunk.source] += 1
            if chunk.breadcrumb:
                chunks_with_breadcrumb += 1
                breadcrumb_counts[chunk.source] += 1
    except CorpusFormatError as exc:
        print(f"Corpus format error: {exc}", file=sys.stderr)
        return 1

    found_sources = set(source_counts)
    missing = [source for source in expected_sources if source not in found_sources]

    print(f"Corpus path: {chunks_path}")
    print(f"Chunks: {total_chunks}")
    print(f"Sources: {len(source_counts)}")
    print(f"Chunks with breadcrumb: {chunks_with_breadcrumb}")
    print(f"Configured sources: {len(expected_sources)}")
    print(f"Configured sources found: {len(expected_sources) - len(missing)}")
    print(f"Configured sources missing: {len(missing)}")
    if missing:
        for source in missing:
            print(f"  missing: {source}")

    print("\nConfigured source counts:")
    for source in expected_sources:
        count = source_counts[source]
        breadcrumb_count = breadcrumb_counts[source]
        percent = (breadcrumb_count * 100 / count) if count else 0
        print(f"{count:5d} chunks  {breadcrumb_count:5d} breadcrumb  {percent:5.1f}%  {source}")

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

