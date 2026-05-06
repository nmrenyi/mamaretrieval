#!/usr/bin/env python
"""Sample benchmark seed chunks from the configured MAMAI RAG corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mamaretrieval.config import corpus_chunks_path, load_config
from mamaretrieval.corpus import iter_chunks
from mamaretrieval.sampling import (
    SourceSamplingReport,
    build_source_targets,
    group_chunks_by_source,
    sample_all_sources,
)


DEFAULT_OUTPUT_PATH = Path("data/sampled_chunks.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML. Defaults to config.yaml.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Output JSONL path. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sampling report without writing sampled chunks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    targets = build_source_targets(config)
    random_seed = int(config["queries"].get("random_seed", 42))
    chunks_path = corpus_chunks_path(config)

    chunks_by_source = group_chunks_by_source(
        iter_chunks(chunks_path), [target.source for target in targets]
    )
    sampled_chunks, reports, missing_sources = sample_all_sources(
        chunks_by_source=chunks_by_source,
        targets=targets,
        random_seed=random_seed,
    )

    _print_report(reports, missing_sources)

    if args.dry_run:
        print("\nDry run: no output written.")
        return 1 if missing_sources else 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sampled_chunk in sampled_chunks:
            handle.write(
                json.dumps(sampled_chunk.to_record(), ensure_ascii=False, sort_keys=True)
                + "\n"
            )

    print(f"\nWrote {len(sampled_chunks)} sampled chunks to {output_path}")
    return 1 if missing_sources else 0


def _print_report(
    reports: list[SourceSamplingReport], missing_sources: list[str]
) -> None:
    print(
        "source | tier | total | filtered | usable | sections | target | sampled | shortfall"
    )
    print("-" * 92)
    for report in reports:
        print(
            f"{report.source} | {report.tier} | {report.total_chunks} | "
            f"{report.filtered_chunks} | {report.usable_chunks} | "
            f"{report.section_count} | {report.target_chunks} | "
            f"{report.sampled_chunks} | {report.shortfall}"
        )

    total_sampled = sum(report.sampled_chunks for report in reports)
    total_filtered = sum(report.filtered_chunks for report in reports)
    print("-" * 92)
    print(f"total_sampled={total_sampled} total_filtered={total_filtered}")

    if missing_sources:
        print("\nMissing configured sources:", file=sys.stderr)
        for source in missing_sources:
            print(f"  {source}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())

