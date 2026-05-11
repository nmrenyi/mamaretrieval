#!/usr/bin/env python3
"""Sample audit queries by proportional stratification across tiers.

Allocates a total quota across tiers using Hamilton's largest-remainder method
(strict proportional to each tier's share of the input), then samples without
replacement within each tier. Output is the deterministic union, sorted by
query_id for diff stability.

Writes:
  - <output-dir>/query_ids.txt        one query_id per line, sorted
  - <output-dir>/sample_breakdown.json  per-tier and per-source counts
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

TIER_ORDER = ["very_high", "high", "moderate_high", "moderate", "low_moderate"]


def hamilton_allocate(populations: dict[str, int], total: int) -> dict[str, int]:
    grand_total = sum(populations.values())
    raw = {k: total * v / grand_total for k, v in populations.items()}
    floor_alloc = {k: int(v) for k, v in raw.items()}
    remainder = total - sum(floor_alloc.values())
    by_frac = sorted(
        populations.keys(),
        key=lambda k: (raw[k] - floor_alloc[k], populations[k]),
        reverse=True,
    )
    for k in by_frac[:remainder]:
        floor_alloc[k] += 1
    return floor_alloc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/queries.jsonl")
    ap.add_argument("--output-dir", default="data/audit")
    ap.add_argument("--total", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    queries = [json.loads(line) for line in Path(args.input).open()]
    by_tier: dict[str, list[dict]] = defaultdict(list)
    for q in queries:
        by_tier[q["tier"]].append(q)

    populations = {t: len(by_tier[t]) for t in by_tier}
    targets = hamilton_allocate(populations, args.total)

    # low_moderate has only 5 queries in the corpus (0.16% share), so strict
    # proportional drops it to 0. Bump to 1 so the tier is represented at all;
    # donor is `moderate` (the smallest non-empty over-allocated tier).
    if populations.get("low_moderate", 0) > 0 and targets.get("low_moderate", 0) == 0:
        targets["low_moderate"] = 1
        targets["moderate"] -= 1

    for t, n in targets.items():
        if n > populations[t]:
            print(
                f"ERROR: tier {t} target {n} exceeds population {populations[t]}",
                file=sys.stderr,
            )
            return 1

    rng = random.Random(args.seed)
    selected: list[dict] = []
    for tier in TIER_ORDER:
        n = targets.get(tier, 0)
        if n == 0:
            continue
        pool = sorted(by_tier[tier], key=lambda q: q["query_id"])
        selected.extend(rng.sample(pool, n))

    selected.sort(key=lambda q: q["query_id"])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids_path = out_dir / "query_ids.txt"
    with ids_path.open("w") as f:
        for q in selected:
            f.write(q["query_id"] + "\n")

    tier_breakdown = []
    for tier in TIER_ORDER:
        pop = populations.get(tier, 0)
        if pop == 0:
            continue
        tier_breakdown.append(
            {
                "tier": tier,
                "population": pop,
                "share_pct": round(100 * pop / sum(populations.values()), 2),
                "sampled": targets.get(tier, 0),
            }
        )

    source_counts = Counter((q["tier"], q["source"]) for q in selected)
    source_breakdown = [
        {"tier": t, "source": s, "sampled": n}
        for (t, s), n in sorted(source_counts.items(), key=lambda x: (-x[1], x[0]))
    ]

    manifest = {
        "input": str(args.input),
        "total": args.total,
        "seed": args.seed,
        "allocation_method": "hamilton_largest_remainder",
        "tier_breakdown": tier_breakdown,
        "source_breakdown": source_breakdown,
    }
    (out_dir / "sample_breakdown.json").write_text(json.dumps(manifest, indent=2))

    print(f"Wrote {len(selected)} query_ids to {ids_path}")
    print()
    print(f"{'Tier':<16} {'Pop':>6} {'Share':>8} {'Sampled':>9}")
    for row in tier_breakdown:
        print(
            f"{row['tier']:<16} {row['population']:>6} "
            f"{row['share_pct']:>7}% {row['sampled']:>9}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
