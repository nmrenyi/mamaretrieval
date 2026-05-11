#!/usr/bin/env python3
"""Verify relevance_labels.jsonl integrity.

Checks every record against:
  1. score == d1_topic * (d2_meaningful + d3_actionable)
  2. d1_topic == False → d2_meaningful == False and d3_actionable == False
  3. d3_actionable == True → d2_meaningful == True

Exits 1 on any violation, 0 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def check(rec: dict) -> list[str]:
    d1 = bool(rec["d1_topic"])
    d2 = bool(rec["d2_meaningful"])
    d3 = bool(rec["d3_actionable"])
    score = rec["score"]
    errs = []
    expected = int(d1) * (int(d2) + int(d3))
    if score != expected:
        errs.append(f"score={score} but D1×(D2+D3)={expected}")
    if not d1 and (d2 or d3):
        errs.append(f"D1=False but D2={d2}, D3={d3}")
    if d3 and not d2:
        errs.append("D3=True but D2=False")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="data/relevance_labels.jsonl",
        help="Path to relevance_labels.jsonl",
    )
    ap.add_argument(
        "--max-violations",
        type=int,
        default=20,
        help="Maximum violations to print before truncating",
    )
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        return 2

    total = 0
    skipped_errors = 0
    violations: list[tuple[int, dict, list[str]]] = []

    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            total += 1
            rec = json.loads(line)
            if rec.get("score", 0) < 0:
                skipped_errors += 1
                continue
            errs = check(rec)
            if errs:
                violations.append((lineno, rec, errs))

    print(f"Records checked: {total}")
    if skipped_errors:
        print(f"Skipped error records (score<0): {skipped_errors}")
    print(f"Violations: {len(violations)}")

    for lineno, rec, errs in violations[: args.max_violations]:
        print(
            f"  line {lineno}  "
            f"qid={rec.get('query_id')}  cid={rec.get('chunk_id')}  "
            f"D1={rec['d1_topic']} D2={rec['d2_meaningful']} D3={rec['d3_actionable']} score={rec['score']}"
        )
        for e in errs:
            print(f"    - {e}")
    if len(violations) > args.max_violations:
        print(f"  ... +{len(violations) - args.max_violations} more")

    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
