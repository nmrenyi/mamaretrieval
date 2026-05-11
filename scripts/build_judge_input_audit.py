#!/usr/bin/env python3
"""Build the judge_relevance.py input file for the Phase 3 audit.

Reassembles data/audit/new_pairs_to_judge.jsonl (flat (q,c) pairs) and
data/queries_audit.jsonl (queries with text) into the per-query
{"query_id", "query_text", "candidates": [{"chunk_id": ...}]} format that
judge_relevance.py consumes.

Output: data/audit/candidates_audit_new_only.jsonl
  One record per audit query whose new_pairs list is non-empty.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new-pairs", default="data/audit/new_pairs_to_judge.jsonl")
    ap.add_argument("--queries", default="data/queries_audit.jsonl")
    ap.add_argument(
        "--output", default="data/audit/candidates_audit_new_only.jsonl"
    )
    args = ap.parse_args()

    queries_by_id = {
        json.loads(line)["query_id"]: json.loads(line)
        for line in Path(args.queries).open()
    }

    by_qid: dict[str, list[str]] = defaultdict(list)
    for line in Path(args.new_pairs).open():
        r = json.loads(line)
        by_qid[r["query_id"]].append(r["chunk_id"])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    total_pairs = 0
    n_queries = 0
    with out.open("w") as f:
        for qid, chunks in by_qid.items():
            if qid not in queries_by_id:
                print(f"ERROR: query_id {qid} not in {args.queries}", file=sys.stderr)
                return 1
            q = queries_by_id[qid]
            payload = {
                "query_id": qid,
                "query_text": q["query_text"],
                "candidates": [{"chunk_id": c} for c in chunks],
            }
            f.write(json.dumps(payload) + "\n")
            total_pairs += len(chunks)
            n_queries += 1

    print(f"Wrote {n_queries} queries / {total_pairs} (q,c) pairs to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
