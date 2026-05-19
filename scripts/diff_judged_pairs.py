#!/usr/bin/env python3
"""Subtract already-judged (q, c) pairs from a candidates file.

Used to build the Tier 3 incremental candidates set: take the full top-20
union (~160k pairs), subtract the 36k pairs already in v2_full_h100.jsonl
(Tier 2 top-3 union), and write only the new pairs to be judged.

Inputs:
    --candidates   data/audit/candidates_v2_top20.jsonl  (full union)
    --judged       data/audit/v2_full_h100.jsonl         (existing labels)

Output:
    --output       data/audit/candidates_v2_top20_new.jsonl  (only new pairs)

Output rows where all chunks were already judged are dropped entirely.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--judged", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    judged_pairs: set[tuple[str, str]] = set()
    for line in Path(args.judged).open():
        r = json.loads(line)
        judged_pairs.add((r["query_id"], r["chunk_id"]))
    print(f"Existing judged pairs: {len(judged_pairs)}", flush=True)

    n_in_queries = 0
    n_out_queries = 0
    n_in_pairs = 0
    n_out_pairs = 0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as out_f:
        for line in Path(args.candidates).open():
            rec = json.loads(line)
            qid = rec["query_id"]
            n_in_queries += 1
            new_chunks = [
                c for c in rec["candidates"]
                if (qid, c["chunk_id"]) not in judged_pairs
            ]
            n_in_pairs += len(rec["candidates"])
            n_out_pairs += len(new_chunks)
            if new_chunks:
                rec["candidates"] = new_chunks
                out_f.write(json.dumps(rec) + "\n")
                n_out_queries += 1

    print(f"Input:  {n_in_queries} queries, {n_in_pairs} (q,c) pairs", flush=True)
    print(f"Output: {n_out_queries} queries with new chunks, "
          f"{n_out_pairs} new (q,c) pairs", flush=True)
    print(f"Skipped: {n_in_pairs - n_out_pairs} already-judged pairs", flush=True)
    print(f"Wrote -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
