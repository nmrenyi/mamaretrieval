#!/usr/bin/env python3
"""Build the Tier 2 v2 candidates file — union of top-3 across 6 retrievers.

For each query in data/queries.jsonl: take each retriever's top-3 chunks
from data/full/<retriever>_top20.jsonl, union them, dedup, and write a
single record matching the Tier 1 schema (data/audit/candidates_v2_pilot.jsonl).

Output schema (one row per query):
    {"query_id": "q_XXXXX",
     "query_text": "...",
     "candidates": [{"chunk_id": "..."}, ...]}

Default output: data/audit/candidates_v2_full.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RETRIEVERS = ("bm25", "medcpt", "octen", "voyage", "lateon", "gecko")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--rankings-dir", default="data/full")
    ap.add_argument("--output", default="data/audit/candidates_v2_full.jsonl")
    ap.add_argument("--top-k", type=int, default=3, help="depth per retriever (default 3)")
    args = ap.parse_args()

    queries: dict[str, str] = {}
    for line in Path(args.queries).open():
        r = json.loads(line)
        queries[r["query_id"]] = r["query_text"]
    print(f"Queries: {len(queries)}", flush=True)

    # Load each retriever's top-K rankings keyed by qid
    per_retriever: dict[str, dict[str, list[str]]] = {}
    for retr in RETRIEVERS:
        path = Path(args.rankings_dir) / f"{retr}_top20.jsonl"
        if not path.exists():
            print(f"ERROR: missing {path}", file=sys.stderr)
            return 1
        m: dict[str, list[str]] = {}
        for line in path.open():
            rec = json.loads(line)
            m[rec["query_id"]] = [c["chunk_id"] for c in rec["results"][: args.top_k]]
        per_retriever[retr] = m
        avg = sum(len(v) for v in m.values()) / max(1, len(m))
        print(f"  {retr}: {len(m)} queries, avg {avg:.1f} chunks/query (capped at top-{args.top_k})",
              flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_pairs = 0
    sizes: list[int] = []
    with out_path.open("w") as f:
        for qid, qtext in queries.items():
            seen: set[str] = set()
            ordered: list[str] = []
            for retr in RETRIEVERS:
                for cid in per_retriever[retr].get(qid, []):
                    if cid not in seen:
                        seen.add(cid)
                        ordered.append(cid)
            sizes.append(len(ordered))
            total_pairs += len(ordered)
            rec = {
                "query_id": qid,
                "query_text": qtext,
                "candidates": [{"chunk_id": cid} for cid in ordered],
            }
            f.write(json.dumps(rec) + "\n")

    print(f"\nWrote {len(queries)} queries -> {out_path}")
    print(f"Total (q, c) pairs: {total_pairs}")
    print(f"Per-query candidate count: min={min(sizes)} max={max(sizes)} "
          f"mean={sum(sizes)/len(sizes):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
