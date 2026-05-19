#!/usr/bin/env python3
"""Extract per-retriever top-K rankings from the Phase 2b union candidates file.

Phase 2b stored bm25/medcpt/octen rankings inside `data/candidates.jsonl` —
each row is a query with a union list of candidate chunks tagged with
per-retriever rank fields. Phase 2b kept up to rank 10 per retriever.

This script materializes per-retriever top-K files in the same schema as
`data/audit/<retriever>_top20.jsonl` so that audit_metrics_v2.py can consume
them without modification.

Output schema (one line per query):
    {"query_id": "q_XXXXX", "retriever": "bm25", "top_k": N,
     "results": [{"chunk_id": "...", "rank": 1, "score": null}, ...]}

We don't carry the original retriever score (Phase 2b didn't preserve it for
bm25/medcpt/octen — only `rrf_score` over the union). Score field is left
null; downstream Variant D doesn't use it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_RETRIEVERS = ("bm25", "medcpt", "octen")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", default="data/candidates.jsonl")
    p.add_argument("--output-dir", default="data/full")
    p.add_argument("--top-k", type=int, default=10,
                   help="how many ranks to keep per retriever (max 10 in Phase 2b; up to 20 in Tier 3)")
    p.add_argument("--retrievers", default=",".join(DEFAULT_RETRIEVERS),
                   help="comma-separated subset to extract (default: bm25,medcpt,octen)")
    args = p.parse_args()
    retrievers = tuple(r.strip() for r in args.retrievers.split(",") if r.strip())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # rows[retriever] = list of output records
    rows: dict[str, list[str]] = {r: [] for r in retrievers}
    n_queries = 0

    for line in Path(args.candidates).open():
        r = json.loads(line)
        qid = r["query_id"]
        candidates = r["candidates"]
        n_queries += 1
        for retr in retrievers:
            field = f"{retr}_rank"
            ranked = [(c["chunk_id"], c[field]) for c in candidates if c.get(field) is not None]
            ranked.sort(key=lambda x: x[1])
            ranked = ranked[: args.top_k]
            rec = {
                "query_id": qid,
                "retriever": retr,
                "top_k": len(ranked),
                "results": [
                    {"chunk_id": cid, "rank": rk, "score": None} for cid, rk in ranked
                ],
            }
            rows[retr].append(json.dumps(rec))

    for retr in retrievers:
        path = out_dir / f"{retr}_top20.jsonl"
        path.write_text("\n".join(rows[retr]) + "\n")
        size_counts = [json.loads(line)["top_k"] for line in rows[retr]]
        avg = sum(size_counts) / len(size_counts) if size_counts else 0
        print(f"{retr}: wrote {len(rows[retr])}/{n_queries} queries -> {path}  "
              f"(avg {avg:.1f} chunks/query, max {max(size_counts) if size_counts else 0})",
              flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
