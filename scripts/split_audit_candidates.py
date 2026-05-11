#!/usr/bin/env python3
"""Split the union-format audit candidates into per-retriever JSONLs.

Reads data/audit/candidates_audit_shard0.jsonl (pool_candidates.py union
format with per-retriever ranks) and writes per-retriever top-K JSONLs
matching the schema used by the other audit retrievers (voyage/lateon/bm25).

For each retriever in `retrievers_used`, picks the candidates whose
`<retriever>_rank` is non-null, sorts by that rank, and writes one record
per query with the same `{query_id, model, top_k, results}` shape.

Per-retriever scores are not preserved by pool_candidates' union output —
only ranks are kept here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Map retriever name in the union file to display model name in the output.
RETRIEVER_MODEL = {
    "bm25": "BM25-Okapi",
    "medcpt": "ncbi/MedCPT-Query-Encoder",
    "octen": "Octen/Octen-Embedding-8B",
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/audit/candidates_audit_shard0.jsonl")
    p.add_argument("--output-dir", default="data/audit")
    p.add_argument(
        "--retrievers",
        nargs="+",
        default=["medcpt", "octen"],
        help="Which retrievers to extract (must be present in `retrievers_used`).",
    )
    args = p.parse_args()

    records = [json.loads(line) for line in Path(args.input).open()]
    print(f"Loaded {len(records)} query records from {args.input}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for retriever in args.retrievers:
        rank_key = f"{retriever}_rank"
        out_path = out_dir / f"{retriever}_top20.jsonl"
        with out_path.open("w") as f:
            for rec in records:
                cands = [c for c in rec["candidates"] if c.get(rank_key) is not None]
                cands.sort(key=lambda c: c[rank_key])
                payload = {
                    "query_id": rec["query_id"],
                    "model": RETRIEVER_MODEL.get(retriever, retriever),
                    "top_k": len(cands),
                    "results": [
                        {"chunk_id": c["chunk_id"], "rank": c[rank_key]}
                        for c in cands
                    ],
                }
                f.write(json.dumps(payload) + "\n")
        print(f"  wrote {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
