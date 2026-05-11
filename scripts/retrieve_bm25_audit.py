#!/usr/bin/env python3
"""Retrieve BM25 top-K candidates for Phase-3 audit queries.

Parses the corpus, builds a BM25-Okapi index (rank_bm25), then runs each of
the 100 audit queries through it and writes the top-K per query as a JSONL
matching the voyage/lateon output shape.

Tokenisation mirrors `scripts/pool_candidates.py` exactly so this is the
top-20 extension of the same Phase-2a BM25 retriever.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mamaretrieval.config import corpus_chunks_path, load_config
from mamaretrieval.corpus import iter_chunks


DEFAULT_TOP_K = 20


def tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-zA-Z0-9]", " ", text.lower()).split()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default=None)
    p.add_argument("--queries", default="data/queries.jsonl")
    p.add_argument("--query-ids", default="data/audit/query_ids.txt")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--output", default="data/audit/bm25_top20.jsonl")
    args = p.parse_args()

    config = load_config("config.yaml")
    corpus_path = (
        Path(args.corpus).expanduser() if args.corpus else corpus_chunks_path(config)
    )
    if not corpus_path.exists():
        print(f"ERROR: corpus not found at {corpus_path}", file=sys.stderr)
        return 1

    audit_ids = [
        line.strip()
        for line in Path(args.query_ids).read_text().splitlines()
        if line.strip()
    ]
    queries_by_id = {
        json.loads(line)["query_id"]: json.loads(line)
        for line in Path(args.queries).open()
    }
    missing = [qid for qid in audit_ids if qid not in queries_by_id]
    if missing:
        print(
            f"ERROR: {len(missing)} query_ids not in {args.queries}: {missing[:5]}",
            file=sys.stderr,
        )
        return 1
    audit_records = [queries_by_id[qid] for qid in audit_ids]

    print(f"Loading corpus from {corpus_path}...", flush=True)
    chunks = [c.to_dict() for c in iter_chunks(corpus_path)]
    print(f"  {len(chunks):,} chunks", flush=True)

    from rank_bm25 import BM25Okapi

    print(f"\nTokenising and building BM25 index...", flush=True)
    t0 = time.time()
    tokenised = [tokenize(c["text"]) for c in chunks]
    index = BM25Okapi(tokenised)
    chunk_ids = [c["chunk_id"] for c in chunks]
    print(f"  built in {time.time()-t0:.1f}s", flush=True)

    print(f"\nRetrieving top-{args.top_k} for {len(audit_records)} queries...", flush=True)
    t0 = time.time()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rec in audit_records:
            scores = index.get_scores(tokenize(rec["query_text"]))
            top_idx = np.argpartition(-scores, kth=args.top_k - 1)[: args.top_k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            payload = {
                "query_id": rec["query_id"],
                "retriever": "bm25",
                "top_k": args.top_k,
                "results": [
                    {
                        "chunk_id": chunk_ids[int(idx)],
                        "rank": rank + 1,
                        "score": float(scores[int(idx)]),
                    }
                    for rank, idx in enumerate(top_idx)
                ],
            }
            f.write(json.dumps(payload) + "\n")
    print(f"  retrieved in {time.time()-t0:.1f}s", flush=True)
    print(f"\nWrote {len(audit_records)} records to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
