#!/usr/bin/env python3
"""Retrieve voyage-4-large top-K candidates for the Phase-3 audit queries.

Loads the cached corpus embeddings (produced by embed_voyage_corpus.py),
embeds the audit queries with input_type=query, computes cosine similarity
(dot product on L2-normalized vectors), and writes the top-K chunk IDs per
query to a per-retriever JSONL.

The output is ordered by descending score; rank 1 is the most similar.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ENDPOINT = "https://api.voyageai.com/v1/embeddings"
DEFAULT_MODEL = "voyage-4-large"
DEFAULT_OUTPUT_DIM = 2048
DEFAULT_TOP_K = 20


def embed_queries(
    texts: list[str],
    api_key: str,
    model: str,
    output_dim: int,
    max_retries: int = 5,
) -> tuple[np.ndarray, int]:
    body = {
        "input": texts,
        "model": model,
        "input_type": "query",
        "output_dimension": output_dim,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    backoff = 2.0
    for attempt in range(max_retries):
        try:
            r = requests.post(ENDPOINT, headers=headers, json=body, timeout=120)
            if r.status_code == 429 or 500 <= r.status_code < 600:
                raise requests.HTTPError(f"{r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            d = r.json()
            embs = np.array([item["embedding"] for item in d["data"]], dtype=np.float32)
            return embs, int(d["usage"]["total_tokens"])
        except (requests.RequestException, ValueError, KeyError) as e:
            if attempt == max_retries - 1:
                raise
            print(
                f"    retry {attempt+1}/{max_retries} after {backoff:.0f}s: {type(e).__name__}",
                flush=True,
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
    raise RuntimeError("unreachable")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queries", default="data/queries.jsonl")
    p.add_argument("--query-ids", default="data/audit/query_ids.txt")
    p.add_argument("--cache-dir", default=".cache")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output-dim", type=int, default=DEFAULT_OUTPUT_DIM)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--output", default="data/audit/voyage_top20.jsonl")
    p.add_argument("--api-key-env", default="MAMAI_VOYAGE_API")
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    stem = f"voyage4large_dim{args.output_dim}"
    corpus_npy = cache_dir / f"{stem}_corpus.npy"
    chunk_ids_path = cache_dir / f"{stem}_chunk_ids.txt"
    if not corpus_npy.exists():
        print(
            f"ERROR: corpus embeddings not found at {corpus_npy}. "
            f"Run scripts/embed_voyage_corpus.py first.",
            file=sys.stderr,
        )
        return 1

    print(f"Loading corpus embeddings from {corpus_npy}...", flush=True)
    corpus_embs = np.load(corpus_npy)
    chunk_ids = chunk_ids_path.read_text().splitlines()
    if corpus_embs.shape[0] != len(chunk_ids):
        print(
            f"ERROR: emb rows ({corpus_embs.shape[0]}) != chunk_ids "
            f"({len(chunk_ids)})",
            file=sys.stderr,
        )
        return 1
    print(f"  {corpus_embs.shape[0]:,} chunks, dim={corpus_embs.shape[1]}", flush=True)

    audit_ids = [
        line.strip()
        for line in Path(args.query_ids).read_text().splitlines()
        if line.strip()
    ]
    print(f"Audit queries to retrieve: {len(audit_ids)}", flush=True)

    queries_by_id = {}
    for line in Path(args.queries).open():
        rec = json.loads(line)
        queries_by_id[rec["query_id"]] = rec
    missing = [qid for qid in audit_ids if qid not in queries_by_id]
    if missing:
        print(f"ERROR: {len(missing)} query_ids not found in {args.queries}: {missing[:5]}", file=sys.stderr)
        return 1

    audit_records = [queries_by_id[qid] for qid in audit_ids]
    texts = [r["query_text"] for r in audit_records]

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"ERROR: env var {args.api_key_env} is not set", file=sys.stderr)
        return 1

    print(f"\nEmbedding {len(texts)} queries with {args.model} (input_type=query)...", flush=True)
    t0 = time.time()
    query_embs, query_tokens = embed_queries(
        texts, api_key, args.model, args.output_dim
    )
    print(
        f"  {query_embs.shape}, {query_tokens} tokens, "
        f"{time.time()-t0:.1f}s",
        flush=True,
    )
    norms = np.linalg.norm(query_embs, axis=1, keepdims=True)
    query_embs = query_embs / np.maximum(norms, 1e-12)

    print(f"\nRunning cosine top-{args.top_k} retrieval...", flush=True)
    t0 = time.time()
    # scores[i, j] = similarity(query i, chunk j)
    scores = query_embs @ corpus_embs.T  # (Q, N)
    k = args.top_k
    # argpartition then sort the partitioned slice for ranked top-k
    top_idx_unsorted = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    top_scores_unsorted = np.take_along_axis(scores, top_idx_unsorted, axis=1)
    sort_within = np.argsort(-top_scores_unsorted, axis=1)
    top_idx = np.take_along_axis(top_idx_unsorted, sort_within, axis=1)
    top_scores = np.take_along_axis(top_scores_unsorted, sort_within, axis=1)
    print(f"  {time.time()-t0:.2f}s", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for qi, rec in enumerate(audit_records):
            results = [
                {
                    "chunk_id": chunk_ids[int(idx)],
                    "rank": rank + 1,
                    "score": float(top_scores[qi, rank]),
                }
                for rank, idx in enumerate(top_idx[qi])
            ]
            f.write(
                json.dumps(
                    {
                        "query_id": rec["query_id"],
                        "model": args.model,
                        "output_dimension": args.output_dim,
                        "top_k": args.top_k,
                        "results": results,
                    }
                )
                + "\n"
            )

    print(f"\nWrote {len(audit_records)} records to {out_path}", flush=True)
    print(f"Query embedding tokens used: {query_tokens}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
