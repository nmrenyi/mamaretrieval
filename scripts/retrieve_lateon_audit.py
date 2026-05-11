#!/usr/bin/env python3
"""Retrieve LateOn (ColBERT late-interaction) top-K candidates for audit queries.

Builds a PLAID index over the full corpus (one-time, cached under
.cache/lateon_plaid_index/), then encodes the 100 audit queries with
`is_query=True` and runs ColBERT MaxSim retrieval.

Why a separate venv: pylate 1.5.0 pins `fast-plaid <= 1.3.0.290`, and several
of its transitive deps lack Python 3.14 wheels. Run with the dedicated 3.12
environment:

    .venv-lateon/bin/python scripts/retrieve_lateon_audit.py

Output: data/audit/lateon_top20.jsonl (one record per audit query).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mamaretrieval.config import corpus_chunks_path, load_config
from mamaretrieval.corpus import iter_chunks


DEFAULT_MODEL = "lightonai/GTE-ModernColBERT-v1"
DEFAULT_TOP_K = 20
DEFAULT_INDEX_FOLDER = ".cache/lateon_plaid_index"
DEFAULT_INDEX_NAME = "mamaretrieval_corpus"


def detect_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default=None)
    p.add_argument("--queries", default="data/queries.jsonl")
    p.add_argument("--query-ids", default="data/audit/query_ids.txt")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--index-folder", default=DEFAULT_INDEX_FOLDER)
    p.add_argument("--index-name", default=DEFAULT_INDEX_NAME)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None, help="cpu, mps, cuda; default = auto")
    p.add_argument(
        "--limit-chunks",
        type=int,
        default=None,
        help="encode only the first N corpus chunks (debugging)",
    )
    p.add_argument(
        "--rebuild-index",
        action="store_true",
        help="rebuild the PLAID index even if it exists",
    )
    p.add_argument("--output", default="data/audit/lateon_top20.jsonl")
    args = p.parse_args()

    from pylate import indexes, models, retrieve

    device = args.device or detect_device()
    print(f"Device: {device}", flush=True)

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

    print(f"Loading ColBERT model: {args.model}", flush=True)
    model = models.ColBERT(model_name_or_path=args.model, device=device)

    index_folder = Path(args.index_folder)
    index_path = index_folder / args.index_name
    has_index = index_path.exists() and any(index_path.iterdir())
    needs_build = args.rebuild_index or not has_index

    if needs_build:
        print(f"Loading corpus from {corpus_path}...", flush=True)
        chunks = [c.to_dict() for c in iter_chunks(corpus_path)]
        if args.limit_chunks:
            chunks = chunks[: args.limit_chunks]
        print(f"  {len(chunks):,} chunks", flush=True)

        print(f"\nEncoding corpus with {args.model} on {device}...", flush=True)
        t0 = time.time()
        documents_embeddings = model.encode(
            [c["text"] for c in chunks],
            batch_size=args.batch_size,
            is_query=False,
            show_progress_bar=True,
        )
        print(f"  encoded in {time.time()-t0:.1f}s", flush=True)

        print(f"\nBuilding PLAID index at {index_path}...", flush=True)
        t0 = time.time()
        index_folder.mkdir(parents=True, exist_ok=True)
        index = indexes.PLAID(
            index_folder=str(index_folder),
            index_name=args.index_name,
            override=True,
        )
        index.add_documents(
            documents_ids=[c["chunk_id"] for c in chunks],
            documents_embeddings=documents_embeddings,
        )
        print(f"  index built in {time.time()-t0:.1f}s", flush=True)
    else:
        print(f"Reusing existing PLAID index at {index_path}", flush=True)
        index = indexes.PLAID(
            index_folder=str(index_folder),
            index_name=args.index_name,
            override=False,
        )

    print(f"\nEncoding {len(audit_records)} audit queries...", flush=True)
    t0 = time.time()
    queries_embeddings = model.encode(
        [r["query_text"] for r in audit_records],
        batch_size=args.batch_size,
        is_query=True,
        show_progress_bar=True,
    )
    print(f"  {time.time()-t0:.1f}s", flush=True)

    print(f"\nRetrieving top-{args.top_k}...", flush=True)
    t0 = time.time()
    retriever = retrieve.ColBERT(index=index)
    search_results = retriever.retrieve(
        queries_embeddings=queries_embeddings,
        k=args.top_k,
    )
    print(f"  {time.time()-t0:.1f}s", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rec, results in zip(audit_records, search_results):
            payload = {
                "query_id": rec["query_id"],
                "model": args.model,
                "top_k": args.top_k,
                "results": [
                    {
                        "chunk_id": str(r["id"]),
                        "rank": rank + 1,
                        "score": float(r["score"]),
                    }
                    for rank, r in enumerate(results)
                ],
            }
            f.write(json.dumps(payload) + "\n")

    print(f"\nWrote {len(audit_records)} records to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
