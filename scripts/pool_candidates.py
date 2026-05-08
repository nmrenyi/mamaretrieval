#!/usr/bin/env python3
"""Phase 2a — Build retrieval candidate pool.

For each query, runs BM25, MedCPT, and Octen-Embedding-8B retrievers,
applies RRF fusion, force-includes the seed chunk, and outputs candidate
pools for Phase 2b LLM relevance judging.

Usage:
  python scripts/pool_candidates.py [options]

Options:
  --queries PATH       default: data/queries.jsonl
  --corpus PATH        default: from config.yaml
  --output PATH        default: data/candidates.jsonl
  --config PATH        default: config.yaml
  --retrievers LIST    comma-separated subset of: bm25,medcpt,octen (default: all)
  --top-k N            top-k per retriever (default: from config.yaml, usually 10)
  --batch-size N       embedding batch size (default: 64)
  --device DEVICE      cuda / cpu / cuda:N (default: cuda if available)
  --cache-dir PATH     directory for embedding caches (default: .cache)
  --shard INDEX COUNT  process 1/COUNT of queries starting at INDEX (0-based)
  --resume             skip query_ids already written to output
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# ── Corpus parsing ─────────────────────────────────────────────────────────────

_SEP_RE = re.compile(
    r"<sep>\[SOURCE:([^|]+)\|PAGE:([^|]+)\|CID:([^\]]+)\]"
)


def parse_corpus(path: str | Path) -> list[dict[str, Any]]:
    """Parse chunks_for_rag.txt into chunk records."""
    path = Path(path).expanduser()
    chunks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    buf: list[str] = []

    def _flush() -> None:
        if current is not None:
            current["text"] = "\n".join(buf).strip()
            chunks.append(current)
        buf.clear()

    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _SEP_RE.match(line.rstrip())
            if m:
                _flush()
                current = {
                    "source": m.group(1),
                    "page": m.group(2),
                    "chunk_id": m.group(3),
                }
            else:
                buf.append(line.rstrip())

    _flush()
    return chunks


# ── BM25 ───────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-zA-Z0-9]", " ", text.lower()).split()


def build_bm25(chunks: list[dict]) -> Any:
    from rank_bm25 import BM25Okapi
    corpus = [_tokenize(c["text"]) for c in chunks]
    print(f"  Building BM25 index over {len(corpus)} chunks...", flush=True)
    return BM25Okapi(corpus)


def retrieve_bm25(
    index: Any,
    chunk_ids: list[str],
    query: str,
    top_k: int,
) -> list[tuple[str, int]]:
    """Returns list of (chunk_id, rank) sorted by rank ascending (rank 1 = best)."""
    scores = index.get_scores(_tokenize(query))
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(chunk_ids[i], rank + 1) for rank, i in enumerate(top_idx)]


# ── MedCPT ─────────────────────────────────────────────────────────────────────

def build_medcpt_corpus_embeddings(
    chunks: list[dict],
    device: str,
    batch_size: int,
    cache_path: Path,
) -> np.ndarray:
    """Encode corpus with MedCPT-Article-Encoder. Returns (N, D) float32."""
    if cache_path.exists():
        print(f"  Loading MedCPT corpus embeddings from {cache_path}", flush=True)
        return np.load(cache_path)

    import torch
    from transformers import AutoModel, AutoTokenizer

    print("  Encoding corpus with ncbi/MedCPT-Article-Encoder...", flush=True)
    tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
    model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(device)
    model.eval()

    all_embs: list[np.ndarray] = []
    texts = [c["text"] for c in chunks]
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        with torch.no_grad():
            enc = tok(
                batch, truncation=True, padding=True,
                return_tensors="pt", max_length=512,
            ).to(device)
            out = model(**enc)
            emb = out.last_hidden_state[:, 0, :]  # CLS token
            emb = torch.nn.functional.normalize(emb, dim=-1)
            all_embs.append(emb.cpu().float().numpy())
        if (i // batch_size) % 20 == 0:
            print(f"    {i}/{len(texts)}", flush=True)

    result = np.vstack(all_embs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, result)
    print(f"  Saved MedCPT embeddings to {cache_path}", flush=True)
    return result


def encode_medcpt_queries(
    queries: list[str],
    device: str,
    batch_size: int,
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
    model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(device)
    model.eval()

    all_embs: list[np.ndarray] = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i : i + batch_size]
        with torch.no_grad():
            enc = tok(
                batch, truncation=True, padding=True,
                return_tensors="pt", max_length=64,
            ).to(device)
            out = model(**enc)
            emb = out.last_hidden_state[:, 0, :]
            emb = torch.nn.functional.normalize(emb, dim=-1)
            all_embs.append(emb.cpu().float().numpy())

    return np.vstack(all_embs)


# ── Octen-Embedding-8B ─────────────────────────────────────────────────────────

def build_octen_corpus_embeddings(
    chunks: list[dict],
    device: str,
    batch_size: int,
    cache_path: Path,
) -> np.ndarray:
    """Encode corpus with Octen-Embedding-8B. Returns (N, D) float32."""
    if cache_path.exists():
        print(f"  Loading Octen corpus embeddings from {cache_path}", flush=True)
        return np.load(cache_path)

    from sentence_transformers import SentenceTransformer

    print("  Encoding corpus with Octen/Octen-Embedding-8B...", flush=True)
    model = SentenceTransformer("Octen/Octen-Embedding-8B", device=device)

    texts = [c["text"] for c in chunks]
    embs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    result = np.array(embs, dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, result)
    print(f"  Saved Octen embeddings to {cache_path}", flush=True)
    return result


def encode_octen_queries(
    queries: list[str],
    device: str,
    batch_size: int,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("Octen/Octen-Embedding-8B", device=device)
    embs = model.encode(
        queries,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.array(embs, dtype=np.float32)


# ── Dense retrieval ────────────────────────────────────────────────────────────

def retrieve_dense(
    corpus_embs: np.ndarray,
    chunk_ids: list[str],
    query_emb: np.ndarray,
    top_k: int,
) -> list[tuple[str, int]]:
    """Cosine similarity search (embeddings pre-normalized). Returns (chunk_id, rank)."""
    scores = corpus_embs @ query_emb  # (N,)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(chunk_ids[i], rank + 1) for rank, i in enumerate(top_idx)]


# ── RRF ────────────────────────────────────────────────────────────────────────

def rrf_merge(
    ranked_lists: dict[str, list[tuple[str, int]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion. Returns (chunk_id, rrf_score) sorted descending."""
    scores: dict[str, float] = {}
    for hits in ranked_lists.values():
        for chunk_id, rank in hits:
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Candidate record builder ───────────────────────────────────────────────────

def build_candidate_list(
    query: dict[str, Any],
    ranked_lists: dict[str, list[tuple[str, int]]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Merge retriever results into a flat candidate list with per-retriever ranks.

    Includes RRF-fused ordering. Force-appends seed chunk if not already present.
    The seed chunk is evaluated by LLM judge like any other candidate — not auto-labeled.
    """
    # Build rank lookup per retriever
    rank_lookup: dict[str, dict[str, int]] = {}
    for ret_name, hits in ranked_lists.items():
        rank_lookup[ret_name] = {cid: rank for cid, rank in hits}

    # Compute RRF to get pool order
    rrf_results = rrf_merge(ranked_lists)

    seed_id = query.get("seed_chunk_id", "")
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []

    for chunk_id, rrf_score in rrf_results:
        seen.add(chunk_id)
        candidates.append({
            "chunk_id": chunk_id,
            **{f"{r}_rank": rank_lookup[r].get(chunk_id) for r in ranked_lists},
            "rrf_score": round(float(rrf_score), 6),
            "seed": chunk_id == seed_id,
        })

    # Force-include seed chunk if missing from all retrievers
    if seed_id and seed_id not in seen:
        candidates.append({
            "chunk_id": seed_id,
            **{f"{r}_rank": None for r in ranked_lists},
            "rrf_score": 0.0,
            "seed": True,
        })

    return candidates


# ── Main ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--queries", default="data/queries.jsonl")
    p.add_argument("--corpus", default=None, help="override config corpus path")
    p.add_argument("--output", default="data/candidates.jsonl")
    p.add_argument("--config", default="config.yaml")
    p.add_argument(
        "--retrievers", default="bm25,medcpt,octen",
        help="comma-separated subset of: bm25,medcpt,octen",
    )
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=None)
    p.add_argument("--cache-dir", default=".cache")
    p.add_argument("--shard", nargs=2, type=int, metavar=("INDEX", "COUNT"),
                   help="shard INDEX (0-based) out of COUNT total shards")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    corpus_path = args.corpus or cfg["corpus"]["chunks_path"]
    top_k = args.top_k or cfg.get("retrieval", {}).get("top_k", 10)
    retrievers = [r.strip() for r in args.retrievers.split(",") if r.strip()]
    cache_dir = Path(args.cache_dir)

    import torch
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load corpus ──────────────────────────────────────────────────────────
    print(f"\nParsing corpus from {corpus_path}...")
    chunks = parse_corpus(corpus_path)
    chunk_ids = [c["chunk_id"] for c in chunks]
    chunk_id_set = set(chunk_ids)
    print(f"  {len(chunks)} chunks loaded")

    # ── Load queries ─────────────────────────────────────────────────────────
    queries: list[dict[str, Any]] = []
    with open(args.queries) as f:
        for line in f:
            queries.append(json.loads(line))
    print(f"Loaded {len(queries)} queries from {args.queries}")

    if args.shard:
        shard_idx, shard_count = args.shard
        queries = [q for i, q in enumerate(queries) if i % shard_count == shard_idx]
        print(f"Shard {shard_idx}/{shard_count}: {len(queries)} queries")

    # ── Resume ───────────────────────────────────────────────────────────────
    done_ids: set[str] = set()
    if args.resume and Path(args.output).exists():
        with open(args.output) as f:
            for line in f:
                done_ids.add(json.loads(line)["query_id"])
        print(f"Resuming: {len(done_ids)} queries already written")

    queries = [q for q in queries if q["query_id"] not in done_ids]
    if not queries:
        print("Nothing to do.")
        return

    # ── Build BM25 index ─────────────────────────────────────────────────────
    bm25_index = None
    if "bm25" in retrievers:
        print("\n── BM25 ──")
        bm25_index = build_bm25(chunks)

    # ── Build dense indexes (corpus embeddings) ───────────────────────────────
    medcpt_embs: np.ndarray | None = None
    if "medcpt" in retrievers:
        print("\n── MedCPT ──")
        medcpt_embs = build_medcpt_corpus_embeddings(
            chunks, device, args.batch_size,
            cache_dir / "medcpt_corpus.npy",
        )

    octen_embs: np.ndarray | None = None
    if "octen" in retrievers:
        print("\n── Octen-Embedding-8B ──")
        octen_embs = build_octen_corpus_embeddings(
            chunks, device, args.batch_size,
            cache_dir / "octen_corpus.npy",
        )

    # ── Encode all queries ────────────────────────────────────────────────────
    query_texts = [q["query_text"] for q in queries]

    medcpt_query_embs: np.ndarray | None = None
    if "medcpt" in retrievers:
        print(f"\nEncoding {len(query_texts)} queries with MedCPT-Query-Encoder...")
        medcpt_query_embs = encode_medcpt_queries(query_texts, device, args.batch_size)

    octen_query_embs: np.ndarray | None = None
    if "octen" in retrievers:
        print(f"\nEncoding {len(query_texts)} queries with Octen-Embedding-8B...")
        octen_query_embs = encode_octen_queries(query_texts, device, args.batch_size)

    # ── Per-query retrieval ───────────────────────────────────────────────────
    print(f"\n── Retrieval ({len(queries)} queries, top-k={top_k}) ──")
    out = open(args.output, "a" if (args.resume and Path(args.output).exists()) else "w")

    for i, query in enumerate(queries):
        ranked_lists: dict[str, list[tuple[str, int]]] = {}

        if bm25_index is not None:
            ranked_lists["bm25"] = retrieve_bm25(
                bm25_index, chunk_ids, query["query_text"], top_k
            )

        if medcpt_embs is not None and medcpt_query_embs is not None:
            ranked_lists["medcpt"] = retrieve_dense(
                medcpt_embs, chunk_ids, medcpt_query_embs[i], top_k
            )

        if octen_embs is not None and octen_query_embs is not None:
            ranked_lists["octen"] = retrieve_dense(
                octen_embs, chunk_ids, octen_query_embs[i], top_k
            )

        candidates = build_candidate_list(query, ranked_lists, top_k)

        record = {
            "query_id": query["query_id"],
            "query_text": query["query_text"],
            "seed_chunk_id": query.get("seed_chunk_id", ""),
            "retrievers_used": sorted(ranked_lists.keys()),
            "candidates": candidates,
        }
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        out.flush()

        if (i + 1) % 100 == 0 or i == 0:
            n_cands = len(candidates)
            print(f"  [{i+1}/{len(queries)}] {query['query_id']} — {n_cands} candidates")

    out.close()
    print(f"\nDone. {len(queries)} queries → {args.output}")


if __name__ == "__main__":
    main()
