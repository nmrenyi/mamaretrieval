#!/usr/bin/env python3
"""Embed the MAMAI guideline corpus with voyage-4-large via the REST API.

Phase-3 audit support: produces a cached, L2-normalized embedding matrix
for the full corpus that the audit retrieval step can dot-product against.

The Python SDK (voyageai 0.2.4) does not yet expose `output_dimension`, so
we POST directly to https://api.voyageai.com/v1/embeddings. Default is
2048 dims (max for voyage-4-large).

Outputs (under <cache-dir>):
  voyage4large_dim<DIM>_corpus.npy        (N, DIM) float32, L2-normalized
  voyage4large_dim<DIM>_chunk_ids.txt     N lines, matching order
  voyage4large_dim<DIM>_manifest.json     model + token spend + timestamp

Resumable: each batch is written to .../voyage4large_dim<DIM>_batches/
with a progress.json that records the last completed batch and tokens spent.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mamaretrieval.config import corpus_chunks_path, load_config
from mamaretrieval.corpus import iter_chunks

ENDPOINT = "https://api.voyageai.com/v1/embeddings"
DEFAULT_MODEL = "voyage-4-large"
DEFAULT_OUTPUT_DIM = 2048
DEFAULT_INPUT_TYPE = "document"
DEFAULT_BATCH_TOKENS = 90_000  # per-request cap is 120K — leave headroom
CHARS_PER_TOKEN = 4  # for batching estimate only; truth comes from API


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def make_batches(chunks: list[dict], batch_tokens: int) -> list[list[int]]:
    batches: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for i, c in enumerate(chunks):
        t = estimate_tokens(c["text"])
        if current and (current_tokens + t > batch_tokens or len(current) >= 1000):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(i)
        current_tokens += t
    if current:
        batches.append(current)
    return batches


def embed_batch(
    texts: list[str],
    api_key: str,
    model: str,
    input_type: str,
    output_dim: int,
    max_retries: int = 5,
) -> tuple[np.ndarray, int]:
    body = {
        "input": texts,
        "model": model,
        "input_type": input_type,
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
    p.add_argument("--corpus", default=None, help="override config corpus path")
    p.add_argument("--cache-dir", default=".cache")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output-dim", type=int, default=DEFAULT_OUTPUT_DIM)
    p.add_argument(
        "--input-type", default=DEFAULT_INPUT_TYPE, choices=["document", "query"]
    )
    p.add_argument("--batch-tokens", type=int, default=DEFAULT_BATCH_TOKENS)
    p.add_argument("--api-key-env", default="MAMAI_VOYAGE_API")
    p.add_argument("--dry-run", action="store_true", help="estimate only, no API calls")
    p.add_argument("--yes", action="store_true", help="skip the y/N prompt")
    p.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="run only the first N batches (debugging)",
    )
    args = p.parse_args()

    config = load_config("config.yaml")
    corpus_path = Path(args.corpus).expanduser() if args.corpus else corpus_chunks_path(config)
    if not corpus_path.exists():
        print(f"ERROR: corpus not found at {corpus_path}", file=sys.stderr)
        return 1

    print(f"Loading corpus from {corpus_path}...", flush=True)
    chunks = [c.to_dict() for c in iter_chunks(corpus_path)]
    print(f"  {len(chunks):,} chunks loaded", flush=True)

    batches = make_batches(chunks, args.batch_tokens)
    est_tokens = sum(estimate_tokens(c["text"]) for c in chunks)
    print()
    print(f"Estimated tokens (corpus): {est_tokens:,} (~{est_tokens/1e6:.2f}M)")
    print(f"Free tier share:           ~{est_tokens/2e8*100:.2f}% of 200M")
    print(f"Batches:                   {len(batches)} (cap {args.batch_tokens:,} tokens each)")
    print(f"Output dimension:          {args.output_dim}")
    print(f"Input type:                {args.input_type}")
    print(f"Model:                     {args.model}")

    if args.dry_run:
        print("\n(dry-run: no API calls made)")
        return 0

    if not args.yes:
        ans = input("\nProceed with embedding? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return 1

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"ERROR: env var {args.api_key_env} is not set", file=sys.stderr)
        return 1

    cache_dir = Path(args.cache_dir)
    stem = f"voyage4large_dim{args.output_dim}"
    batches_dir = cache_dir / f"{stem}_batches"
    batches_dir.mkdir(parents=True, exist_ok=True)

    progress_path = batches_dir / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        last_done = progress["last_completed_batch"]
        tokens_spent = progress["tokens_spent"]
        print(
            f"\nResuming: {last_done + 1}/{len(batches)} batches done, "
            f"{tokens_spent:,} tokens spent.",
            flush=True,
        )
    else:
        last_done = -1
        tokens_spent = 0

    limit = args.limit_batches if args.limit_batches is not None else len(batches)
    start = time.time()
    print()
    for i in range(len(batches)):
        if i <= last_done:
            continue
        if i >= limit:
            print(f"  hit --limit-batches={limit}, stopping early.", flush=True)
            break
        batch_idxs = batches[i]
        texts = [chunks[j]["text"] for j in batch_idxs]
        embs, used = embed_batch(
            texts, api_key, args.model, args.input_type, args.output_dim
        )
        if embs.shape != (len(texts), args.output_dim):
            print(
                f"ERROR batch {i}: returned shape {embs.shape}, "
                f"expected ({len(texts)}, {args.output_dim})",
                file=sys.stderr,
            )
            return 1

        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / np.maximum(norms, 1e-12)

        np.save(batches_dir / f"batch_{i:05d}.npy", embs)
        tokens_spent += used
        progress_path.write_text(
            json.dumps(
                {
                    "last_completed_batch": i,
                    "tokens_spent": tokens_spent,
                    "n_batches": len(batches),
                },
                indent=2,
            )
        )

        if i % 20 == 0 or i == limit - 1 or i == len(batches) - 1:
            elapsed = time.time() - start
            done_now = i - last_done
            rate = done_now / max(elapsed, 0.1)
            remaining = min(limit, len(batches)) - i - 1
            eta = remaining / max(rate, 0.001)
            print(
                f"  batch {i+1:>4}/{len(batches)}  "
                f"tokens={tokens_spent:>10,}  "
                f"rate={rate:>4.1f} b/s  eta={eta:>4.0f}s",
                flush=True,
            )

    # Final concat — only if we ran to the end
    if args.limit_batches is not None and args.limit_batches < len(batches):
        elapsed = time.time() - start
        print(
            f"\nStopped after {limit} batches in {elapsed:.1f}s. "
            f"Resume by re-running without --limit-batches.",
            flush=True,
        )
        return 0

    print("\nConcatenating batches...", flush=True)
    parts = [np.load(batches_dir / f"batch_{i:05d}.npy") for i in range(len(batches))]
    corpus_embs = np.concatenate(parts, axis=0)
    expected_shape = (len(chunks), args.output_dim)
    if corpus_embs.shape != expected_shape:
        print(
            f"ERROR: final shape {corpus_embs.shape} != expected {expected_shape}",
            file=sys.stderr,
        )
        return 1

    out_npy = cache_dir / f"{stem}_corpus.npy"
    out_ids = cache_dir / f"{stem}_chunk_ids.txt"
    out_manifest = cache_dir / f"{stem}_manifest.json"
    np.save(out_npy, corpus_embs)
    out_ids.write_text("\n".join(c["chunk_id"] for c in chunks) + "\n")
    out_manifest.write_text(
        json.dumps(
            {
                "model": args.model,
                "output_dimension": args.output_dim,
                "input_type": args.input_type,
                "n_chunks": len(chunks),
                "total_tokens": tokens_spent,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "endpoint": ENDPOINT,
                "corpus_path": str(corpus_path),
            },
            indent=2,
        )
    )
    shutil.rmtree(batches_dir)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s.", flush=True)
    print(f"  Embeddings:   {out_npy}  shape={corpus_embs.shape}")
    print(f"  Chunk IDs:    {out_ids}")
    print(f"  Manifest:     {out_manifest}")
    print(
        f"  Total tokens: {tokens_spent:,} "
        f"(~{tokens_spent/1e6:.2f}M, ~{tokens_spent/2e8*100:.2f}% of 200M free tier)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
