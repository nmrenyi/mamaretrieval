#!/usr/bin/env bash
# Entrypoint for a single pool_candidates shard job on the cluster.
# Environment variables set by the job submitter:
#   REPO_DIR, SHARD_INDEX, SHARD_COUNT, RETRIEVERS, TOP_K, BATCH_SIZE, DEVICE

set -euo pipefail

REPO_DIR="${REPO_DIR:-/lightscratch/users/yiren/mamaretrieval}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
RETRIEVERS="${RETRIEVERS:-bm25,medcpt,octen}"
TOP_K="${TOP_K:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-cuda}"
CACHE_DIR="${CACHE_DIR:-$REPO_DIR/.cache}"

cd "$REPO_DIR"

PYTHONPATH="$REPO_DIR" python -u scripts/pool_candidates.py \
  --queries data/queries.jsonl \
  --output "data/candidates_shard${SHARD_INDEX}.jsonl" \
  --retrievers "$RETRIEVERS" \
  --top-k "$TOP_K" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --cache-dir "$CACHE_DIR" \
  --shard "$SHARD_INDEX" "$SHARD_COUNT" \
  --resume
