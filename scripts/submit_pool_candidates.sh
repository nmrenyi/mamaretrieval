#!/usr/bin/env bash
# Submit parallel pool_candidates shard jobs on the EPFL light cluster.
#
# Phase 2a retriever set: BM25 + MedCPT + Octen-Embedding-8B
# Each shard processes 1/SHARD_COUNT of the 3,185 queries.
# Corpus embeddings are cached in CACHE_DIR — all shards share the same cache.
#
# After all jobs complete, merge outputs:
#   cat data/candidates_shard{0..4}.jsonl > data/candidates.jsonl

set -euo pipefail

JOB_PREFIX="${JOB_PREFIX:-mamaretrieval-pool}"
IMAGE="${IMAGE:-registry.rcp.epfl.ch/light/yiren/mamai-guidelines:amd64-cuda-yiren-latest}"
PROJECT="${PROJECT:-light-yiren}"
SERVER="${SERVER:-light}"
SERVER_SCRATCH="${SERVER_SCRATCH:-/mnt/light/scratch/users/yiren/mamaretrieval}"
REPO_DIR="${REPO_DIR:-/lightscratch/users/yiren/mamaretrieval}"
SHARD_COUNT="${SHARD_COUNT:-5}"
RETRIEVERS="${RETRIEVERS:-bm25,medcpt,octen}"
TOP_K="${TOP_K:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-cuda}"
CACHE_DIR="${CACHE_DIR:-$REPO_DIR/.cache}"

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_ROOT="$SERVER:$SERVER_SCRATCH"

if [[ ! -f "$LOCAL_ROOT/data/queries.jsonl" ]]; then
  echo "ERROR: data/queries.jsonl not found. Run scripts/generate_queries.py first." >&2
  exit 1
fi

echo "Syncing scripts and queries to cluster..."
ssh "$SERVER" "mkdir -p '$SERVER_SCRATCH/scripts' '$SERVER_SCRATCH/data' '$SERVER_SCRATCH/logs'"
rsync -av --delete \
  --exclude="__pycache__/" \
  "$LOCAL_ROOT/scripts/" "$SERVER_ROOT/scripts/"
rsync -av --delete \
  --exclude="__pycache__/" \
  "$LOCAL_ROOT/mamaretrieval/" "$SERVER_ROOT/mamaretrieval/" 2>/dev/null || true
rsync -av "$LOCAL_ROOT/data/queries.jsonl" "$SERVER_ROOT/data/queries.jsonl"

echo "Submitting $SHARD_COUNT shard jobs..."
for shard in $(seq 0 $((SHARD_COUNT - 1))); do
  JOB_NAME="${JOB_PREFIX}-shard${shard}"
  ssh "$SERVER" "runai delete job '$JOB_NAME' --project '$PROJECT' >/dev/null 2>&1 || true"
  ssh "$SERVER" runai submit "$JOB_NAME" \
    --image "$IMAGE" \
    --pvc light-scratch:/lightscratch \
    --gpu 1 \
    --cpu 8 --cpu-limit 8 \
    --memory 96G --memory-limit 96G \
    --large-shm \
    --node-pool h100 \
    --project "$PROJECT" \
    --run-as-uid 296712 \
    --run-as-gid 84257 \
    -e REPO_DIR="$REPO_DIR" \
    -e SHARD_INDEX="$shard" \
    -e SHARD_COUNT="$SHARD_COUNT" \
    -e RETRIEVERS="$RETRIEVERS" \
    -e TOP_K="$TOP_K" \
    -e BATCH_SIZE="$BATCH_SIZE" \
    -e DEVICE="$DEVICE" \
    -e CACHE_DIR="$CACHE_DIR" \
    -e HF_HOME="$REPO_DIR/hf_cache" \
    -e PYTHONUSERBASE="$REPO_DIR/python_user" \
    -e RUNAI_HOME="$REPO_DIR/runai_home" \
    -- bash "$REPO_DIR/scripts/run_pool_candidates_job.sh"
  echo "  Submitted: $JOB_NAME"
done

echo
echo "Monitor jobs:"
echo "  ssh $SERVER 'runai list jobs --project $PROJECT'"
for shard in $(seq 0 $((SHARD_COUNT - 1))); do
  echo "  ssh $SERVER 'runai logs ${JOB_PREFIX}-shard${shard} -f --project $PROJECT'"
done
echo
echo "After all jobs complete, sync and merge:"
echo "  rsync -av $SERVER_ROOT/data/candidates_shard*.jsonl data/"
echo "  cat data/candidates_shard{0..$((SHARD_COUNT-1))}.jsonl > data/candidates.jsonl"
