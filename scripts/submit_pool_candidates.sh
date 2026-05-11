#!/usr/bin/env bash
# Submit parallel pool_candidates shard jobs on the EPFL light cluster.
#
# Phase 2a retriever set: BM25 + MedCPT + Octen-Embedding-8B
# Each shard processes 1/SHARD_COUNT of the 3,185 queries.
# Corpus embeddings are written to CACHE_DIR on the scratch PVC and shared
# across shards (whichever shard builds them first, the rest load from cache).
#
# After all jobs complete, sync and merge:
#   rsync -av light:/mnt/light/scratch/users/yiren/mamaretrieval/data/candidates_shard*.jsonl data/
#   cat data/candidates_shard{0..4}.jsonl > data/candidates.jsonl

set -euo pipefail

JOB_PREFIX="${JOB_PREFIX:-mamaretrieval-pool}"
IMAGE="${IMAGE:-registry.rcp.epfl.ch/light/yiren/mamai-guidelines:amd64-cuda-yiren-latest}"
PROJECT="${PROJECT:-light-yiren}"
SERVER="${SERVER:-light}"
SERVER_SCRATCH="${SERVER_SCRATCH:-/mnt/light/scratch/users/yiren/mamaretrieval}"
REPO_DIR="${REPO_DIR:-/lightscratch/users/yiren/mamaretrieval}"
CORPUS_PATH="${CORPUS_PATH:-/lightscratch/users/yiren/mamai-medical-guidelines/processed/chunks_for_rag.txt}"
SHARD_COUNT="${SHARD_COUNT:-5}"
RETRIEVERS="${RETRIEVERS:-bm25,medcpt,octen}"
TOP_K="${TOP_K:-10}"
QUERIES_PATH="${QUERIES_PATH:-data/queries.jsonl}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-cuda}"
CACHE_DIR="${CACHE_DIR:-$REPO_DIR/.cache}"
HF_API_KEY_FILE_AT="${HF_API_KEY_FILE_AT:-/lightscratch/users/yiren/keys/hf_key.txt}"

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_ROOT="$SERVER:$SERVER_SCRATCH"

if [[ ! -f "$LOCAL_ROOT/$QUERIES_PATH" ]]; then
  echo "ERROR: $QUERIES_PATH not found locally." >&2
  exit 1
fi

echo "Syncing repo to cluster..."
ssh "$SERVER" "mkdir -p '$SERVER_SCRATCH/scripts' '$SERVER_SCRATCH/data' '$SERVER_SCRATCH/logs'"
rsync -av --delete \
  --exclude="__pycache__/" \
  "$LOCAL_ROOT/scripts/" "$SERVER_ROOT/scripts/"
rsync -av "$LOCAL_ROOT/config.yaml" "$SERVER_ROOT/config.yaml"
ssh "$SERVER" "mkdir -p '$SERVER_SCRATCH/$(dirname "$QUERIES_PATH")'"
rsync -av "$LOCAL_ROOT/$QUERIES_PATH" "$SERVER_ROOT/$QUERIES_PATH"

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
    -e CORPUS_PATH="$CORPUS_PATH" \
    -e SHARD_INDEX="$shard" \
    -e SHARD_COUNT="$SHARD_COUNT" \
    -e RETRIEVERS="$RETRIEVERS" \
    -e TOP_K="$TOP_K" \
    -e QUERIES_PATH="$QUERIES_PATH" \
    -e BATCH_SIZE="$BATCH_SIZE" \
    -e DEVICE="$DEVICE" \
    -e CACHE_DIR="$CACHE_DIR" \
    -e HF_HOME="$REPO_DIR/hf_cache" \
    -e HF_API_KEY_FILE_AT="$HF_API_KEY_FILE_AT" \
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
