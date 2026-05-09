#!/usr/bin/env bash
# Submit parallel relevance judge shard jobs on the EPFL light cluster.
#
# Phase 2b: runs judge_relevance.py on all 78,571 (query, chunk) pairs.
# Each shard processes 1/SHARD_COUNT of the 3,185 queries with all their candidates.
#
# After all jobs complete, sync and merge:
#   rsync -av light:/mnt/light/scratch/users/yiren/mamaretrieval/data/relevance_labels_shard*.jsonl data/
#   cat data/relevance_labels_shard{0..4}.jsonl > data/relevance_labels.jsonl

set -euo pipefail

JOB_PREFIX="${JOB_PREFIX:-mamaretrieval-judge}"
IMAGE="${IMAGE:-registry.rcp.epfl.ch/light/yiren/mamai-guidelines:amd64-cuda-yiren-latest}"
PROJECT="${PROJECT:-light-yiren}"
SERVER="${SERVER:-light}"
SERVER_SCRATCH="${SERVER_SCRATCH:-/mnt/light/scratch/users/yiren/mamaretrieval}"
REPO_DIR="${REPO_DIR:-/lightscratch/users/yiren/mamaretrieval}"
CORPUS_PATH="${CORPUS_PATH:-/lightscratch/users/yiren/mamai-medical-guidelines/processed/chunks_for_rag.txt}"
MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}"
WORKERS="${WORKERS:-8}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SHARD_COUNT="${SHARD_COUNT:-5}"
HF_API_KEY_FILE_AT="${HF_API_KEY_FILE_AT:-/lightscratch/users/yiren/keys/hf_key.txt}"

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_ROOT="$SERVER:$SERVER_SCRATCH"

if [[ ! -f "$LOCAL_ROOT/data/candidates.jsonl" ]]; then
  echo "ERROR: data/candidates.jsonl not found. Run Phase 2a first." >&2
  exit 1
fi

echo "Syncing repo to cluster..."
ssh "$SERVER" "mkdir -p '$SERVER_SCRATCH/scripts' '$SERVER_SCRATCH/data' '$SERVER_SCRATCH/logs'"
rsync -av --delete \
  --exclude="__pycache__/" \
  "$LOCAL_ROOT/scripts/" "$SERVER_ROOT/scripts/"
rsync -av "$LOCAL_ROOT/config.yaml"           "$SERVER_ROOT/config.yaml"
rsync -av "$LOCAL_ROOT/data/candidates.jsonl" "$SERVER_ROOT/data/candidates.jsonl"

echo "Submitting $SHARD_COUNT shard jobs..."
for shard in $(seq 0 $((SHARD_COUNT - 1))); do
  JOB_NAME="${JOB_PREFIX}-shard${shard}"
  ssh "$SERVER" "runai delete job '$JOB_NAME' --project '$PROJECT' >/dev/null 2>&1 || true"
  ssh "$SERVER" runai submit "$JOB_NAME" \
    --image "$IMAGE" \
    --pvc light-scratch:/lightscratch \
    --gpu "$TENSOR_PARALLEL" \
    --cpu 16 --cpu-limit 16 \
    --memory 256G --memory-limit 256G \
    --large-shm \
    --node-pool h100 \
    --project "$PROJECT" \
    --run-as-uid 296712 \
    --run-as-gid 84257 \
    -e REPO_DIR="$REPO_DIR" \
    -e CORPUS_PATH="$CORPUS_PATH" \
    -e MODEL="$MODEL" \
    -e TENSOR_PARALLEL="$TENSOR_PARALLEL" \
    -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    -e MAX_NUM_SEQS="$MAX_NUM_SEQS" \
    -e GDN_PREFILL_BACKEND="$GDN_PREFILL_BACKEND" \
    -e WORKERS="$WORKERS" \
    -e MAX_TOKENS="$MAX_TOKENS" \
    -e TEMPERATURE="$TEMPERATURE" \
    -e SHARD_INDEX="$shard" \
    -e SHARD_COUNT="$SHARD_COUNT" \
    -e HF_HOME="$REPO_DIR/hf_cache" \
    -e PYTHONUSERBASE="$REPO_DIR/python_user" \
    -e RUNAI_HOME="$REPO_DIR/runai_home" \
    -e HF_API_KEY_FILE_AT="$HF_API_KEY_FILE_AT" \
    -- bash "$REPO_DIR/scripts/run_judge_relevance_job.sh"
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
echo "  rsync -av $SERVER_ROOT/data/relevance_labels_shard*.jsonl data/"
echo "  cat data/relevance_labels_shard{0..$((SHARD_COUNT-1))}.jsonl > data/relevance_labels.jsonl"
