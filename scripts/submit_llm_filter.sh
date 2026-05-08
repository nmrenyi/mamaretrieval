#!/usr/bin/env bash
# Submit 5 parallel LLM filter jobs, one shard per GPU.
#
# After all jobs complete, merge outputs:
#   cat data/llm_filtered_chunks_shard{0..4}.jsonl > data/llm_filtered_chunks.jsonl
#   cat data/llm_filter_results_shard{0..4}.jsonl  > data/llm_filter_results.jsonl

set -euo pipefail

JOB_PREFIX="${JOB_PREFIX:-mamaretrieval-filter}"
IMAGE="${IMAGE:-registry.rcp.epfl.ch/light/yiren/mamai-guidelines:amd64-cuda-yiren-latest}"
PROJECT="${PROJECT:-light-yiren}"
SERVER="${SERVER:-light}"
SERVER_SCRATCH="${SERVER_SCRATCH:-/mnt/light/scratch/users/yiren/mamaretrieval}"
REPO_DIR="${REPO_DIR:-/lightscratch/users/yiren/mamaretrieval}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B-FP8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}"
WORKERS="${WORKERS:-8}"
NUM_PREDICT="${NUM_PREDICT:-8192}"
SHARD_COUNT="${SHARD_COUNT:-5}"
FILTER_INPUT="${FILTER_INPUT:-data/sampled_chunks.jsonl}"

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_ROOT="$SERVER:$SERVER_SCRATCH"

if [[ ! -f "$LOCAL_ROOT/$FILTER_INPUT" ]]; then
  echo "ERROR: $FILTER_INPUT not found. Run scripts/sample_chunks.py first." >&2
  exit 1
fi

echo "Syncing scripts and filter input to cluster..."
ssh "$SERVER" "mkdir -p '$SERVER_SCRATCH/scripts' '$SERVER_SCRATCH/data' '$SERVER_SCRATCH/logs'"
rsync -av --delete \
  --exclude="__pycache__/" \
  "$LOCAL_ROOT/scripts/" "$SERVER_ROOT/scripts/"
rsync -av --delete \
  --exclude="__pycache__/" \
  "$LOCAL_ROOT/mamaretrieval/" "$SERVER_ROOT/mamaretrieval/"
rsync -av "$LOCAL_ROOT/$FILTER_INPUT" "$SERVER_ROOT/$FILTER_INPUT"

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
    -e MODEL="$MODEL" \
    -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    -e MAX_NUM_SEQS="$MAX_NUM_SEQS" \
    -e GDN_PREFILL_BACKEND="$GDN_PREFILL_BACKEND" \
    -e WORKERS="$WORKERS" \
    -e NUM_PREDICT="$NUM_PREDICT" \
    -e FILTER_INPUT="$FILTER_INPUT" \
    -e SHARD_INDEX="$shard" \
    -e SHARD_COUNT="$SHARD_COUNT" \
    -e HF_HOME="$REPO_DIR/hf_cache" \
    -e PYTHONUSERBASE="$REPO_DIR/python_user" \
    -e RUNAI_HOME="$REPO_DIR/runai_home" \
    -- bash "$REPO_DIR/scripts/run_llm_filter_job.sh"
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
echo "  rsync -av $SERVER_ROOT/data/llm_filter*_shard*.jsonl data/"
echo "  cat data/llm_filtered_chunks_shard{0..$((SHARD_COUNT-1))}.jsonl > data/llm_filtered_chunks.jsonl"
echo "  cat data/llm_filter_results_shard{0..$((SHARD_COUNT-1))}.jsonl  > data/llm_filter_results.jsonl"
