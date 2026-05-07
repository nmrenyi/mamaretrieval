#!/usr/bin/env bash
# Submit a Qwen3.6-27B-FP8 vLLM thinking-capture job for the first 20 sampled chunks.

set -euo pipefail

JOB_NAME="${JOB_NAME:-mamaretrieval-qwen36-27b-fp8-top20}"
IMAGE="${IMAGE:-registry.rcp.epfl.ch/light/yiren/mamai-guidelines:amd64-cuda-yiren-latest}"
PROJECT="${PROJECT:-light-yiren}"
SERVER="${SERVER:-light}"
SERVER_SCRATCH="${SERVER_SCRATCH:-/mnt/light/scratch/users/yiren/mamaretrieval}"
REPO_DIR="${REPO_DIR:-/lightscratch/users/yiren/mamaretrieval}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B-FP8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}"

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_ROOT="$SERVER:$SERVER_SCRATCH"

if [[ ! -f "$LOCAL_ROOT/data/sampled_chunks.jsonl" ]]; then
  echo "ERROR: data/sampled_chunks.jsonl not found. Run scripts/sample_chunks.py first." >&2
  exit 1
fi

echo "Preparing cluster workspace..."
ssh "$SERVER" "mkdir -p '$SERVER_SCRATCH/scripts' '$SERVER_SCRATCH/data' '$SERVER_SCRATCH/logs'"

echo "Syncing scripts and sampled chunks..."
rsync -av --delete \
  --exclude="__pycache__/" \
  "$LOCAL_ROOT/scripts/" "$SERVER_ROOT/scripts/"
rsync -av "$LOCAL_ROOT/data/sampled_chunks.jsonl" "$SERVER_ROOT/data/sampled_chunks.jsonl"

ssh "$SERVER" "runai delete job '$JOB_NAME' --project '$PROJECT' >/dev/null 2>&1 || true"

echo "Submitting job: $JOB_NAME"
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
  -e MAX_TOKENS="$MAX_TOKENS" \
  -e GDN_PREFILL_BACKEND="$GDN_PREFILL_BACKEND" \
  -e HF_HOME="$REPO_DIR/hf_cache" \
  -e PYTHONUSERBASE="$REPO_DIR/python_user" \
  -e RUNAI_HOME="$REPO_DIR/runai_home" \
  -- bash "$REPO_DIR/scripts/run_qwen36_capture_top20_job.sh"

echo
echo "Job submitted. Monitor with:"
echo "  ssh $SERVER 'runai logs $JOB_NAME -f --project $PROJECT'"
echo
echo "Copy results after completion with:"
echo "  rsync -av $SERVER_ROOT/data/qwen36_27b_fp8_top20_thinking_*/ data/"
