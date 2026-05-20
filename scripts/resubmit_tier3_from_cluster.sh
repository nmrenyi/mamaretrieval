#!/usr/bin/env bash
# Resubmit both Tier 3 judge shards directly from the cluster (skips ssh
# prefix). Used when the local ssh sessions can't authenticate with runai
# but the user's interactive session on the cluster can.
#
# Usage (from your working ssh session on light):
#   cd /lightscratch/users/yiren/mamaretrieval
#   bash scripts/resubmit_tier3_from_cluster.sh

set -euo pipefail

PROJECT="${PROJECT:-light-yiren}"
IMAGE="${IMAGE:-registry.rcp.epfl.ch/light/yiren/mamai-guidelines:amd64-cuda-yiren-latest}"
REPO_DIR="${REPO_DIR:-/lightscratch/users/yiren/mamaretrieval}"
CORPUS_PATH="${CORPUS_PATH:-/lightscratch/users/yiren/mamai-medical-guidelines/processed/chunks_for_rag.txt}"
HF_API_KEY_FILE_AT="${HF_API_KEY_FILE_AT:-/lightscratch/users/yiren/keys/hf_key.txt}"

JOB_PREFIX=mamaretrieval-judge-v2-top20-h100
SHARD_COUNT=2
INPUT=data/audit/candidates_v2_top20_new.jsonl
OUTPUT_DIR=data/audit
OUTPUT_PREFIX=v2_top20_new_h100

echo "Step 1: deleting any existing shards..."
for shard in 0 1; do
  runai delete job "${JOB_PREFIX}-shard${shard}" --project "$PROJECT" 2>&1 | tail -2 || true
done

echo
echo "Step 2: submitting fresh shards..."
for shard in 0 1; do
  JOB_NAME="${JOB_PREFIX}-shard${shard}"
  runai submit "$JOB_NAME" \
    --image "$IMAGE" \
    --pvc light-scratch:/lightscratch \
    --gpu 8 \
    --cpu 16 --cpu-limit 16 \
    --memory 256G --memory-limit 256G \
    --large-shm \
    --node-pool h100 \
    --project "$PROJECT" \
    --run-as-uid 296712 \
    --run-as-gid 84257 \
    -e REPO_DIR="$REPO_DIR" \
    -e CORPUS_PATH="$CORPUS_PATH" \
    -e MODEL=Qwen/Qwen3.5-397B-A17B-FP8 \
    -e TENSOR_PARALLEL=8 \
    -e MAX_MODEL_LEN=32768 \
    -e MAX_NUM_SEQS=32 \
    -e GDN_PREFILL_BACKEND=triton \
    -e WORKERS=32 \
    -e MAX_TOKENS=0 \
    -e TEMPERATURE=0.0 \
    -e JUDGE_TIMEOUT=600 \
    -e THINKING_BUDGET=10000 \
    -e THINKING_TOKEN_BUDGET=25000 \
    -e SHARD_INDEX="$shard" \
    -e SHARD_COUNT="$SHARD_COUNT" \
    -e LIMIT=0 \
    -e INPUT="$INPUT" \
    -e OUTPUT_DIR="$OUTPUT_DIR" \
    -e OUTPUT_PREFIX="$OUTPUT_PREFIX" \
    -e RUBRIC=v2_graded \
    -e RAW_OUTPUT= \
    -e VLLM_LOG_SUFFIX= \
    -e HF_HOME="$REPO_DIR/hf_cache" \
    -e PYTHONUSERBASE="$REPO_DIR/python_user_judge_shard${shard}" \
    -e RUNAI_HOME="$REPO_DIR/runai_home" \
    -e HF_API_KEY_FILE_AT="$HF_API_KEY_FILE_AT" \
    -- bash "$REPO_DIR/scripts/run_judge_relevance_job.sh"
  echo "  Submitted: $JOB_NAME"
done

echo
echo "Done. Monitor with:"
echo "  runai list jobs --project $PROJECT | grep v2-top20"
