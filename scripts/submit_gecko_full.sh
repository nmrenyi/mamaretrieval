#!/usr/bin/env bash
# Submit the Gecko full-query retrieval job on the EPFL light cluster.
#
# Single shard — embeds 3,185 queries with the on-device Gecko TFLite model,
# scores against the pre-computed sqlite chunk embeddings, writes top-20.
# Uses CPU (TFLite quantized is CPU-friendly; no GPU needed but we request 1
# to keep the pod responsive against other jobs).

set -euo pipefail

JOB_NAME="${JOB_NAME:-mamaretrieval-gecko-full}"
IMAGE="${IMAGE:-registry.rcp.epfl.ch/light/yiren/mamai-guidelines:amd64-cuda-yiren-latest}"
PROJECT="${PROJECT:-light-yiren}"
SERVER="${SERVER:-light}"
SERVER_SCRATCH="${SERVER_SCRATCH:-/mnt/light/scratch/users/yiren/mamaretrieval}"
REPO_DIR="${REPO_DIR:-/lightscratch/users/yiren/mamaretrieval}"
QUERIES_PATH="${QUERIES_PATH:-data/queries.jsonl}"
QUERY_IDS_PATH="${QUERY_IDS_PATH:-data/all_query_ids.txt}"
OUTPUT_PATH="${OUTPUT_PATH:-data/full/gecko_top20.jsonl}"
QUERIES_NPY="${QUERIES_NPY:-data/full/gecko_queries.npy}"
MODEL_PATH="${MODEL_PATH:-/lightscratch/users/yiren/model_backup/Gecko_1024_quant.tflite}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/lightscratch/users/yiren/model_backup/sentencepiece.model}"
SQLITE_PATH="${SQLITE_PATH:-/lightscratch/users/yiren/model_backup/embeddings.sqlite}"
CHUNKS_PATH="${CHUNKS_PATH:-/lightscratch/users/yiren/mamai-medical-guidelines/processed/chunks_for_rag.txt}"
TOP_K="${TOP_K:-20}"
NODE_POOL="${NODE_POOL:-h100}"
PYTHONUSERBASE_PATH="${PYTHONUSERBASE_PATH:-$REPO_DIR/python_user_gecko}"

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_ROOT="$SERVER:$SERVER_SCRATCH"

for f in "$QUERIES_PATH" "$QUERY_IDS_PATH"; do
  if [[ ! -f "$LOCAL_ROOT/$f" ]]; then
    echo "ERROR: $f not found locally." >&2
    exit 1
  fi
done

echo "Syncing repo to cluster..."
ssh "$SERVER" "mkdir -p '$SERVER_SCRATCH/scripts' '$SERVER_SCRATCH/data/full' '$SERVER_SCRATCH/logs'"
rsync -av --delete \
  --exclude="__pycache__/" \
  "$LOCAL_ROOT/scripts/" "$SERVER_ROOT/scripts/"
rsync -av "$LOCAL_ROOT/config.yaml" "$SERVER_ROOT/config.yaml"
rsync -av "$LOCAL_ROOT/$QUERIES_PATH" "$SERVER_ROOT/$QUERIES_PATH"
rsync -av "$LOCAL_ROOT/$QUERY_IDS_PATH" "$SERVER_ROOT/$QUERY_IDS_PATH"

echo "Submitting $JOB_NAME..."
ssh "$SERVER" "runai delete job '$JOB_NAME' --project '$PROJECT' >/dev/null 2>&1 || true"
ssh "$SERVER" runai submit "$JOB_NAME" \
  --image "$IMAGE" \
  --pvc light-scratch:/lightscratch \
  --gpu 1 \
  --cpu 8 --cpu-limit 8 \
  --memory 64G --memory-limit 64G \
  --large-shm \
  --node-pool "$NODE_POOL" \
  --project "$PROJECT" \
  --run-as-uid 296712 \
  --run-as-gid 84257 \
  -e REPO_DIR="$REPO_DIR" \
  -e MODEL_PATH="$MODEL_PATH" \
  -e TOKENIZER_PATH="$TOKENIZER_PATH" \
  -e SQLITE_PATH="$SQLITE_PATH" \
  -e CHUNKS_PATH="$CHUNKS_PATH" \
  -e QUERIES_PATH="$QUERIES_PATH" \
  -e QUERY_IDS_PATH="$QUERY_IDS_PATH" \
  -e OUTPUT_PATH="$OUTPUT_PATH" \
  -e QUERIES_NPY="$QUERIES_NPY" \
  -e TOP_K="$TOP_K" \
  -e PYTHONUSERBASE="$PYTHONUSERBASE_PATH" \
  -e RUNAI_HOME="$REPO_DIR/runai_home" \
  -- bash "$REPO_DIR/scripts/run_gecko_full_job.sh"

echo
echo "Monitor:"
echo "  ssh $SERVER 'runai list jobs --project $PROJECT'"
echo "  ssh $SERVER 'runai logs $JOB_NAME -f --project $PROJECT'"
echo
echo "Sync result after completion:"
echo "  rsync -av $SERVER_ROOT/$OUTPUT_PATH data/full/"
