#!/usr/bin/env bash
# Submit the LateOn (ColBERT) audit retrieval job on the EPFL light cluster.
#
# Single shard — encodes the 63k-chunk corpus into per-token ColBERT
# embeddings, builds a PLAID index, retrieves top-20 for the 100 audit
# queries, and writes data/audit/lateon_top20.jsonl on the cluster
# scratch volume.

set -euo pipefail

JOB_NAME="${JOB_NAME:-mamaretrieval-lateon-audit}"
IMAGE="${IMAGE:-registry.rcp.epfl.ch/light/yiren/mamai-guidelines:amd64-cuda-yiren-latest}"
PROJECT="${PROJECT:-light-yiren}"
SERVER="${SERVER:-light}"
SERVER_SCRATCH="${SERVER_SCRATCH:-/mnt/light/scratch/users/yiren/mamaretrieval}"
REPO_DIR="${REPO_DIR:-/lightscratch/users/yiren/mamaretrieval}"
CORPUS_PATH="${CORPUS_PATH:-/lightscratch/users/yiren/mamai-medical-guidelines/processed/chunks_for_rag.txt}"
QUERIES_PATH="${QUERIES_PATH:-data/queries.jsonl}"
QUERY_IDS_PATH="${QUERY_IDS_PATH:-data/audit/query_ids.txt}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TOP_K="${TOP_K:-20}"
NODE_POOL="${NODE_POOL:-h100}"
PYTHONUSERBASE_PATH="${PYTHONUSERBASE_PATH:-$REPO_DIR/python_user_lateon}"
HF_API_KEY_FILE_AT="${HF_API_KEY_FILE_AT:-/lightscratch/users/yiren/keys/hf_key.txt}"

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_ROOT="$SERVER:$SERVER_SCRATCH"

for f in "$QUERIES_PATH" "$QUERY_IDS_PATH"; do
  if [[ ! -f "$LOCAL_ROOT/$f" ]]; then
    echo "ERROR: $f not found locally." >&2
    exit 1
  fi
done

echo "Syncing repo to cluster..."
ssh "$SERVER" "mkdir -p '$SERVER_SCRATCH/scripts' '$SERVER_SCRATCH/data/audit' '$SERVER_SCRATCH/logs'"
rsync -av --delete \
  --exclude="__pycache__/" \
  "$LOCAL_ROOT/scripts/" "$SERVER_ROOT/scripts/"
rsync -av "$LOCAL_ROOT/config.yaml" "$SERVER_ROOT/config.yaml"
rsync -av "$LOCAL_ROOT/mamaretrieval/" "$SERVER_ROOT/mamaretrieval/"
rsync -av "$LOCAL_ROOT/$QUERIES_PATH" "$SERVER_ROOT/$QUERIES_PATH"
rsync -av "$LOCAL_ROOT/$QUERY_IDS_PATH" "$SERVER_ROOT/$QUERY_IDS_PATH"

echo "Submitting $JOB_NAME..."
ssh "$SERVER" "runai delete job '$JOB_NAME' --project '$PROJECT' >/dev/null 2>&1 || true"
ssh "$SERVER" runai submit "$JOB_NAME" \
  --image "$IMAGE" \
  --pvc light-scratch:/lightscratch \
  --gpu 1 \
  --cpu 8 --cpu-limit 8 \
  --memory 96G --memory-limit 96G \
  --large-shm \
  --node-pool "$NODE_POOL" \
  --project "$PROJECT" \
  --run-as-uid 296712 \
  --run-as-gid 84257 \
  -e REPO_DIR="$REPO_DIR" \
  -e CORPUS_PATH="$CORPUS_PATH" \
  -e QUERIES_PATH="$QUERIES_PATH" \
  -e QUERY_IDS_PATH="$QUERY_IDS_PATH" \
  -e DEVICE="$DEVICE" \
  -e BATCH_SIZE="$BATCH_SIZE" \
  -e TOP_K="$TOP_K" \
  -e HF_HOME="$REPO_DIR/hf_cache" \
  -e HF_API_KEY_FILE_AT="$HF_API_KEY_FILE_AT" \
  -e PYTHONUSERBASE="$PYTHONUSERBASE_PATH" \
  -e RUNAI_HOME="$REPO_DIR/runai_home" \
  -- bash "$REPO_DIR/scripts/run_lateon_audit_job.sh"

echo
echo "Monitor:"
echo "  ssh $SERVER 'runai list jobs --project $PROJECT'"
echo "  ssh $SERVER 'runai logs $JOB_NAME -f --project $PROJECT'"
echo
echo "Sync result after completion:"
echo "  rsync -av $SERVER_ROOT/data/audit/lateon_top20.jsonl data/audit/"
