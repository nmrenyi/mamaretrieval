#!/usr/bin/env bash
# In-pod runner for the LateOn (ColBERT) audit retrieval job.
#
# Installs pylate into a per-user prefix, then runs retrieve_lateon_audit.py
# against the 100 audit queries with CUDA.
#
# Required env vars: (none — runs as a single shard)
# Optional env vars (with defaults):
#   REPO_DIR, CORPUS_PATH, QUERIES_PATH, QUERY_IDS_PATH, DEVICE, BATCH_SIZE,
#   TOP_K, MODEL, INDEX_FOLDER, OUTPUT_PATH

set -euo pipefail

REPO_DIR="${REPO_DIR:-/lightscratch/users/yiren/mamaretrieval}"
CORPUS_PATH="${CORPUS_PATH:-/lightscratch/users/yiren/mamai-medical-guidelines/processed/chunks_for_rag.txt}"
QUERIES_PATH="${QUERIES_PATH:-data/queries.jsonl}"
QUERY_IDS_PATH="${QUERY_IDS_PATH:-data/audit/query_ids.txt}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TOP_K="${TOP_K:-20}"
MODEL="${MODEL:-lightonai/GTE-ModernColBERT-v1}"
INDEX_FOLDER="${INDEX_FOLDER:-$REPO_DIR/.cache/lateon_plaid_index}"
INDEX_NAME="${INDEX_NAME:-mamaretrieval_corpus}"
OUTPUT_PATH="${OUTPUT_PATH:-data/audit/lateon_top20.jsonl}"

export HOME="${RUNAI_HOME:-$REPO_DIR/runai_home}"
export HF_HOME="${HF_HOME:-$REPO_DIR/hf_cache}"
export PYTHONUSERBASE="${PYTHONUSERBASE:-$REPO_DIR/python_user}"
export PATH="$PYTHONUSERBASE/bin:$HOME/.local/bin:$PATH"

HF_API_KEY_FILE_AT="${HF_API_KEY_FILE_AT:-/lightscratch/users/yiren/keys/hf_key.txt}"
if [[ -f "$HF_API_KEY_FILE_AT" ]]; then
  export HF_TOKEN="$(cat "$HF_API_KEY_FILE_AT")"
  echo "HF token loaded from $HF_API_KEY_FILE_AT"
else
  echo "WARNING: HF token file not found at $HF_API_KEY_FILE_AT — model downloads may fail" >&2
fi

cd "$REPO_DIR"
mkdir -p logs data data/audit "$HOME" "$HF_HOME" "$PYTHONUSERBASE" "$(dirname "$INDEX_FOLDER")"

echo "Installing pylate + torchvision + transformers into $PYTHONUSERBASE..."
echo "(pip resolves a coherent torch/torchvision/transformers set. transformers is force-upgraded so its modernbert module guards the flash_attn import — the image's older transformers imports flash_attn unconditionally, and flash_attn's C extension breaks under the new torch ABI.)"
python3 -m pip install --user --upgrade pylate torchvision transformers pyyaml
echo "pylate install done."

echo "Starting LateOn audit retrieval..."
python3 -u scripts/retrieve_lateon_audit.py \
  --corpus "$CORPUS_PATH" \
  --queries "$QUERIES_PATH" \
  --query-ids "$QUERY_IDS_PATH" \
  --model "$MODEL" \
  --index-folder "$INDEX_FOLDER" \
  --index-name "$INDEX_NAME" \
  --top-k "$TOP_K" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --output "$OUTPUT_PATH"

echo "LateOn audit retrieval complete: $OUTPUT_PATH"
