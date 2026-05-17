#!/usr/bin/env bash
# In-pod runner for the Gecko retrieval job (full query set).
#
# Installs sentencepiece + ai_edge_litert into a per-user prefix, then runs
# retrieve_gecko_audit.py against the full 3,185 queries with the on-device
# TFLite model.
#
# Required env vars: (none — single shard)
# Optional env vars (with defaults below):
#   REPO_DIR, MODEL_PATH, TOKENIZER_PATH, SQLITE_PATH, CHUNKS_PATH,
#   QUERIES_PATH, QUERY_IDS_PATH, OUTPUT_PATH, QUERIES_NPY, TOP_K

set -euo pipefail

REPO_DIR="${REPO_DIR:-/lightscratch/users/yiren/mamaretrieval}"
MODEL_PATH="${MODEL_PATH:-/lightscratch/users/yiren/model_backup/Gecko_1024_quant.tflite}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/lightscratch/users/yiren/model_backup/sentencepiece.model}"
SQLITE_PATH="${SQLITE_PATH:-/lightscratch/users/yiren/model_backup/embeddings.sqlite}"
CHUNKS_PATH="${CHUNKS_PATH:-/lightscratch/users/yiren/mamai-medical-guidelines/processed/chunks_for_rag.txt}"
QUERIES_PATH="${QUERIES_PATH:-data/queries.jsonl}"
QUERY_IDS_PATH="${QUERY_IDS_PATH:-data/all_query_ids.txt}"
OUTPUT_PATH="${OUTPUT_PATH:-data/full/gecko_top20.jsonl}"
QUERIES_NPY="${QUERIES_NPY:-data/full/gecko_queries.npy}"
TOP_K="${TOP_K:-20}"

export HOME="${RUNAI_HOME:-$REPO_DIR/runai_home}"
export HF_HOME="${HF_HOME:-$REPO_DIR/hf_cache}"
export PYTHONUSERBASE="${PYTHONUSERBASE:-$REPO_DIR/python_user_gecko}"
export PATH="$PYTHONUSERBASE/bin:$HOME/.local/bin:$PATH"

cd "$REPO_DIR"
mkdir -p logs data data/full "$HOME" "$HF_HOME" "$PYTHONUSERBASE" "$(dirname "$OUTPUT_PATH")"

echo "Installing sentencepiece + ai_edge_litert + numpy into $PYTHONUSERBASE..."
python3 -m pip install --user --upgrade sentencepiece ai-edge-litert numpy
echo "deps installed."

echo "Sanity-checking input paths..."
for f in "$MODEL_PATH" "$TOKENIZER_PATH" "$SQLITE_PATH" "$CHUNKS_PATH" "$QUERIES_PATH" "$QUERY_IDS_PATH"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: required input not found: $f" >&2
    exit 1
  fi
done
echo "all inputs present."

echo "Starting Gecko retrieval (full query set)..."
python3 -u scripts/retrieve_gecko_audit.py \
  --queries "$QUERIES_PATH" \
  --query-ids "$QUERY_IDS_PATH" \
  --model "$MODEL_PATH" \
  --tokenizer "$TOKENIZER_PATH" \
  --sqlite "$SQLITE_PATH" \
  --chunks-txt "$CHUNKS_PATH" \
  --output "$OUTPUT_PATH" \
  --queries-npy "$QUERIES_NPY" \
  --top-k "$TOP_K"

echo "Gecko full retrieval complete: $OUTPUT_PATH"
