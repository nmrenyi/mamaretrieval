#!/usr/bin/env bash
# Entrypoint for a single pool_candidates shard job on the cluster.
# Required env vars:
#   SHARD_INDEX, SHARD_COUNT
#
# Optional env vars (all have defaults):
#   REPO_DIR, CORPUS_PATH, RETRIEVERS, TOP_K, BATCH_SIZE, DEVICE, CACHE_DIR

set -euo pipefail

REPO_DIR="${REPO_DIR:-/lightscratch/users/yiren/mamaretrieval}"
CORPUS_PATH="${CORPUS_PATH:-/lightscratch/users/yiren/mamai-medical-guidelines/processed/chunks_for_rag.txt}"
RETRIEVERS="${RETRIEVERS:-bm25,medcpt,octen}"
TOP_K="${TOP_K:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-cuda}"
CACHE_DIR="${CACHE_DIR:-$REPO_DIR/.cache}"

: "${SHARD_INDEX:?ERROR: SHARD_INDEX must be set}"
: "${SHARD_COUNT:?ERROR: SHARD_COUNT must be set}"

export HOME="${RUNAI_HOME:-$REPO_DIR/runai_home}"
export HF_HOME="${HF_HOME:-$REPO_DIR/hf_cache}"
export PYTHONUSERBASE="${PYTHONUSERBASE:-$REPO_DIR/python_user}"
export PATH="$PYTHONUSERBASE/bin:$HOME/.local/bin:$PATH"

cd "$REPO_DIR"
mkdir -p logs data "$HOME" "$HF_HOME" "$PYTHONUSERBASE" "$CACHE_DIR"

# Install missing packages
python3 - <<'PY'
import importlib.util
import subprocess
import sys

needed = []
for pkg, import_name in [
    ("rank-bm25",            "rank_bm25"),
    ("sentence-transformers","sentence_transformers"),
    ("transformers",         "transformers"),
    ("tqdm",                 "tqdm"),
    ("pyyaml",               "yaml"),
    ("numpy",                "numpy"),
]:
    if importlib.util.find_spec(import_name) is None:
        needed.append(pkg)

if needed:
    print(f"Installing: {needed}", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", *needed])
else:
    print("All packages present.", flush=True)
PY

echo "Starting pool_candidates: shard ${SHARD_INDEX}/${SHARD_COUNT}"
python3 -u scripts/pool_candidates.py \
  --queries data/queries.jsonl \
  --corpus "$CORPUS_PATH" \
  --output "data/candidates_shard${SHARD_INDEX}.jsonl" \
  --config config.yaml \
  --retrievers "$RETRIEVERS" \
  --top-k "$TOP_K" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --cache-dir "$CACHE_DIR" \
  --shard "$SHARD_INDEX" "$SHARD_COUNT" \
  --resume

echo "Shard ${SHARD_INDEX} complete."
