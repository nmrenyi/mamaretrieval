#!/usr/bin/env bash
# Run inside a Run:ai pod. Starts vLLM and runs the relevance judge for one shard.
#
# Required env vars:
#   SHARD_INDEX, SHARD_COUNT
#
# Optional env vars (all have defaults):
#   REPO_DIR, CORPUS_PATH, MODEL, TENSOR_PARALLEL, MAX_MODEL_LEN, MAX_NUM_SEQS,
#   GPU_MEMORY_UTILIZATION, GDN_PREFILL_BACKEND, WORKERS, MAX_TOKENS, TEMPERATURE,
#   HF_API_KEY_FILE_AT

set -euo pipefail

REPO_DIR="${REPO_DIR:-/lightscratch/users/yiren/mamaretrieval}"
CORPUS_PATH="${CORPUS_PATH:-/lightscratch/users/yiren/mamai-medical-guidelines/processed/chunks_for_rag.txt}"
MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}"
WORKERS="${WORKERS:-8}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"

: "${SHARD_INDEX:?ERROR: SHARD_INDEX must be set}"
: "${SHARD_COUNT:?ERROR: SHARD_COUNT must be set}"

export HOME="${RUNAI_HOME:-$REPO_DIR/runai_home}"
export HF_HOME="${HF_HOME:-$REPO_DIR/hf_cache}"
export PYTHONUSERBASE="${PYTHONUSERBASE:-$REPO_DIR/python_user}"
export PATH="$PYTHONUSERBASE/bin:$HOME/.local/bin:$PATH"

cd "$REPO_DIR"
mkdir -p logs data "$HOME" "$HF_HOME" "$PYTHONUSERBASE"

HF_API_KEY_FILE_AT="${HF_API_KEY_FILE_AT:-/lightscratch/users/yiren/keys/hf_key.txt}"
if [[ -f "$HF_API_KEY_FILE_AT" ]]; then
  export HF_TOKEN="$(cat "$HF_API_KEY_FILE_AT")"
  echo "HF token loaded from $HF_API_KEY_FILE_AT"
else
  echo "WARNING: HF token file not found at $HF_API_KEY_FILE_AT — model downloads may fail" >&2
fi

# Install vLLM if missing or outdated
python3 - <<'PY'
import importlib.metadata
import importlib.util
import subprocess
import sys


def version_tuple(value: str) -> tuple[int, ...]:
    parts = []
    for part in value.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        parts.append(int(digits or 0))
    return tuple(parts)


needs_install = importlib.util.find_spec("vllm") is None
if not needs_install:
    try:
        needs_install = version_tuple(importlib.metadata.version("vllm")) < (0, 19, 0)
    except importlib.metadata.PackageNotFoundError:
        needs_install = True

if needs_install:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--user", "--upgrade", "vllm>=0.19.0"]
    )
PY

# Start vLLM server in the background
VLLM_LOG="logs/vllm_judge_shard${SHARD_INDEX}.log"
vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --reasoning-parser qwen3 \
  --tensor-parallel-size "$TENSOR_PARALLEL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --language-model-only \
  --gdn-prefill-backend "$GDN_PREFILL_BACKEND" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "$VLLM_PID" > "logs/vllm_judge_shard${SHARD_INDEX}.pid"

echo "Waiting for vLLM to become ready..."
for _attempt in $(seq 1 180); do
  if python3 - <<'PY' >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=2).read()
PY
  then
    echo "vLLM ready."
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited before becoming ready. Last log lines:" >&2
    tail -n 120 "$VLLM_LOG" >&2 || true
    exit 1
  fi
  sleep 10
done

OUTPUT="data/relevance_labels_shard${SHARD_INDEX}.jsonl"

echo "Starting relevance judge: shard ${SHARD_INDEX}/${SHARD_COUNT}, workers=${WORKERS}"
python3 -u scripts/judge_relevance.py \
  --backend openai \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --model "$MODEL" \
  --input data/candidates.jsonl \
  --corpus "$CORPUS_PATH" \
  --output "$OUTPUT" \
  --shard "$SHARD_INDEX" "$SHARD_COUNT" \
  --workers "$WORKERS" \
  --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --resume

echo "Shard ${SHARD_INDEX} complete. Output: $OUTPUT"
