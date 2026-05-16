#!/usr/bin/env bash
# Run inside a Run:ai pod. Starts vLLM and runs the relevance judge for one shard.
#
# Required env vars:
#   SHARD_INDEX, SHARD_COUNT
#
# Optional env vars (all have defaults):
#   REPO_DIR, CORPUS_PATH, MODEL, TENSOR_PARALLEL, MAX_MODEL_LEN, MAX_NUM_SEQS,
#   GPU_MEMORY_UTILIZATION, GDN_PREFILL_BACKEND, KV_CACHE_DTYPE, WORKERS,
#   MAX_TOKENS, TEMPERATURE, HF_API_KEY_FILE_AT, LIMIT (0 = no limit; set >0 for test runs)

set -euo pipefail

REPO_DIR="${REPO_DIR:-/lightscratch/users/yiren/mamaretrieval}"
CORPUS_PATH="${CORPUS_PATH:-/lightscratch/users/yiren/mamai-medical-guidelines/processed/chunks_for_rag.txt}"
MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
WORKERS="${WORKERS:-8}"
MAX_TOKENS="${MAX_TOKENS:-25000}"   # leaves >5k tokens headroom for prompt (default vLLM max_model_len = 32768)
TEMPERATURE="${TEMPERATURE:-0.0}"
JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-1800}"
THINKING_BUDGET="${THINKING_BUDGET:-0}"
LIMIT="${LIMIT:-0}"
RUBRIC="${RUBRIC:-v1_boolean}"   # v1_boolean (legacy) or v2_graded (Phase 4)
RAW_OUTPUT="${RAW_OUTPUT:-}"     # path for raw-response side file (v2 only); empty = auto

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

# Allow torch.compile to finish before NCCL watchdog fires (compile takes ~10min)
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

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
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "$VLLM_PID" > "logs/vllm_judge_shard${SHARD_INDEX}.pid"

echo "Waiting for vLLM to become ready..."
_vllm_ready=0
for _attempt in $(seq 1 360); do
  if grep -q "Application startup complete" "$VLLM_LOG" 2>/dev/null; then
    echo "vLLM ready (startup complete in log)."
    _vllm_ready=1
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited before becoming ready. Last log lines:" >&2
    tail -n 120 "$VLLM_LOG" >&2 || true
    exit 1
  fi
  sleep 10
done
if [[ "$_vllm_ready" -eq 0 ]]; then
  echo "ERROR: vLLM did not become ready within 60 minutes (torch.compile may have hung). Last log lines:" >&2
  tail -n 40 "$VLLM_LOG" >&2 || true
  kill "$VLLM_PID" 2>/dev/null || true
  exit 1
fi

INPUT="${INPUT:-data/candidates.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-data}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-relevance_labels}"
OUTPUT="${OUTPUT:-${OUTPUT_DIR}/${OUTPUT_PREFIX}_shard${SHARD_INDEX}.jsonl}"
mkdir -p "$OUTPUT_DIR"

LIMIT_ARGS=()
RESUME_ARGS=(--resume)
if [[ "$LIMIT" -gt 0 ]]; then
  LIMIT_ARGS=(--limit "$LIMIT" --shuffle)
  RESUME_ARGS=()   # keep error records in test mode so we can inspect them
  echo "Test mode: limiting to $LIMIT queries (shuffled), no dedup."
fi

echo "Starting relevance judge: shard ${SHARD_INDEX}/${SHARD_COUNT}, workers=${WORKERS}, rubric=${RUBRIC}"
python3 -u scripts/judge_relevance.py \
  --backend openai \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --model "$MODEL" \
  --input "$INPUT" \
  --corpus "$CORPUS_PATH" \
  --output "$OUTPUT" \
  --shard "$SHARD_INDEX" "$SHARD_COUNT" \
  --workers "$WORKERS" \
  --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --timeout "$JUDGE_TIMEOUT" \
  --rubric "$RUBRIC" \
  ${RAW_OUTPUT:+--raw-output "$RAW_OUTPUT"} \
  ${THINKING_BUDGET:+--thinking-budget "$THINKING_BUDGET"} \
  "${RESUME_ARGS[@]}" \
  "${LIMIT_ARGS[@]}"

echo "Shard ${SHARD_INDEX} complete. Output: $OUTPUT"
