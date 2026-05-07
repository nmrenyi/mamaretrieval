#!/usr/bin/env bash
# Run inside a Run:ai pod. Starts vLLM and captures pretty thinking outputs.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/lightscratch/users/yiren/mamaretrieval}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B-FP8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

export HOME="${RUNAI_HOME:-$REPO_DIR/runai_home}"
export HF_HOME="${HF_HOME:-$REPO_DIR/hf_cache}"
export PYTHONUSERBASE="${PYTHONUSERBASE:-$REPO_DIR/python_user}"
export PATH="$PYTHONUSERBASE/bin:$HOME/.local/bin:$PATH"

cd "$REPO_DIR"
mkdir -p logs data "$HOME" "$HF_HOME" "$PYTHONUSERBASE"

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

VLLM_LOG="logs/qwen36_27b_fp8_vllm.log"
vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --reasoning-parser qwen3 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --language-model-only \
  --gdn-prefill-backend "$GDN_PREFILL_BACKEND" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "$VLLM_PID" > logs/qwen36_27b_fp8_vllm.pid

for _attempt in $(seq 1 180); do
  if python3 - <<'PY' >/dev/null 2>&1
import urllib.request

urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=2).read()
PY
  then
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited before becoming ready. Last log lines:" >&2
    tail -n 120 "$VLLM_LOG" >&2 || true
    exit 1
  fi
  sleep 10
done

python3 - <<'PY' >/dev/null
import urllib.request

urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5).read()
PY

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="data/qwen36_27b_fp8_top20_thinking_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUTPUT_DIR"

FAILED=0
FAILED_COUNT=0
for line in $(seq 1 20); do
  printf -v padded "%02d" "$line"
  echo "Capturing chunk line $line..."
  if ! python3 scripts/capture_qwen_thinking_example.py \
    --backend openai \
    --base-url http://127.0.0.1:8000/v1 \
    --api-key EMPTY \
    --model "$MODEL" \
    --input data/sampled_chunks.jsonl \
    --chunk-line "$line" \
    --output-prefix "$OUTPUT_DIR/line_${padded}" \
    --num-predict "$MAX_TOKENS" \
    --timeout 3600 \
    --pretty-only; then
    FAILED=1
    FAILED_COUNT=$((FAILED_COUNT + 1))
    echo "FAILED line $line" > "$OUTPUT_DIR/line_${padded}_error.txt"
  fi
done

echo "Output directory: $OUTPUT_DIR"
find "$OUTPUT_DIR" -maxdepth 1 -type f -name "*_pretty.txt" -print | sort
echo "Capture failures: $FAILED_COUNT"
if [[ "$FAILED" -ne 0 ]]; then
  echo "One or more captures failed; keeping diagnostic outputs without restarting the Run:ai job." >&2
fi
exit 0
