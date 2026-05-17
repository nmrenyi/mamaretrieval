#!/usr/bin/env bash
# Supervisor for the Tier 1 v2.1 audit-scale pilot.
#
# Watches the H100 job; resubmits on terminal Failed/Error (pod-level
# preemption is handled automatically by runai). Exits 0 on Succeeded.

set -u

JOB=mamaretrieval-judge-v2-pilot-h100-shard0
PROJ=light-yiren
SCRATCH=/mnt/light/scratch/users/yiren/mamaretrieval
LOG="$SCRATCH/logs/vllm_judge_shard0.log"
OUT="$SCRATCH/data/audit/v2_pilot_h100_shard0.jsonl"
RAW="$SCRATCH/data/audit/v2_pilot_h100_shard0.raw.jsonl"
LOCAL_REPO=/Users/renyi/Downloads/mamaretrieval

attempt=1
max_attempts=20
prev_state=""
prev_loading_pct=-1
ready_seen=0
judge_seen=0
prev_lines=0
last_progress_ts=0
errhash=""

emit() { echo "[$(date +%H:%M:%S)] $*"; }

ssh_safe() {
  ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new light "$@" 2>/dev/null || true
}

resubmit() {
  emit "RESUBMIT attempt $((attempt+1))/$max_attempts"
  ssh_safe "runai delete job $JOB --project $PROJ --suppress-deprecation-message 2>&1" \
    | tail -2 | sed 's/^/  delete: /'
  sleep 15
  cd "$LOCAL_REPO"
  out=$(JOB_PREFIX=mamaretrieval-judge-v2-pilot-h100 SHARD_COUNT=1 NODE_POOL=h100 \
        RUBRIC=v2_graded INPUT_PATH=data/audit/candidates_v2_pilot.jsonl \
        OUTPUT_DIR=data/audit OUTPUT_PREFIX=v2_pilot_h100 \
        THINKING_BUDGET=10000 THINKING_TOKEN_BUDGET=25000 MAX_TOKENS=0 \
        bash scripts/submit_judge_relevance.sh 2>&1)
  if echo "$out" | grep -q "submitted successfully"; then
    emit "RESUBMIT OK"
    attempt=$((attempt+1))
    prev_state=""
    prev_loading_pct=-1
    ready_seen=0
    judge_seen=0
    prev_lines=0
    last_progress_ts=0
    errhash=""
    return 0
  else
    emit "RESUBMIT FAILED:"
    echo "$out" | tail -8 | sed 's/^/  /'
    return 1
  fi
}

while true; do
  state=$(ssh_safe "runai describe job $JOB --project $PROJ --suppress-deprecation-message 2>/dev/null | grep -E '^Status' | awk '{print \$2}'")

  if [ -n "$state" ] && [ "$state" != "$prev_state" ]; then
    emit "STATE[a=$attempt]: $prev_state -> $state"
    prev_state=$state
  fi

  case "$state" in
    Succeeded)
      emit "JOB SUCCEEDED on attempt $attempt"
      lines=$(ssh_safe "wc -l < $OUT 2>/dev/null" | tr -d ' ')
      raw_lines=$(ssh_safe "wc -l < $RAW 2>/dev/null" | tr -d ' ')
      errs=$(ssh_safe "grep -c '\"_error\":' $OUT 2>/dev/null" | tr -d ' ')
      emit "OUTPUT: lines=${lines:-0} raw=${raw_lines:-0} errors=${errs:-0} (target=1150)"
      exit 0
      ;;
    Failed|Error)
      emit "JOB TERMINAL FAILURE: $state (attempt $attempt)"
      ssh_safe "tail -30 $LOG 2>/dev/null" | tail -30 | sed 's/^/  LOG: /'
      if [ $attempt -ge $max_attempts ]; then
        emit "MAX ATTEMPTS REACHED ($max_attempts) — giving up"
        exit 1
      fi
      if ! resubmit; then
        emit "Resubmit failed, retry after 60s backoff"
        sleep 60
        continue
      fi
      sleep 30
      continue
      ;;
  esac

  # vLLM model load progress (only while Running, pre-ready)
  if [ "$state" = "Running" ] && [ "$ready_seen" -eq 0 ]; then
    if ssh_safe "grep -q 'Application startup complete' $LOG"; then
      emit "vLLM READY [a=$attempt]"
      ready_seen=1
    else
      shards=$(ssh_safe "grep -oE 'Loading safetensors checkpoint shards: *[0-9]+%' $LOG | tail -1")
      if [ -n "$shards" ]; then
        pct=$(echo "$shards" | grep -oE '[0-9]+' | head -1)
        if [ -n "$pct" ] && { [ "$prev_loading_pct" -eq -1 ] || [ $((pct - prev_loading_pct)) -ge 25 ] || [ "$pct" = "100" ]; }; then
          emit "loading: $pct%"
          prev_loading_pct=$pct
        fi
      fi
    fi
  fi

  # Critical errors in vLLM log
  errlines=$(ssh_safe "grep -nE 'Traceback|CUDA out of memory|RuntimeError|AssertionError|FATAL|exited before becoming ready|did not become ready' $LOG | tail -3")
  if [ -n "$errlines" ]; then
    new_hash=$(echo "$errlines" | md5sum | awk '{print $1}')
    if [ "$errhash" != "$new_hash" ]; then
      emit "LOG ERROR:"
      echo "$errlines" | sed 's/^/  /'
      errhash="$new_hash"
    fi
  fi

  # Judge progress (output line count)
  if [ "$ready_seen" -eq 1 ] && [ "$state" = "Running" ]; then
    lines=$(ssh_safe "wc -l < $OUT 2>/dev/null" | tr -d ' ')
    lines=${lines:-0}
    if [ "$judge_seen" -eq 0 ] && [ "$lines" -gt 0 ] 2>/dev/null; then
      emit "JUDGE STARTED (lines=$lines)"
      judge_seen=1
      prev_lines=$lines
      last_progress_ts=$(date +%s)
    elif [ "$judge_seen" -eq 1 ]; then
      now=$(date +%s)
      if [ "$lines" -gt "$prev_lines" ] 2>/dev/null && [ $((now - last_progress_ts)) -ge 180 ]; then
        emit "PROGRESS: $lines/1150 (+$((lines - prev_lines)) in $((now - last_progress_ts))s)"
        prev_lines=$lines
        last_progress_ts=$now
      elif [ "$lines" -eq "$prev_lines" ] 2>/dev/null && [ $((now - last_progress_ts)) -ge 900 ]; then
        emit "STALL: no new records in 15 min (lines=$lines)"
        last_progress_ts=$now
      fi
    fi
  fi

  sleep 90
done
