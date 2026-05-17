#!/usr/bin/env bash
# Tier 2 supervisor for the H100 judge job (solo, after H200 was killed).
#
# - Reports state transitions, vLLM load %, judge start, periodic progress.
# - Auto-resubmits on terminal Failed/Error (up to MAX_ATTEMPTS).
# - Exits 0 on Succeeded after syncing the output to data/audit/.
#
# Fixes the false-positive vLLM-ready check that was in supervise_pilot.sh:
# the previous version checked `if ssh_safe "grep -q ..."` but ssh_safe
# always returns 0 (it wraps with `|| true`), so the `if` always passed.
# Here we capture grep's stdout (non-empty iff match) and test that.

set -u

JOB=mamaretrieval-judge-v2-full-h100-shard0
PROJ=light-yiren
SCRATCH=/mnt/light/scratch/users/yiren/mamaretrieval
LOG="$SCRATCH/logs/vllm_judge_shard0_h100.log"
OUT="$SCRATCH/data/audit/v2_full_h100_shard0.jsonl"
RAW="$SCRATCH/data/audit/v2_full_h100_shard0.raw.jsonl"
LOCAL_REPO=/Users/renyi/Downloads/mamaretrieval
TARGET_ROWS=36418

attempt=1
max_attempts=10
prev_state=""
prev_loading_pct=-1
ready_seen=0
judge_seen=0
prev_lines=0
last_progress_ts=0
errhash=""

emit() { echo "[$(date +%H:%M:%S)] $*"; }

ssh_safe() {
  # SSH with connect AND server-alive timeouts so a hung session terminates.
  ssh -o ConnectTimeout=20 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 \
      -o StrictHostKeyChecking=accept-new light "$@" 2>/dev/null || true
}

resubmit() {
  emit "RESUBMIT attempt $((attempt+1))/$max_attempts"
  ssh_safe "runai delete job $JOB --project $PROJ --suppress-deprecation-message 2>&1" \
    | tail -2 | sed 's/^/  delete: /'
  sleep 15
  cd "$LOCAL_REPO"
  out=$(JOB_PREFIX=mamaretrieval-judge-v2-full-h100 \
        SHARD_COUNT=1 NODE_POOL=h100 RUBRIC=v2_graded \
        INPUT_PATH=data/audit/candidates_v2_full.jsonl \
        OUTPUT_DIR=data/audit OUTPUT_PREFIX=v2_full_h100 \
        THINKING_BUDGET=10000 THINKING_TOKEN_BUDGET=25000 MAX_TOKENS=0 \
        VLLM_LOG_SUFFIX=_h100 \
        PYTHONUSERBASE_PATH=/lightscratch/users/yiren/mamaretrieval/python_user_judge_h100 \
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
      emit "Syncing $OUT and ${OUT%.jsonl}.raw.jsonl"
      rsync -av "light:$OUT" "light:$RAW" data/audit/ 2>&1 | tail -3 | sed 's/^/  /'
      lines=$(wc -l < "data/audit/$(basename $OUT)" 2>/dev/null | tr -d ' ')
      raw_lines=$(wc -l < "data/audit/$(basename $RAW)" 2>/dev/null | tr -d ' ')
      emit "OUTPUT: lines=${lines:-0} raw=${raw_lines:-0} (target=$TARGET_ROWS)"
      exit 0
      ;;
    Failed|Error)
      emit "JOB TERMINAL FAILURE: $state (attempt $attempt)"
      # Give runai a chance to auto-recreate the pod (Error -> Running cycle
      # was seen during the H200/H100 race). Only resubmit if it stays
      # terminal for the next poll too.
      sleep 60
      recheck=$(ssh_safe "runai describe job $JOB --project $PROJ --suppress-deprecation-message 2>/dev/null | grep -E '^Status' | awk '{print \$2}'")
      if [ "$recheck" = "Running" ] || [ "$recheck" = "Pending" ]; then
        emit "[recovered] state is now $recheck — runai auto-recreated the pod"
        prev_state=$recheck
        # Reset readiness; the new pod starts fresh.
        ready_seen=0
        prev_loading_pct=-1
        continue
      fi
      ssh_safe "tail -30 $LOG 2>/dev/null" | tail -30 | sed 's/^/  LOG: /'
      if [ $attempt -ge $max_attempts ]; then
        emit "MAX ATTEMPTS REACHED ($max_attempts) — giving up"
        exit 1
      fi
      if ! resubmit; then
        emit "Resubmit failed, retry after 60s"
        sleep 60
        continue
      fi
      sleep 30
      continue
      ;;
  esac

  # vLLM model-load progress (only while Running and not yet ready)
  if [ "$state" = "Running" ] && [ "$ready_seen" -eq 0 ]; then
    # FIX: use stdout-non-empty test, not exit code (ssh_safe always returns 0)
    if [ -n "$(ssh_safe "grep 'Application startup complete' $LOG 2>/dev/null")" ]; then
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

  # Judge progress
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
      if [ "$lines" -gt "$prev_lines" ] 2>/dev/null && [ $((now - last_progress_ts)) -ge 600 ]; then
        emit "PROGRESS: $lines/$TARGET_ROWS (+$((lines - prev_lines)) in $((now - last_progress_ts))s)"
        prev_lines=$lines
        last_progress_ts=$now
      elif [ "$lines" -eq "$prev_lines" ] 2>/dev/null && [ $((now - last_progress_ts)) -ge 1200 ]; then
        emit "STALL: no new records in 20 min (lines=$lines)"
        last_progress_ts=$now
      fi
    fi
  fi

  sleep 180
done
