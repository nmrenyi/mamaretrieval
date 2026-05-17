#!/usr/bin/env bash
# Tier 2 race supervisor: watch H200 + H100 judge jobs in parallel.
#
# - Emits state transitions per pool.
# - Reports vLLM loading % and judge progress (output line count).
# - When EITHER pool hits Succeeded: kills the other and rsyncs the
#   winning output back to data/audit/.
# - On terminal Failed/Error for a pool: resubmits that pool only (up to
#   MAX_ATTEMPTS) so the race continues.
#
# Plain bash 3.2 compatible (macOS default) — no associative arrays.

set -u

H200_JOB=mamaretrieval-judge-v2-full-h200-shard0
H100_JOB=mamaretrieval-judge-v2-full-h100-shard0
PROJ=light-yiren
SCRATCH=/mnt/light/scratch/users/yiren/mamaretrieval
H200_LOG="$SCRATCH/logs/vllm_judge_shard0.log"
H100_LOG="$SCRATCH/logs/vllm_judge_shard0_h100.log"
H200_OUT="$SCRATCH/data/audit/v2_full_h200_shard0.jsonl"
H100_OUT="$SCRATCH/data/audit/v2_full_h100_shard0.jsonl"
LOCAL_REPO=/Users/renyi/Downloads/mamaretrieval
MAX_ATTEMPTS=10

# Per-pool state (initialized below for both pools)
state_h200=""; load_h200=-1; ready_h200=0; judge_h200=0; lines_h200=0; ts_h200=0; att_h200=1
state_h100=""; load_h100=-1; ready_h100=0; judge_h100=0; lines_h100=0; ts_h100=0; att_h100=1

ssh_safe() { ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new light "$@" 2>/dev/null || true; }
emit() { echo "[$(date +%H:%M:%S)] $*"; }

job_for() { case "$1" in h200) echo "$H200_JOB";; h100) echo "$H100_JOB";; esac; }
log_for() { case "$1" in h200) echo "$H200_LOG";; h100) echo "$H100_LOG";; esac; }
out_for() { case "$1" in h200) echo "$H200_OUT";; h100) echo "$H100_OUT";; esac; }

resubmit() {
  local pool=$1
  local job=$(job_for "$pool")
  local cur_att new_att
  eval "cur_att=\$att_$pool"
  new_att=$((cur_att + 1))
  emit "RESUBMIT $pool (attempt $new_att/$MAX_ATTEMPTS)"
  ssh_safe "runai delete job $job --project $PROJ --suppress-deprecation-message 2>&1" \
    | tail -2 | sed "s/^/  delete: /"
  sleep 15
  cd "$LOCAL_REPO"
  local prefix=mamaretrieval-judge-v2-full-${pool}
  local out_prefix=v2_full_${pool}
  if [ "$pool" = "h100" ]; then
    VLLM_LOG_SUFFIX=_h100 \
    PYTHONUSERBASE_PATH=/lightscratch/users/yiren/mamaretrieval/python_user_judge_h100 \
    JOB_PREFIX=$prefix SHARD_COUNT=1 NODE_POOL=$pool RUBRIC=v2_graded \
    INPUT_PATH=data/audit/candidates_v2_full.jsonl \
    OUTPUT_DIR=data/audit OUTPUT_PREFIX=$out_prefix \
    THINKING_BUDGET=10000 THINKING_TOKEN_BUDGET=25000 MAX_TOKENS=0 \
    bash scripts/submit_judge_relevance.sh 2>&1 \
      | grep -iE "submitted successfully|ERROR" | head -3 | sed "s/^/  /"
  else
    JOB_PREFIX=$prefix SHARD_COUNT=1 NODE_POOL=$pool RUBRIC=v2_graded \
    INPUT_PATH=data/audit/candidates_v2_full.jsonl \
    OUTPUT_DIR=data/audit OUTPUT_PREFIX=$out_prefix \
    THINKING_BUDGET=10000 THINKING_TOKEN_BUDGET=25000 MAX_TOKENS=0 \
    bash scripts/submit_judge_relevance.sh 2>&1 \
      | grep -iE "submitted successfully|ERROR" | head -3 | sed "s/^/  /"
  fi
  eval "att_$pool=$new_att"
  eval "state_$pool=''"
  eval "load_$pool=-1"
  eval "ready_$pool=0"
  eval "judge_$pool=0"
  eval "lines_$pool=0"
  eval "ts_$pool=0"
}

declare_winner() {
  local winner=$1
  local loser
  if [ "$winner" = "h200" ]; then loser=h100; else loser=h200; fi
  emit "WINNER: $winner SUCCEEDED — killing $loser and syncing"
  ssh_safe "runai delete job $(job_for $loser) --project $PROJ --suppress-deprecation-message 2>&1" \
    | tail -2 | sed "s/^/  delete: /"
  local out_path=$(out_for $winner)
  local raw_path="${out_path%.jsonl}.raw.jsonl"
  emit "Syncing $out_path"
  rsync -av "light:$out_path" "light:$raw_path" data/audit/ 2>&1 | tail -3 | sed "s/^/  /"
  local local_jsonl="data/audit/$(basename $out_path)"
  local n=$(wc -l < "$local_jsonl" 2>/dev/null | tr -d ' ')
  emit "Local output: ${n:-0} rows (target 36418)"
}

while true; do
  for pool in h200 h100; do
    job=$(job_for "$pool")
    state_var="state_$pool"
    prev_state=""
    eval "prev_state=\$$state_var"

    state=$(ssh_safe "runai describe job $job --project $PROJ --suppress-deprecation-message 2>/dev/null | grep -E '^Status' | awk '{print \$2}'")
    if [ -n "$state" ] && [ "$state" != "$prev_state" ]; then
      eval "att_val=\$att_$pool"
      emit "[$pool a=$att_val] STATE: $prev_state -> $state"
      eval "$state_var='$state'"
    fi

    case "$state" in
      Succeeded)
        declare_winner "$pool"
        exit 0
        ;;
      Failed|Error)
        emit "[$pool] TERMINAL FAILURE"
        ssh_safe "tail -20 $(log_for $pool) 2>/dev/null" | tail -20 | sed "s/^/  LOG: /"
        eval "att_val=\$att_$pool"
        if [ "$att_val" -ge $MAX_ATTEMPTS ]; then
          emit "[$pool] max attempts reached — staying down"
          eval "$state_var=DEAD"
          continue
        fi
        resubmit "$pool"
        continue
        ;;
    esac

    eval "ready_val=\$ready_$pool"
    if [ "$state" = "Running" ] && [ "$ready_val" -eq 0 ]; then
      log=$(log_for "$pool")
      if ssh_safe "grep -q 'Application startup complete' $log"; then
        emit "[$pool] vLLM READY"
        eval "ready_$pool=1"
      else
        shards=$(ssh_safe "grep -oE 'Loading safetensors checkpoint shards: *[0-9]+%' $log | tail -1")
        if [ -n "$shards" ]; then
          pct=$(echo "$shards" | grep -oE '[0-9]+' | head -1)
          eval "load_val=\$load_$pool"
          if [ -n "$pct" ]; then
            if [ "$load_val" -eq -1 ] || [ $((pct - load_val)) -ge 25 ] || [ "$pct" = "100" ]; then
              emit "[$pool] loading: $pct%"
              eval "load_$pool=$pct"
            fi
          fi
        fi
      fi
    fi

    eval "ready_val=\$ready_$pool"
    if [ "$ready_val" -eq 1 ] && [ "$state" = "Running" ]; then
      out=$(out_for "$pool")
      cur_lines=$(ssh_safe "wc -l < $out 2>/dev/null" | tr -d ' ')
      cur_lines=${cur_lines:-0}
      eval "judge_val=\$judge_$pool"
      eval "prev_lines=\$lines_$pool"
      eval "prev_ts=\$ts_$pool"
      if [ "$judge_val" -eq 0 ] && [ "$cur_lines" -gt 0 ] 2>/dev/null; then
        emit "[$pool] JUDGE STARTED (lines=$cur_lines)"
        eval "judge_$pool=1"
        eval "lines_$pool=$cur_lines"
        eval "ts_$pool=$(date +%s)"
      elif [ "$judge_val" -eq 1 ]; then
        now=$(date +%s)
        if [ "$cur_lines" -gt "$prev_lines" ] 2>/dev/null && [ $((now - prev_ts)) -ge 300 ]; then
          emit "[$pool] PROGRESS: $cur_lines/36418 (+$((cur_lines - prev_lines)) in $((now - prev_ts))s)"
          eval "lines_$pool=$cur_lines"
          eval "ts_$pool=$now"
        fi
      fi
    fi
  done

  sleep 120
done
