#!/usr/bin/env bash
# Tier 3 supervisor: 2 shards on H100, each with WORKERS=32, judging
# data/audit/candidates_v2_top20_new.jsonl (194,546 new (q, c) pairs).
#
# Same logic as supervise_tier2_2shard.sh; only the JOB names, log paths,
# output paths, and TARGET_TOTAL differ. See that file for the design.

set -u

PROJ=light-yiren
SCRATCH=/mnt/light/scratch/users/yiren/mamaretrieval
LOCAL_REPO=/Users/renyi/Downloads/mamaretrieval

JOB0=mamaretrieval-judge-v2-top20-h100-shard0
JOB1=mamaretrieval-judge-v2-top20-h100-shard1
LOG0="$SCRATCH/logs/vllm_judge_shard0.log"
LOG1="$SCRATCH/logs/vllm_judge_shard1.log"
OUT0="$SCRATCH/data/audit/v2_top20_new_h100_shard0.jsonl"
OUT1="$SCRATCH/data/audit/v2_top20_new_h100_shard1.jsonl"
RAW0="$SCRATCH/data/audit/v2_top20_new_h100_shard0.raw.jsonl"
RAW1="$SCRATCH/data/audit/v2_top20_new_h100_shard1.raw.jsonl"
TARGET_TOTAL=194546
TARGET_PER_SHARD=$((TARGET_TOTAL / 2))

MAX_ATTEMPTS=10

state_0=""; load_0=-1; ready_0=0; judge_0=0; lines_0=0; ts_0=0; att_0=1; success_0=0
state_1=""; load_1=-1; ready_1=0; judge_1=0; lines_1=0; ts_1=0; att_1=1; success_1=0

ssh_safe() {
  ssh -o ConnectTimeout=20 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 \
      -o StrictHostKeyChecking=accept-new light "$@" 2>/dev/null || true
}
emit() { echo "[$(date +%H:%M:%S)] $*"; }

job_for() { case "$1" in 0) echo "$JOB0";; 1) echo "$JOB1";; esac; }
log_for() { case "$1" in 0) echo "$LOG0";; 1) echo "$LOG1";; esac; }
out_for() { case "$1" in 0) echo "$OUT0";; 1) echo "$OUT1";; esac; }
raw_for() { case "$1" in 0) echo "$RAW0";; 1) echo "$RAW1";; esac; }

resubmit() {
  local shard=$1
  local job=$(job_for "$shard")
  local cur_att new_att
  eval "cur_att=\$att_$shard"
  new_att=$((cur_att + 1))
  emit "[shard $shard] RESUBMIT attempt $new_att/$MAX_ATTEMPTS"
  ssh_safe "runai delete job $job --project $PROJ --suppress-deprecation-message 2>&1" \
    | tail -2 | sed "s/^/  delete: /"
  sleep 15
  cd "$LOCAL_REPO"
  out=$(JOB_PREFIX=mamaretrieval-judge-v2-top20-h100 \
        SHARD_COUNT=2 NODE_POOL=h100 RUBRIC=v2_graded \
        INPUT_PATH=data/audit/candidates_v2_top20_new.jsonl \
        OUTPUT_DIR=data/audit OUTPUT_PREFIX=v2_top20_new_h100 \
        THINKING_BUDGET=10000 THINKING_TOKEN_BUDGET=25000 MAX_TOKENS=0 \
        WORKERS=32 \
        bash scripts/submit_judge_relevance.sh 2>&1)
  if echo "$out" | grep -q "submitted successfully"; then
    emit "[shard $shard] RESUBMIT OK (both shards re-issued)"
    eval "att_$shard=$new_att"
    for s in 0 1; do
      eval "state_$s=''"
      eval "load_$s=-1"
      eval "ready_$s=0"
      eval "judge_$s=0"
      eval "lines_$s=0"
      eval "ts_$s=0"
    done
    return 0
  else
    emit "[shard $shard] RESUBMIT FAILED:"
    echo "$out" | tail -8 | sed "s/^/  /"
    return 1
  fi
}

while true; do
  for shard in 0 1; do
    job=$(job_for "$shard")
    state_var="state_$shard"
    eval "prev_state=\$$state_var"

    state=$(ssh_safe "runai describe job $job --project $PROJ --suppress-deprecation-message 2>/dev/null | grep -E '^Status' | awk '{print \$2}'")
    if [ -n "$state" ] && [ "$state" != "$prev_state" ]; then
      eval "att_val=\$att_$shard"
      emit "[shard $shard a=$att_val] STATE: $prev_state -> $state"
      eval "$state_var='$state'"
    fi

    case "$state" in
      Succeeded)
        eval "succ_val=\$success_$shard"
        if [ "$succ_val" -eq 0 ]; then
          emit "[shard $shard] SUCCEEDED"
          eval "success_$shard=1"
          emit "Syncing $(out_for $shard) and $(raw_for $shard)"
          rsync -av "light:$(out_for $shard)" "light:$(raw_for $shard)" data/audit/ 2>&1 | tail -3 | sed "s/^/  /"
        fi
        ;;
      Failed|Error)
        emit "[shard $shard] TERMINAL FAILURE: $state"
        sleep 60
        recheck=$(ssh_safe "runai describe job $job --project $PROJ --suppress-deprecation-message 2>/dev/null | grep -E '^Status' | awk '{print \$2}'")
        if [ "$recheck" = "Running" ] || [ "$recheck" = "Pending" ]; then
          emit "[shard $shard recovered] state is now $recheck"
          eval "$state_var='$recheck'"
          eval "ready_$shard=0"
          eval "load_$shard=-1"
          continue
        fi
        ssh_safe "tail -20 $(log_for $shard) 2>/dev/null" | tail -20 | sed "s/^/  LOG: /"
        eval "att_val=\$att_$shard"
        if [ "$att_val" -ge $MAX_ATTEMPTS ]; then
          emit "[shard $shard] max attempts reached"
          eval "$state_var=DEAD"
          continue
        fi
        resubmit "$shard"
        continue
        ;;
    esac

    eval "ready_val=\$ready_$shard"
    if [ "$state" = "Running" ] && [ "$ready_val" -eq 0 ]; then
      log=$(log_for "$shard")
      if [ -n "$(ssh_safe "grep 'Application startup complete' $log 2>/dev/null")" ]; then
        emit "[shard $shard] vLLM READY"
        eval "ready_$shard=1"
      else
        shards_line=$(ssh_safe "grep -oE 'Loading safetensors checkpoint shards: *[0-9]+%' $log | tail -1")
        if [ -n "$shards_line" ]; then
          pct=$(echo "$shards_line" | grep -oE '[0-9]+' | head -1)
          eval "load_val=\$load_$shard"
          if [ -n "$pct" ]; then
            if [ "$load_val" -eq -1 ] || [ $((pct - load_val)) -ge 25 ] || [ "$pct" = "100" ]; then
              emit "[shard $shard] loading: $pct%"
              eval "load_$shard=$pct"
            fi
          fi
        fi
      fi
    fi

    eval "ready_val=\$ready_$shard"
    if [ "$ready_val" -eq 1 ] && [ "$state" = "Running" ]; then
      out=$(out_for "$shard")
      cur_lines=$(ssh_safe "wc -l < $out 2>/dev/null" | tr -d ' ')
      cur_lines=${cur_lines:-0}
      eval "judge_val=\$judge_$shard"
      eval "prev_lines=\$lines_$shard"
      eval "prev_ts=\$ts_$shard"
      if [ "$judge_val" -eq 0 ] && [ "$cur_lines" -gt 0 ] 2>/dev/null; then
        emit "[shard $shard] JUDGE STARTED (lines=$cur_lines)"
        eval "judge_$shard=1"
        eval "lines_$shard=$cur_lines"
        eval "ts_$shard=$(date +%s)"
      elif [ "$judge_val" -eq 1 ]; then
        now=$(date +%s)
        if [ "$cur_lines" -gt "$prev_lines" ] 2>/dev/null && [ $((now - prev_ts)) -ge 1800 ]; then
          emit "[shard $shard] PROGRESS: $cur_lines/~$TARGET_PER_SHARD (+$((cur_lines - prev_lines)) in $((now - prev_ts))s)"
          eval "lines_$shard=$cur_lines"
          eval "ts_$shard=$now"
        elif [ "$cur_lines" -eq "$prev_lines" ] 2>/dev/null && [ $((now - prev_ts)) -ge 1800 ]; then
          emit "[shard $shard] STALL: no new records in 30 min (lines=$cur_lines)"
          eval "ts_$shard=$now"
        fi
      fi
    fi
  done

  eval "s0=\$success_0"
  eval "s1=\$success_1"
  if [ "$s0" -eq 1 ] && [ "$s1" -eq 1 ]; then
    emit "BOTH SHARDS SUCCEEDED — syncing complete"
    local0=$(wc -l < "data/audit/$(basename $OUT0)" 2>/dev/null | tr -d ' ')
    local1=$(wc -l < "data/audit/$(basename $OUT1)" 2>/dev/null | tr -d ' ')
    emit "Local rows: shard0=${local0:-0}, shard1=${local1:-0}, total=$((${local0:-0} + ${local1:-0})) / $TARGET_TOTAL"
    emit "To merge into final Tier 3 labels:"
    emit "  cat data/audit/v2_full_h100.jsonl \\"
    emit "      data/audit/v2_top20_new_h100_shard{0,1}.jsonl \\"
    emit "      > data/audit/v2_top20_all.jsonl"
    exit 0
  fi

  sleep 240
done
