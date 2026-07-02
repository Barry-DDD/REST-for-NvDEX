#!/usr/bin/env bash
set -euo pipefail


## Before this step, read merged simple root file 

BASE="/public/home/liuz1/work/26.03.18_rest/nubb_ex/output/detsim_normal_root"
SEL="${BASE}/selected_E1MeV"
LOG="log_detsim_run_add_filter.log"

mkdir -p "$SEL"
: > "$LOG"

# 每行：infile outname id tree thr
jobs=(
  "co60.root    co60.root    22 HitTree 1000.0"
  "u238.root    u238.root    23 HitTree 1000.0"
  "th232.root   th232.root   24 HitTree 1000.0"
  "2nubb_gs.root   2nubb_gs.root   21 HitTree 1000.0"
  "2nubb_ex.root 2nubb_ex.root 1  HitTree 1000.0"
)

i=1
total=${#jobs[@]}
for line in "${jobs[@]}"; do
  read -r infile outfile id tree thr <<<"$line"
  inpath="$BASE/$infile"
  outpath="$SEL/$outfile"
  cmd="root -l -b -q 'add_id_and_filter.C(\"$inpath\",\"$outpath\",${id},\"${tree}\",${thr})'"
  echo "[$(date +'%F %T')] ($i/$total) $cmd" | tee -a "$LOG"
  eval $cmd >>"$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date +'%F %T')] Command $i failed (rc=$rc). See $LOG" | tee -a "$LOG"
    exit $rc
  fi
  echo "[$(date +'%F %T')] Done ($i/$total)" | tee -a "$LOG"
  i=$((i+1))
done

echo "All done. Log: $LOG"
