#!/bin/bash
# enroll_d5_runs.sh — device_5, three runs as SEPARATE invocations.
#
# --run-ids 1,2,3 (one shared TX init) produced run_1 truncated to 628 MB / 51 frames
# and runs 2,3 completely empty (signal=None, 0 MB) with nothing logged. One init per
# run is slower but is the pattern that has always worked.
set -u
CAPTURES="${CAPTURES_ROOT:-$HOME/captures}"
cd "$(dirname "$0")"          # flowgraphs/ (lab_exp3_enroll_calib.py lives here)
LOG="$PWD/logs"; mkdir -p "$LOG"      # local run logs
for R in 1 2 3; do
  echo "########## device_5 run $R ##########"
  python3 -u lab_exp3_enroll_calib.py --device-id 5 --run-id $R \
      --tx-serial 192.168.10.3 --tx-antenna "J1" --fix-gain 0.3689 --g0 0.3689 --probe-secs 8 \
      --tree enrollment > $LOG/enroll_d5_r$R.log 2>&1
  grep -E "^RESULT" $LOG/enroll_d5_r$R.log || echo "  (no RESULT line — run $R failed)"
  sleep 5
done
echo "########## device_5 DONE ##########"
ls -lh $CAPTURES/enrollment/device_5/clean_run_*.bin
