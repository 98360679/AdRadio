#!/bin/bash
# enroll_d4_runs.sh — device_4, three runs as SEPARATE invocations.
#
# --run-ids 1,2,3 (one shared TX init) produced run_1 truncated to 628 MB / 51 frames
# and runs 2,3 completely empty (signal=None, 0 MB) with nothing logged. One init per
# run is slower but is the pattern that has always worked.
set -u
CAPTURES="${CAPTURES_ROOT:-$HOME/captures}"
cd "$(dirname "$0")"          # flowgraphs/ (enroll_calib.py lives here)
LOG="$PWD/logs"; mkdir -p "$LOG"      # local run logs
for R in 1 2 3; do
  echo "########## device_4 run $R ##########"
  python3 -u enroll_calib.py --device-id 4 --run-id $R \
      --tx-serial 3259368 --tx-antenna "TX/RX" --target-db -30 --g0 0.80 --probe-secs 8 \
      --tree enrollment > $LOG/enroll_d4_r$R.log 2>&1
  grep -E "^RESULT" $LOG/enroll_d4_r$R.log || echo "  (no RESULT line — run $R failed)"
  sleep 5
done
echo "########## device_4 DONE ##########"
ls -lh $CAPTURES/enrollment/device_4/clean_run_*.bin
