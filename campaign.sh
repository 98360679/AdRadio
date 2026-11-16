#!/bin/bash
# campaign.sh — generic OTA attack campaign (E2 single-channel, E3 two-channel).
#
# The attack dir and ch0 replay are REQUIRED env vars (no dated hardcode):
#   A=/path/to/attack_<date>     the output dir (holds ch0, banks, scores)
#   CH0=$A/ch0_run3_<N>.bin      the matched victim replay (from build_ch0_matched.py)
# Optional: LOG (default $A/logs), plus MODEL/SCALE/EPS_GRID/METHODS/PERT_D1/PERT_UNT/
#           TGT/RUN/SECS_2CH/SECS_SC/PASSES_SC/GUARD_FRAC/BER_PASSES/KEEP_CAPTURES.
#
#   A=... CH0=... ./campaign.sh sc_tgt_d1 | sc_unt | ch2_tgt_d1 | ch2_unt | all
set -u
cd "$(dirname "$(readlink -f "$0")")"
A="${A:?set A=/path/to/attack_<date> (the campaign output dir)}"
CH0="${CH0:?set CH0=\$A/ch0_run3_<N>.bin (the matched victim replay)}"
LOG="${LOG:-$A/logs}"; mkdir -p "$LOG"
FP=65661                 # matched ch0, padded to the bundle spacing (24% duty)
METHODS="${METHODS:-fgsm pgd_a0.01 pgd_a0.05 pgd_a0.10}"
# grid starts at 0.01: the 18 finer values below it are dead in EVERY arm measured
# (delta is <1.4% of payload rms there) and only cost airtime.
KEEP_CAPTURES="${KEEP_CAPTURES:-0}"  # 1 = retain rx_*.bin so a scorer fix can be applied
                                # OFFLINE. Four re-runs of one arm cost ~25 min of transmit
                                # EACH, purely because the capture had been deleted.
GUARD_FRAC="${GUARD_FRAC:-0.12}"  # classifier-only boundary trim. 0.12 discards frames 0..16
                                # (17 of 136) per block; the value was hardcoded and NEVER
                                # measured. It guards block MISASSIGNMENT from RX clock drift,
                                # not frame damage -- block edges are exactly frame-aligned
                                # (L = 136 x 65661). PASSES>=2 removes the need entirely.
BER_PASSES="${BER_PASSES:-1}"   # >1 unions decode passes by frame id. Measured NO-OP on real
                                # captures (136->136->136); the ~20%% loss that motivated it is
                                # specific to the ZERO-PADDED clean ch0 file, whose exact-zero
                                # gaps make sync_short's normalised autocorrelation unstable.
EPS_GRID="${EPS_GRID:-0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"
MODEL="${MODEL:?set MODEL=fingerprint_cvnn<date>_payload.pt}"
VICTIM="${VICTIM:-device_5}"   # the victim class (enrolled last); override if your victim differs
# Deltas are crafted against the RECEIVED clean_run_3 (that is where the victim's
# fingerprint lives), so they come out in the RECEIVED amplitude domain. ch0 transmits at
# TX level. Measured on the 103 crafted frames: received active rms 0.03399 vs ch0 0.29875,
# so delta must be scaled by 8.789 (18.88 dB) or every labelled PSR is ~19 dB optimistic.
# Leaving this at 1.0 produced the 8/15 run 1 null result: 0.0% at EVERY epsilon and a
# completely untouched link at eps 1.0, because every point was ~19 dB under budget.
# (This is the same quantity as the 8.4697 factor recorded earlier.)
SCALE="${SCALE:-8.7625}"
RUN=${RUN:-1}          # run index, so repeats do not overwrite each other
# one ch0 pass = 0.442 s; 17 eps
SECS_2CH="${SECS_2CH:-150}"   # env-overridable; must exceed blocks x 1.35s (19 eps x 4 phase = 103 s)
PASSES_SC=${PASSES_SC:-1}   # 2 = discard a settle pass (fixes start-up contamination)
SECS_SC=${SECS_SC:-60}      # 19 eps x 1 phase x 2 passes x 1.353 s = 51 s + margin

run_2ch () {            # $1 tag, $2 pert dir, $3 extra score args
  local TAG=$1 PERT=$2 EXTRA=$3
  for M in $METHODS; do
    echo "=========== $M  (2-channel $TAG, run $RUN) ==========="
    rm -f $A/ch1_${TAG}_r${RUN}_${M}.bin
    python3 -u build_ch1.py --pert-dir $PERT --ch0 $CH0 --method $M --frame-period $FP --passes 1 --epsilons "$EPS_GRID" \
        ${PHASES:+--phases $PHASES} \
        --scale $SCALE --out $A/ch1_${TAG}_r${RUN}_${M}.bin --index-out $A/blk_${TAG}_r${RUN}_${M}.csv \
        > $LOG/09_${TAG}_r${RUN}_build_${M}.log 2>&1 || { echo "BUILD FAILED $M"; continue; }
    tail -1 $LOG/09_${TAG}_r${RUN}_build_${M}.log
    python3 -u run_2ch.py --ch0 $CH0 --ch1 $A/ch1_${TAG}_r${RUN}_${M}.bin \
        --out $A/rx_${TAG}_r${RUN}_${M}.bin --secs $SECS_2CH --tx-gain 0.493 --adv-gain 0.493 \
        --rx-gain 0.93 > $LOG/09_${TAG}_r${RUN}_run_${M}.log 2>&1 || { echo "RUN FAILED $M"; rm -f $A/ch1_${TAG}_r${RUN}_${M}.bin $A/rx_${TAG}_r${RUN}_${M}.bin; continue; }
    grep "capture complete" $LOG/09_${TAG}_r${RUN}_run_${M}.log
    python3 -u score_2ch.py --capture $A/rx_${TAG}_r${RUN}_${M}.bin --blocks $A/blk_${TAG}_r${RUN}_${M}.csv \
        --ch0 $CH0 --model $MODEL --victim "$VICTIM" $EXTRA --pert-dir $PERT --method $M --ber --guard-frac $GUARD_FRAC --decode-passes $BER_PASSES \
        --out $A/score_${TAG}_r${RUN}_${M}.csv > $LOG/09_${TAG}_r${RUN}_score_${M}.log 2>&1 || echo "SCORE FAILED $M"
    sed -n '/=== 2-channel OTA/,$p' $LOG/09_${TAG}_r${RUN}_score_${M}.log
    # delete ONLY if the score CSV exists: a scoring bug must not cost a 300 s transmit
    if [ "$KEEP_CAPTURES" = "1" ]; then
      rm -f $A/ch1_${TAG}_r${RUN}_${M}.bin   # regenerable from the delta bank; the CAPTURE is not
      echo "  KEEPING capture $A/rx_${TAG}_r${RUN}_${M}.bin (KEEP_CAPTURES=1)"
    elif [ -s $A/score_${TAG}_r${RUN}_${M}.csv ]; then
      rm -f $A/ch1_${TAG}_r${RUN}_${M}.bin $A/rx_${TAG}_r${RUN}_${M}.bin $A/rx_${TAG}_r${RUN}_${M}.bin.rxlog
    else
      echo "  KEEPING capture $A/rx_${TAG}_r${RUN}_${M}.bin — scoring produced no CSV"
    fi
  done
}

run_sc () {             # $1 tag, $2 pert dir, $3 extra score args
  local TAG=$1 PERT=$2 EXTRA=$3
  for M in $METHODS; do
    echo "=========== $M  (single-channel $TAG, run $RUN) ==========="
    rm -f $A/ch0sc_${TAG}_r${RUN}_${M}.bin
    python3 -u build_ch0_sc.py --pert-dir $PERT --ch0 $CH0 --method $M --frame-period $FP --passes $PASSES_SC --epsilons "$EPS_GRID" \
        --scale $SCALE --headroom 0.95 --out $A/ch0sc_${TAG}_r${RUN}_${M}.bin --index-out $A/blk_${TAG}_r${RUN}_${M}.csv \
        --zeros-out $A/ch1_zeros.bin > $LOG/09_${TAG}_r${RUN}_build_${M}.log 2>&1 \
        || { echo "BUILD FAILED $M"; continue; }
    grep -c WARNING $LOG/09_${TAG}_r${RUN}_build_${M}.log | xargs -I{} echo "  clipping warnings: {}"
    python3 -u run_2ch.py --ch0 $A/ch0sc_${TAG}_r${RUN}_${M}.bin --ch1 $A/ch1_zeros.bin \
        --out $A/rx_${TAG}_r${RUN}_${M}.bin --secs $SECS_SC --tx-gain 0.493 --adv-gain 0.493 \
        --rx-gain 0.93 > $LOG/09_${TAG}_r${RUN}_run_${M}.log 2>&1 || { echo "RUN FAILED $M"; rm -f $A/ch0sc_${TAG}_r${RUN}_${M}.bin $A/rx_${TAG}_r${RUN}_${M}.bin; continue; }
    grep "capture complete" $LOG/09_${TAG}_r${RUN}_run_${M}.log
    python3 -u score_2ch.py --capture $A/rx_${TAG}_r${RUN}_${M}.bin --blocks $A/blk_${TAG}_r${RUN}_${M}.csv \
        --ch0 $CH0 --model $MODEL --victim "$VICTIM" $EXTRA --pert-dir $PERT --method $M --ber --guard-frac $GUARD_FRAC --decode-passes $BER_PASSES \
        --out $A/score_${TAG}_r${RUN}_${M}.csv > $LOG/09_${TAG}_r${RUN}_score_${M}.log 2>&1 || echo "SCORE FAILED $M"
    sed -n '/=== 2-channel OTA/,$p' $LOG/09_${TAG}_r${RUN}_score_${M}.log
    rm -f $A/ch0sc_${TAG}_r${RUN}_${M}.bin
    if [ "$KEEP_CAPTURES" = "1" ]; then
      echo "  KEEPING capture $A/rx_${TAG}_r${RUN}_${M}.bin (KEEP_CAPTURES=1)"
    else
      rm -f $A/rx_${TAG}_r${RUN}_${M}.bin $A/rx_${TAG}_r${RUN}_${M}.bin.rxlog
    fi
  done
}

# tgt arms score against TGT (default device_1); PERT_D1/PERT_UNT are the crafted bank dirs.
case "${1:-all}" in
  sc_tgt_d1)  run_sc  sc_tgt_d1  ${PERT_D1:?set PERT_D1=<bank dir>}  "--target ${TGT:-device_1}" ;;
  sc_unt)     run_sc  sc_unt     ${PERT_UNT:?set PERT_UNT=<bank dir>} "--select fool" ;;
  ch2_tgt_d1) run_2ch ch2_tgt_d1 ${PERT_D1:?set PERT_D1=<bank dir>}  "--target ${TGT:-device_1}" ;;
  ch2_unt)    run_2ch ch2_unt    ${PERT_UNT:?set PERT_UNT=<bank dir>} "--select fool" ;;
  all)
    run_sc  sc_tgt_d1  ${PERT_D1:?set PERT_D1=<bank dir>}  "--target ${TGT:-device_1}"
    run_sc  sc_unt     ${PERT_UNT:?set PERT_UNT=<bank dir>} "--select fool"
    run_2ch ch2_tgt_d1 ${PERT_D1:?set PERT_D1=<bank dir>}  "--target ${TGT:-device_1}"
    run_2ch ch2_unt    ${PERT_UNT:?set PERT_UNT=<bank dir>} "--select fool"
    ;;
  *) echo "usage: $0 [sc_tgt_d1|sc_unt|ch2_tgt_d1|ch2_unt|all]"; exit 1 ;;
esac
echo "=========== CAMPAIGN DONE ==========="
