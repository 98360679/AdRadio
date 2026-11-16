#!/bin/bash
# run_2ch.sh — both TWO-CHANNEL arms (E3) via the generic campaign.sh.
#
# Required env:  A     attack/output dir (holds ch0, banks, scores)
#                CH0   matched victim replay ($A/ch0_run3_<N>.bin)
#                MODEL fingerprint checkpoint
#                SCALE received->TX scale (README §5)
# Optional env:  METHODS EPS_GRID SECS_2CH GUARD_FRAC BER_PASSES KEEP_CAPTURES
#                TGT      (default device_1 — the example target; set to any target class)
#                PERT_D1  (default $A/pert_tgt_<TGT>_masked)   PERT_UNT (default $A/pert_unt_masked)
#                RUN      (default r1)
set -u
cd "$(dirname "$(readlink -f "$0")")"
: "${A:?set A=/path/to/attack_<date>}"
: "${CH0:?set CH0=\$A/ch0_run3_<N>.bin}"
: "${MODEL:?set MODEL=fingerprint_cvnn<date>_payload.pt}"
: "${SCALE:?set SCALE=<value from README §5>}"

export A CH0 MODEL SCALE
export KEEP_CAPTURES="${KEEP_CAPTURES:-0}" GUARD_FRAC="${GUARD_FRAC:-0}" BER_PASSES="${BER_PASSES:-1}" \
       SECS_2CH="${SECS_2CH:-150}" \
       METHODS="${METHODS:-fgsm bim mim pgd_a0.01 pgd_a0.05 pgd_a0.10}" \
       EPS_GRID="${EPS_GRID:-0.01,0.012,0.014,0.016,0.018,0.02,0.022,0.025,0.03,0.035,0.04,0.045,0.05,0.06,0.08,0.1}"
RUN="${RUN:-r1}"; TGT="${TGT:-device_1}"   # device_1 is the example default target
PERT_D1="${PERT_D1:-$A/pert_tgt_${TGT}_masked}"; PERT_UNT="${PERT_UNT:-$A/pert_unt_masked}"

echo "########## 1/2  ch2 targeted $TGT ##########"
RUN="$RUN" TGT="$TGT" PERT_D1="$PERT_D1" ./campaign.sh ch2_tgt_d1
echo "  exit=$?"
echo "########## 2/2  ch2 untargeted ##########"
RUN="$RUN" PERT_UNT="$PERT_UNT" ./campaign.sh ch2_unt
echo "  exit=$?"
echo "########## 2-CHANNEL DONE ##########"
ls -1 "$A"/score_ch2_*_r"$RUN"_*.csv 2>/dev/null || true
