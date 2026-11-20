#!/bin/bash
# run_sc.sh — both SINGLE-CHANNEL arms (E2) via the generic campaign.sh.
#
# Required env:  A     attack/output dir (holds ch0, banks, scores)
#                CH0   matched victim replay ($A/ch0_run3_<N>.bin)
#                MODEL fingerprint checkpoint
#                SCALE received->TX scale (README §5)
# Optional env:  METHODS EPS_GRID SECS_SC PASSES_SC GUARD_FRAC BER_PASSES KEEP_CAPTURES
#                TGT      (default device_1 — the example target; set to any target class)
#                PERT_D1  (default $A/pert_tgt_<TGT>_sc)   PERT_UNT (default $A/pert_unt_sc)
#                RUN      (default 1)
set -u
cd "$(dirname "$(readlink -f "$0")")"
: "${A:?set A=/path/to/attack_<date>}"
: "${CH0:?set CH0=\$A/ch0_run3_<N>.bin}"
: "${MODEL:?set MODEL=fingerprint_cvnn<date>_payload.pt}"
: "${SCALE:?set SCALE=<value from README §5>}"

FINE=$(python3 -c "print(','.join(f'{round(0.1+0.01*i,2):g}' for i in range(41)))")
export A CH0 MODEL SCALE
export KEEP_CAPTURES="${KEEP_CAPTURES:-0}" GUARD_FRAC="${GUARD_FRAC:-0}" BER_PASSES="${BER_PASSES:-1}" \
       PASSES_SC="${PASSES_SC:-2}" SECS_SC="${SECS_SC:-130}" \
       METHODS="${METHODS:-fgsm bim mim pgd_a0.01 pgd_a0.05 pgd_a0.10}" \
       EPS_GRID="${EPS_GRID:-$FINE}"
RUN="${RUN:-1}"; TGT="${TGT:-device_1}"   # device_1 is the example default target
PERT_D1="${PERT_D1:-$A/pert_tgt_${TGT}_sc}"; PERT_UNT="${PERT_UNT:-$A/pert_unt_sc}"

echo "########## 1/2  sc targeted $TGT ##########"
RUN="$RUN" TGT="$TGT" PERT_D1="$PERT_D1" ../experiment_3/campaign.sh sc_tgt_d1
echo "  exit=$?"
echo "########## 2/2  sc untargeted ##########"
RUN="$RUN" PERT_UNT="$PERT_UNT" ../experiment_3/campaign.sh sc_unt
echo "  exit=$?"
echo "########## SINGLE-CHANNEL DONE ##########"
ls -1 "$A"/score_sc_*_r"$RUN"_*.csv 2>/dev/null || true
