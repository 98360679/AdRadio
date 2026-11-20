# AdvRadio — Radio-Fingerprint Adversarial Attacks (Reproducibility Guide)

Over-the-air (OTA) adversarial attacks against a Wi-Fi (802.11 a/g) **device-fingerprinting**
neural network. A victim radio transmits; an **adversary** radio adds a crafted perturbation so the
receiver's classifier mis-identifies the victim as another enrolled device — ideally while the
802.11 link still decodes (BER ≈ 0). The study is organized as **five experiments**: a software upper
bound (Type I), a single-radio OTA attack (Type II), a realistic two-radio OTA attack (Type III),
robustness against authentication-model variations (Experiment 4), and adversary-realism variations
(Experiment 5).

This README is the **WiSec 2027 artifact entry point**: it documents the environment, every
experiment we ran, and the exact step-by-step commands to reproduce each result and figure. **This repo
is code only** — the scripts, the shipped result CSVs/figures (`docs/`), and the GNU Radio flowgraphs;
data and models are hosted externally (reproduction tiers in **§7**).

> ### 🛈 No SDR hardware? You can still reproduce almost everything.
> **Tiers 1–3 need no radios.** The over-the-air signals are already *recorded* and shipped, so:
> - **Tier 1** — regenerate **every figure and table** from the shipped `docs/*.csv` (Python only, nothing to download).
> - **Tiers 2–3** — recompute reachability / model-variation / validation and re-score the OTA fool%/BER by running the classifier on the shipped `.bin` recordings.
>
> Only **Tier 4** (collecting *new* transmissions) needs the testbed (§2). Models ship **pre-trained**, and every script runs on **CPU** (GPU optional). See §7 for the tier table.

> **Terminology (paper vs. code).** The paper calls the **1,024-sample model input** an **I/Q frame**
> (the sliding-window input, matching our prior work — 1,024 samples here vs. 288 there). In the code and
> this guide, that same unit is a **window** (`WIN = 1024`; see `frame_to_windows` in `fp_model.py`).
> The code/README **frame** (`FRAME_LEN = 15360`; e.g. "detected frames", `frames_decoded`,
> `ch0_run3_<N>.bin`) is the full 802.11 transmission, which the paper does not name separately and which
> yields **13 windows**. In short: **paper "I/Q frame" = code/README "window"**, and a code/README
> "frame" is one 802.11 transmission (13 windows each).

> **Note on "the paper."** There is no manuscript PDF in this repo; results are reproduced from the
> `plots/` scripts, the workbook `docs/exp3_all_results_byexp.xlsx`, and the CSVs under `docs/`.

---

## Contents

1. [What this reproduces](#1-what-this-reproduces)
2. [Environment & hardware](#2-environment--hardware)
3. [Repository layout](#3-repository-layout)
4. [Capture-to-attack pipeline (step by step)](#4-capture-to-attack-pipeline-step-by-step)
5. [Experiment catalog](#5-experiment-catalog)
6. [Quick start — reproduce results without radios](#6-quick-start--reproduce-results-without-radios)
7. [Datasets & models](#7-datasets--models)
8. [Regenerating every figure, table, and workbook](#8-regenerating-every-figure-table-and-workbook)
9. [Reading results](#9-reading-results)

---

## 1. What this reproduces

**Five experiments.** Experiments 1–3 vary *how* the perturbation δ reaches the receiver (increasing
physical realism); Experiments 4–5 vary *what* classifier is attacked and *how realistic* the adversary
is.

| Experiment 1-3 | What it measures | Radio? | δ scaling |
|---|---|---|---|
| **Experiment 1: Type I Baseline** | Software upper bound: smallest ℓ∞ radius to flip the victim to each class, shortfall Δ, ρ = R/ε_phys, proxy BER | no | — |
| **Experiment 2: Type II Adversary** | δ summed into the victim replay, radiated through the victim's own chain (ch1 silent) | yes | SCALE ≈ 13.06 |
| **Experiment 3: Type III Adversary** | Victim replay on ch0, δ on ch1 from a **separate MIMO-locked adversary antenna** (the realistic attack) | yes | SCALE = 1 |

**Experiment 4: Robustness Against Authentication Model Variations.** Does the fragility depend on the
classifier? Four variations (Exp 4.4 is reported as a table):

| Experiment 4 | One-line finding |
|---|---|
| **Exp 4.1: a real-valued model** | Fragility is task-, not architecture-, bound (CNN1D reproduces CVNN reachability, corr 0.987) |
| **Exp 4.2: transfer across models** | δ crafted on CVNN transfers to CNN1D and physics SVM/RF (τ ≈ 0.90, near-black-box) |
| **Exp 4.3: explicit physical features** | Partially resists (~2× budget) but is not a defense |
| **Exp 4.4: spectral representation** | Fragility is representation-invariant; geometry is representation-dependent (corr 0.324) |

**Experiment 5: Adversary Realism Variations.** How does the attack hold up under a more realistic
adversary? Two variations, both run through the Experiment 3 two-channel pipeline:

| Experiment 5 | One-line finding |
|---|---|
| **Exp 5.1: perturbation power** | Adversary power / measured received PSR sweep: two-channel fooling, target rate, and BER vs adversary gain; distance-ratio reach |
| **Exp 5.2: one perturbation for every frame** | Matched (per-frame) vs universal (one-waveform) δ — a realizable universal δ needs larger ε than per-frame matched δ |

---

## 2. Environment & hardware

### Software

```bash
pip install -r requirements.txt     # numpy, scipy, matplotlib, scikit-learn, torch
```
For GPU training install the CUDA build of `torch` from https://pytorch.org/. Everything except OTA
capture (Experiment 1, all scoring, all figure regeneration) needs only Python + PyTorch — no GNU Radio.

**Capture / OTA only (Experiments 2–3 and enrollment):** GNU Radio 3.10 + UHD, with the `gr-foo` and
`gr-ieee802-11` OOT modules **built and installed** (they are not vendored in this repo — obtain
`gr-ieee802-11` from https://github.com/98360679/gr-ieee802-11 and `gr-foo` from its upstream, both on
the `maint-3.10` branch):
```bash
cd gr-foo && mkdir build && cd build && cmake .. && make && sudo make install && sudo ldconfig
cd ../../gr-ieee802-11 && mkdir build && cd build && cmake .. && make && sudo make install && sudo ldconfig
```
(`gr-foo` first.) Python modules `gnuradio`, `ieee802_11`, `pmt` come from this system install, not pip.
The `flowgraphs/` scripts import `ieee802_11`.

### Hardware testbed (Experiments 2–3)

- **Victim** — USRP2/N-series `192.168.10.3`, antenna **J1**, TX gain **0.3689**, enrolled as the
  victim class (`device_5` in this testbed).
- **Adversary** — USRP2/N-series `192.168.10.4`, antenna **J1**, **MIMO-cabled to the victim**; the
  only attacker.
- **Other enrolled devices** — B200mini/B205mini (`device_1 3263606`, `device_2 32593C4`,
  `device_3 3259376`, `device_4 3259368`), swapped one at a time into TX/RX; these are the
  classifier's other legitimate classes = the impersonation *targets*, not attackers.
- **Receiver** — b205mini, **`rx_gain = 0.93` everywhere** — a mismatch makes the *unperturbed* victim misclassify and turns every fool% into a baseline artifact; the §4.5 gate catches it.

### Re-running from scratch on your own hardware (Tier 4)

You do **not** need our exact radios — the serials/IPs above are only what we used. To repeat the full
pipeline on a different SDR testbed you need, by **role**:

- **≥ 5 distinct transmitters** — one is the **victim**, the other four are the enrolled impersonation
  **targets** (any SDRs with stable, distinguishable hardware impairments; we used N-series + B20x-mini
  USRPs). They go through the TX chain one at a time during enrollment.
- **One adversary transmitter** (for two-channel Experiment 3) that is **clock/PPS-locked to the
  victim** — a shared 10 MHz reference + PPS so the two streams are phase-coherent (we MIMO-cabled two
  N-series; any UHD radios that can share a reference work).
- **One receiver.**

All radios run through **UHD + GNU Radio 3.10** (USRP family). Match these settings so results are comparable:

| setting | value |
|---|---|
| center frequency | **5.29 GHz** |
| sample rate | **5 MS/s** |
| modulation | 802.11a/g **BPSK, rate 1/2** |
| RX gain | **fixed** (we used 0.93), identical for enrollment and attack |
| TX gain | **calibrated per device** to a common target RX level (via `enroll_calib.py`) |

Then follow **§4 end-to-end**: enroll (clean runs per device), then train (§4.2), validate (§4.3), build
the victim replay (§4.4), pass the silent-adversary gate (§4.5), craft the δ banks (§4.6), and run the OTA
campaigns (§5, `run_sc.sh` / `run_2ch.sh`).

> **Expect different absolute numbers.** A device fingerprint is specific to the *physical radios*, so
> your classifier, its clean accuracy, and the exact PSR / fool% crossings **will differ** from ours —
> you train a new model on *your* hardware, not load ours. What reproduces is the **qualitative result**:
> a low-power adversary fools the OTA classifier while the 802.11 link still decodes. The most common
> pitfall is an **uncalibrated gain** that makes the *clean* victim misclassify — the §4.5 gate catches
> it in ~60 s.

---

## 3. Repository layout

```
AdvRadio/                               # WiSec 2027 artifact (code only)
├── README.md  requirements.txt  LICENSE
├── trained_models/                     # fingerprint_{cvnn,cnn1d,freqcvnn}.pt — classifiers (gitignored; hosted externally, §7)
│
├── lib/                                # shared modules (imported across every experiment)
│   ├── fp_model.py                #   CVNN input windowing — single source of truth
│   ├── cvnn.py  cnn1d.py  freqcvnn.py   # classifier architectures
│   ├── attack_matrix.py           #   attack crafting (FGSM/BIM/MIM/PGD)
│   └── extract_frames.py          #   energy-detect + window extraction
│
├── training/                           # enrollment, training, and validation of the victim
│   ├── flowgraphs/                     #   GNU Radio 3.10 — OTA capture (§4.1)
│   │   ├── enroll_calib.py    #     enrollment capture driver
│   │   ├── enroll_d{1..5}_runs.sh      #     per-device enrollment wrappers
│   │   └── wifi_tx_updated.py  wifi_rx.py  wifi_adversary_tx.py
│   ├── train_fingerprint.py       #   train the CVNN
│   └── validation_confusion.py    #   held-out val accuracy + confusion (95.98%)
│
├── experiment_1/   # Type I — digital reachability (no radio)
│   └── exp1_reachability.py  exp1_make_perturbation.py
├── experiment_2/   # Type II — single-channel OTA
│   └── build_ch0_sc.py  run_sc.sh
├── experiment_3/   # Type III — two-channel OTA  (the shared OTA pipeline)
│   └── run_2ch.{py,sh}  score_2ch.py  campaign.sh  build_ch1.py  build_ch0_matched.py
│       exp3_rx_capture.py  exp3_adversary_tx.py  exp3_make_ota_perturbations.py
│       exp3_map_detected_ids.py  exp3_ber_eval.py
├── experiment_4/   # Robustness vs authentication-model variation (Exp 4.1–4.4)
│   └── exp3_transfer_cnn1d_cvnn.py (Exp 4.2)   exp3_a3_feature_model.py (Exp 4.3)
├── experiment_5/   # Adversary-realism variation (Exp 5.1, 5.2) — reuses the experiment_3 pipeline
│   └── README.md                       #   Exp 5.1/5.2 run via experiment_3; figures in plots/ (see §6)
│
├── plots/          # ALL figure/graph generators (self-locating; run from anywhere)
│   ├── plot_exp{1,2,3}_*.py  plot_tables.py          # Exp 1/2/3 paper figures + result tables
│   ├── make_a1_compare.py  make_cnn1d_fooling_plot.py  make_transfer_plot.py
│   │   make_ota_multimodel_plot.py  make_a3_plot.py  # Exp 4 (A1–A4)
│   ├── make_b1_combined_cvnn.py  make_pwr_fool_ber.py  make_b2_matched_vs_unmatched.py  # Exp 5 (B1, B2)
│   └── make_results_workbook_byexp.py               # -> docs/…xlsx master index
│
└── docs/           # all result CSVs, rendered figures, workbooks, and write-ups
```

**Running scripts from the new layout.** Every entry script self-locates the repo root (the
directory holding `.git`), adds `lib/` to `sys.path`, and resolves `docs/` and the `.pt` models
relative to that root — so a script runs correctly **from any working directory**
(e.g. `python3 plots/make_b1_combined_cvnn.py` or `python3 experiment_1/exp1_reachability.py`).
The shared OTA runners (`experiment_3/campaign.sh`, `experiment_2/run_sc.sh`) reference each other
by repo-relative path, so the Experiment 2/3 campaigns work across the folder split.

**Hosted externally, not in git (§7):** the ~19 GB recompute bundle (enrollment + models) and the raw
OTA capture sessions (provenance).

**Required separately for OTA (§2):** the `gr-foo` and `gr-ieee802-11` GNU Radio 3.10 OOT modules, built
and installed. They are **not** vendored here; the `training/flowgraphs/` scripts import the installed
`ieee802_11` module.

**Navigating.** The canonical reproducible path is the §4 pipeline; the single authoritative data
index is `docs/exp3_all_results_byexp.xlsx`. Superseded/dated snapshots from the development repo were
**not** carried into this artifact.

---

## 4. Capture-to-attack pipeline (step by step)

Values in `<angle brackets>` are yours to fill: `<N>` number of
detected victim frames, `<scale>` from §4.5, `<victim>` the
victim label (e.g. `device_5`), `<V>` its 1-based class index (e.g. `5`), `<target>` the class to
impersonate, `<K>` the target's 1-based index.

| Stage | Script | Radio | § |
|---|---|---|---|
| 1. Collect enrollment | `training/flowgraphs/enroll_d{1..5}_runs.sh`, then `training/flowgraphs/enroll_calib.py` | yes | 6.1 |
| 2. Train model | `training/train_fingerprint.py` | no | 6.2 |
| 3. Validate | `training/validation_confusion.py` | no | 6.3 |
| 4. Build victim replay | `experiment_3/exp3_map_detected_ids.py`, `experiment_3/build_ch0_matched.py` | no | 6.4 |
| 5. Silent-adversary gate | `experiment_3/run_2ch.py`, `experiment_3/score_2ch.py` | yes | 6.5 |
| 6. Craft δ banks | `experiment_3/exp3_make_ota_perturbations.py` | no | 6.6 |
| Experiment 1: Type I Baseline | `experiment_1/exp1_reachability.py` | no | 7 |
| Experiment 2: Type II Adversary | `experiment_2/run_sc.sh` / `campaign.sh sc_*` | yes | 7 |
| Experiment 3: Type III Adversary | `experiment_3/run_2ch.sh` / `campaign.sh ch2_*` | yes | 7 |

### 4.1 Collect enrollment — `training/flowgraphs/enroll_calib.py`
Bring a TX radio up once, calibrate its gain to a target RX level, capture 3 clean runs/device via the
per-device wrappers (one radio init per run — the pattern that works).
Produces `$CAPTURES_ROOT/enrollment/device_N/clean_run_{1,2,3}.bin` (+ sidecars).
Key args: `--device-id N` (1–5), `--run-id`, `--tx-serial` (serial= for the B200/B205 radios, IP for the usrp2
victim), `--tx-antenna` (`TX/RX` or `J1`), `--target-db` (use **−30**), `--fix-gain` (victim only:
**0.3689**), `--g0`, `--date` (new every session), `--tree enrollment|validation`.
```bash
cd training/flowgraphs
# edit each wrapper: attacker devices --target-db -30 --g0 <start>, drop --fix-gain; victim keeps --fix-gain 0.3689
bash enroll_d1_runs.sh    # then swap radio, enroll_d2_runs.sh, d3, d4, d5
```
Each run ≈ 1 GB, 100+ frames, `bad_packet=no`, within tolerance of −30 dB.

### 4.2 Train — `training/train_fingerprint.py`
**Refit every session** — a reused prior-day model fails on a re-leveled enrollment. Needs a `_5dev`
symlink tree (`device_N` linking to `device_(N+1)`, **clean_run only**). Produces `fingerprint_<tag>.pt`.
Args: `--root`, `--val-run` (3), `--epochs` (use **80**),
`--tag`, `--lr`, `--cfo-correct`, `--rebuild`.
```bash
ENR=$CAPTURES_ROOT/enrollment
TRN=$CAPTURES_ROOT/train; rm -rf "$TRN"
for n in 1 2 3 4 5; do mkdir -p "$TRN/device_$n"
  for r in 1 2 3; do ln -s "$ENR/device_$((n+1))/clean_run_$r.bin" "$TRN/device_$n/clean_run_$r.bin"; done; done
python3 training/train_fingerprint.py --val-run 3 --epochs 80 --root "$TRN" --tag cvnn
#   -> trained_models/fingerprint_cvnn.pt
```

### 4.3 Validate — `training/validation_confusion.py`
Held-out run-3 per-device accuracy + confusion (leakage-safe). Want **victim = 100%** and the target class strong; overall frame-level **95.98%**.
```bash
python3 training/validation_confusion.py --model trained_models/fingerprint_cvnn.pt --base "$ENR"
```

### 4.4 Build the victim replay `ch0`
A clean baseband replay reproducing the victim frame-for-frame (base for crafting *and* the Experiment 2/3 ch0).
```bash
A=$CAPTURES_ROOT/attack_assets; mkdir -p "$A"; VIC="$ENR/<victim>"
python3 experiment_3/exp3_map_detected_ids.py --capture $VIC/clean_run_3.bin --rx-frames $VIC/rx_frames_run_3.jsonl \
    --out $A/detected_ids_run3.csv --thr-mult 2.0
python3 experiment_3/build_ch0_matched.py --dev-dir $VIC --run 3 --ids $A/detected_ids_run3.csv \
    --period 65661 --out $A/ch0_run3_<N>.bin        # N = number of detected frames
```
`--thr-mult 2.0` **must match** §4.6 so δ[k] ↔ ch0 frame k.

### 4.5 Silent-adversary gate — `experiment_3/run_2ch.py` + `experiment_3/score_2ch.py`
Prove the unperturbed victim replay reads **the victim class** through today's testbed before trusting any
campaign. Transmit ch0 + a zeros ch1 and score.
```bash
python3 - <<'PY'   # zeros ch1 + 1-row block map
import numpy as np
A="$CAPTURES_ROOT/attack_assets"; ch0=np.fromfile(f"{A}/ch0_run3_<N>.bin",dtype=np.complex64); n=len(ch0)
np.zeros(n,dtype=np.complex64).tofile(f"{A}/ch1_zeros.bin")
open(f"{A}/blk_control.csv","w").write("block,method,epsilon,phase_deg,start_sample,n_samples,score_from,score_to\n0,none,0.0,0.0,0,%d,0,%d\n"%(n,n))
PY
python3 experiment_3/run_2ch.py --ch0 $A/ch0_run3_<N>.bin --ch1 $A/ch1_zeros.bin --out $A/rx_control.bin \
    --secs 30 --tx-gain 0.493 --adv-gain 0.493 --rx-gain 0.93
python3 experiment_3/score_2ch.py --capture $A/rx_control.bin --blocks $A/blk_control.csv --ch0 $A/ch0_run3_<N>.bin \
    --model trained_models/fingerprint_cvnn.pt --victim <victim> --select fool --ber --guard-frac 0 --out $A/score_control.csv
```
**Pass = landed the victim class, fool 0%, BER 0.** Else fix the testbed (usually rx-gain). Then compute
**SCALE** = `ch0_active_rms / received_active_rms` (δ is crafted at received amplitude; the build
re-scales it to TX level) — see the snippet in the appendix of the per-session runlog; use it as
`<scale>` for Experiment 2 (two-channel Experiment 3 uses `SCALE=1`).

### 4.6 Craft δ banks — `experiment_3/exp3_make_ota_perturbations.py`
Optimize δ against the **received** victim, payload-only, MAC header masked. Produces a bank dir of
compact `.npy` deltas `(N, 13312)` + `manifest.csv`, one per method×ε.
Args: `--model`, `--ch0` (= the **received** capture), `--from-capture`, `--thr-mult 2.0` (match §4.4),
`--nper <N>`, `--mask-lead 780`, `--tgt-dst <K>` (target 1-based index) **or** `--untargeted`,
`--victim-class <V>`, `--methods` (default 6: `fgsm bim mim pgd_a0.01 pgd_a0.05 pgd_a0.10`), `--epsilons`.
```bash
RX=$ENR/<victim>/clean_run_3.bin
COMMON="--model trained_models/fingerprint_cvnn.pt --ch0 $RX --from-capture --thr-mult 2.0 --nper <N> --mask-lead 780 --victim-class <V>"
MASKED="0.01 0.012 0.014 0.016 0.018 0.02 0.022 0.025 0.03 0.035 0.04 0.045 0.05 0.06 0.08 0.1"          # Exp 3, 16 pts
FINE=$(python3 -c "print(' '.join(f'{round(0.1+0.01*i,2):g}' for i in range(41)))")                      # Exp 2, 41 pts 0.10-0.50
python3 experiment_3/exp3_make_ota_perturbations.py $COMMON --tgt-dst <K> --epsilons $MASKED --out-dir $A/pert_tgt_<target>_masked
python3 experiment_3/exp3_make_ota_perturbations.py $COMMON --tgt-dst <K> --epsilons $FINE   --out-dir $A/pert_tgt_<target>_sc
python3 experiment_3/exp3_make_ota_perturbations.py $COMMON --untargeted  --epsilons $MASKED --out-dir $A/pert_unt_masked
python3 experiment_3/exp3_make_ota_perturbations.py $COMMON --untargeted  --epsilons $FINE   --out-dir $A/pert_unt_sc
```

---

## 5. Experiment catalog

Each experiment below lists its one-line goal, how to reproduce it, the result data, and the figure it
produces. Shared setup (model + enrollment + δ banks) comes from §4. The master data index for every
experiment is `docs/exp3_all_results_byexp.xlsx`.

### Experiment 1 (Type I Baseline) — Classifier reachability (no radio)
```bash
python3 experiment_1/exp1_reachability.py --model trained_models/fingerprint_cvnn.pt --base "$ENR" --out reachability.csv
```
Data: `reachability_matrix.csv`, `reachability_shortfall_curves.{npz,csv}`,
`exp3_eps_phys_empirical.{csv,npz}`, `reachability_cvnn.csv`. Figures: `plots/make_e1_diagram.py`,
`plots/make_digital_vs_ota.py`. Produces the reachability table.

### Experiment 2 (Type II Adversary) — `campaign.sh sc_tgt_d1 | sc_unt`
δ summed into the victim replay (ch1 = zeros), fine grid 0.10–0.50. `campaign.sh` reads **exported**
`A` and `CH0`.
```bash
export A=$CAPTURES_ROOT/attack_assets; export CH0=$A/ch0_run3_<N>.bin
FINE=$(python3 -c "print(','.join(f'{round(0.1+0.01*i,2):g}' for i in range(41)))")
export MODEL=trained_models/fingerprint_cvnn.pt KEEP_CAPTURES=0 GUARD_FRAC=0 BER_PASSES=1 \
       PASSES_SC=2 SECS_SC=130 SCALE=<scale> METHODS="fgsm bim mim pgd_a0.01 pgd_a0.05 pgd_a0.10" EPS_GRID="$FINE"
export RUN=1 TGT=<target> PERT_D1=$A/pert_tgt_<target>_sc PERT_UNT=$A/pert_unt_sc
cd experiment_2 && ./run_sc.sh        # both single-channel arms (calls ../experiment_3/campaign.sh)
```
Output `$A/score_sc_{tgt_d1,unt}_r<RUN>_<method>.csv`. Figure: `plots/make_sc_vs_2ch_plot.py`.

### Experiment 3 (Type III Adversary) — `campaign.sh ch2_tgt_d1 | ch2_unt`
Victim replay on ch0, δ on ch1, MIMO-locked, masked 16-pt grid, **SCALE=1**. The Qt GUI shows per method.
```bash
export A=$CAPTURES_ROOT/attack_assets; export CH0=$A/ch0_run3_<N>.bin
export MODEL=trained_models/fingerprint_cvnn.pt KEEP_CAPTURES=0 GUARD_FRAC=0 BER_PASSES=1 SECS_2CH=150 \
       SCALE=1 METHODS="fgsm bim mim pgd_a0.01 pgd_a0.05 pgd_a0.10" \
       EPS_GRID="0.01,0.012,0.014,0.016,0.018,0.02,0.022,0.025,0.03,0.035,0.04,0.045,0.05,0.06,0.08,0.1"
export RUN=r1 TGT=<target> PERT_D1=$A/pert_tgt_<target>_masked PERT_UNT=$A/pert_unt_masked
cd experiment_3 && ./run_2ch.sh       # both two-channel arms
```
Output `$A/score_ch2_{tgt_d1,unt}_r<RUN>_<method>.csv`. The Experiment 3 paper figures (fooling, BER,
frame rejections) are rendered from embedded data by `plots/plot_exp3_sidebyside.py` and
`plots/plot_exp3_rejection.py`.

### Experiment 4 — Robustness against authentication-model variations
Does the fragility depend on the classifier? Four variations (Exp 4.4 is a table). Crafting as in §4.

- **Exp 4.1 — a real-valued model (CNN1D vs CVNN).** Train the real CNN1D baseline (`lib/cnn1d.py`), run its
  reachability, compare to CVNN. Data: `reachability_matrix_cnn1d.csv`, `docs/a1_*_ota_ber.csv`,
  `docs/a1cvnn_*_ota_ber.csv`. Figures: `plots/make_a1_compare.py` produces `docs/a1_cnn1d_vs_cvnn.*`,
  `plots/make_cnn1d_fooling_plot.py`.
- **Exp 4.2 — transfer across models.** δ crafted on CVNN applied to CNN1D and the physics SVM/RF.
  `experiment_4/exp3_transfer_cnn1d_cvnn.py`. Data: `exp3_transfer_cnn1d_cvnn.{csv,npz}`,
  `exp3_ota_multimodel.csv`. Figures: `plots/make_transfer_plot.py`, `plots/make_ota_multimodel_plot.py`.
  (Workbook sheets `A2-ML`, `A2-transfer-raw`.)
- **Exp 4.3 — explicit physical features (SVM/RF).** Feature-explicit classifier on explicit impairment features.
  `experiment_4/exp3_a3_feature_model.py`. Data: `exp3_a3_feature_model.{csv,npz}`. Figure:
  `plots/make_a3_plot.py` produces `docs/a3_feature_model_robustness.*`.
- **Exp 4.4 — spectral representation (frequency-domain CVNN).** FFT-first classifier (`lib/freqcvnn.py`); representation-invariance
  of the fragility, **reported as a table** (no figure). Data: `reachability_matrix_freqcvnn.csv`.

### Experiment 5 — Adversary realism variations
Both run through the Experiment 3 two-channel pipeline (§4); no dedicated scripts.

- **Exp 5.1 — perturbation power.** Two-channel fooling, target rate, and BER vs adversary gain and
  measured received PSR (ε=0.01); distance-ratio reach. Runs on `$CAPTURES_ROOT/attack_b1`. Data:
  `docs/b1_cvnn_foolber_<method>.csv` (untargeted), `docs/b1_cvnn_tgt_d3_foolber_<method>.csv`
  (targeted device_3). Figures: `plots/make_b1_combined_cvnn.py`, `plots/make_pwr_fool_ber.py`.
- **Exp 5.2 — one perturbation for every frame.** Per-frame matched δ vs one universal waveform reused on every frame
  (the realizable adversary). Data: `docs/b2_matched_vs_unmatched.csv`. Figure:
  `plots/make_b2_matched_vs_unmatched.py` produces `docs/b2_matched_vs_unmatched.*`.

---

## 6. Quick start — reproduce results without radios

This is **Tier 1** (§7): Python + the shipped `docs/` CSVs, nothing downloaded. Parts **A** and **B**
need no models; only part **C** (reachability) loads a `.pt` (optional — retrain via §4.2). Scripts
self-locate the repo root, so they run from any directory.

**A. Re-render the Experiment 3: Type III Adversary figures** from the shipped data:
```bash
python3 plots/plot_exp3_sidebyside.py    # fooling + BER vs ε (embedded data)
python3 plots/plot_exp3_rejection.py     # frame rejections vs ε
```

**B. Rebuild the master results workbook** (one sheet per Exp 4.1–4.4, Exp 5.1–5.2):
```bash
python3 plots/make_results_workbook_byexp.py   # -> docs/exp3_all_results_byexp.xlsx
```

**C. Run Experiment 1: Type I Classifier reachability** (needs a model in `trained_models/` + the enrollment
windows, no radio):
```bash
python3 experiment_1/exp1_reachability.py \
  --model trained_models/fingerprint_cvnn.pt \
  --base  $CAPTURES_ROOT/enrollment \
  --out   reachability_quickstart.csv
```

Full from-raw reproduction (capture, train, craft, OTA, score) is §4; per-experiment
reproduction is §5.

---

## 7. Datasets & models

Reproduction has tiers; **most reviewers need nothing beyond this repo.**

| Tier | Reproduces | Data to download | Size |
|---|---|---|---|
| **1 Results** | every figure, table, and the workbook | **nothing** — the shipped `docs/*.csv` in this repo (verified: the `plots/` scripts regenerate) | **0** |
| **2 Recompute** | Experiment 1 reachability, Experiment 4, retraining, feature-domain crafting (no radios) | the **minimal bundle** below | **~19 GB** |
| **3 Re-score raw OTA** | OTA fool%/BER from the raw `rx_*.bin` recordings (no radios) | the full attack capture dirs | ~90 GB |
| **4 Re-run OTA** | the whole pipeline | everything **+ the radios (§2)** | — |

So: **clone the repo and run the scripts to reproduce every reported result.** Download data only to
recompute from scratch.

### Minimal bundle (Tier 2) — OneDrive

> **Download:** <https://uofnebraska-my.sharepoint.com/:f:/g/personal/98360679_nebraska_edu/IgDLBegB2Vk1QKrjYaZdmbXrAdCCzBtYiBOjWo1bUS7E3fk?e=kaU0lv>  *(also cited in the paper)*
>
> After downloading, set `CAPTURES_ROOT` to the bundle root (it contains `enrollment/`, `models/`, `attack_assets/`) and copy `models/*.pt` into `trained_models/` (§7.3).

```
captures/
├── enrollment/           #   per-device enrollment captures (§7.1)   ~19 GB
└── attack_assets/        #   victim replay + δ waveform (ch0_run3_107.bin, ch1_e04_s1.bin)  ~0.3 GB
trained_models/           #   fingerprint_{cvnn,cnn1d,freqcvnn}.pt — place at repo root (§7.3)  ~4 MB
```

The raw OTA capture sessions (Tier 3, ~90 GB of `rx_*.bin`) are provenance for byte-level audit and are
**available on request / on the dataset DOI**, not in the minimal bundle.

**Setting the location.** Every script reads the captures root from the `CAPTURES_ROOT` environment
variable, defaulting to `~/captures`. Point it at wherever you unpacked the bundle:
```bash
export CAPTURES_ROOT=/path/to/captures
```
Then create the models folder and copy the three `.pt` (§7.3) into it:
```bash
mkdir -p trained_models && cp /path/to/bundle/models/*.pt trained_models/
```
(scripts take `--model trained_models/fingerprint_cvnn.pt`). The folder is not in the repo — you create it.

### 7.1 Enrollment dataset — `${CAPTURES_ROOT}/enrollment/`

- **Format:** complex64 interleaved I/Q at **Fs = 5 MHz**, ~30 s per run.
- **Classes:** 5 fingerprint classes **`device_1 … device_5`** (`NUM_CLASSES = 5` in `fp_model.py`;
  `device_5` = the adversary radio, captured as a victim class).
- **Layout:** `${CAPTURES_ROOT}/enrollment/device_<N>/` with
  `clean_run_{1,2,3}.bin` (~1.24–1.28 GB each) plus sidecars
  `frame_run_*.bin`, `frame_index_run_*.csv`, `rx_frames_run_*.jsonl`, `ber_frames_run_*.jsonl`,
  `tx_ids_run_*.csv`.
- **Split (leakage-safe):** per device, **runs 1,2 = train, run 3 = held-out validation**
  (no window leakage). `${CAPTURES_ROOT}/validation/` holds separate BER/decode validation captures,
  and `${CAPTURES_ROOT}/train/` is the 5-class symlink tree the trainer builds (§4.2).
- Windowing: `WIN = 1024` complex samples tiled **non-overlapping** (stride = `WIN`) over each frame's
  payload region `[PRE_ROLL : PRE_ROLL+ACTIVE] = [500 : 13812]` in `fp_model.py` — i.e. skipping the
  preamble and SIGNAL field (samples 100–500). Each window is the paper's **I/Q frame** (the model
  input), power-normalized to unit average power.
  - **13 I/Q frames per 802.11 frame:** `ACTIVE = 13312 = 13 × 1024` (the largest whole multiple of `WIN`
    fitting the payload; any leftover shorter than one window is dropped), so each 802.11 frame yields
    **13** windows. The counts follow: 6,731 Normal frames × 13 = 87,503 I/Q frames; the 107-frame OTA
    replay × 13 = 1,391.

### 7.2 Attack data — `${CAPTURES_ROOT}/attack_*/` (Tier 3, provenance)

These are the **raw OTA capture sessions** — needed only to re-score the over-the-air results from the
raw `rx_*.bin` recordings (Tier 3) or re-transmit them (Tier 4). The **derived result CSVs the figures
actually read ship in `docs/`** (Tier 1), so you do *not* need these dirs to regenerate any figure. The
only part in the minimal bundle is `attack_assets/`'s small TX waveforms (`ch0_run3_107.bin`,
`ch1_e04_s1.bin`). The role-based dirs:

| Directory | Role | Key contents |
|---|---|---|
| `attack_assets/` | OTA victim/adversary waveforms (Exp 2/3) | `ch0_run3_107.bin` (victim replay), `ch1_e04_s1.bin` (adversary δ) |
| `attack_b1/` | Exp 5.1 adversary-power sweep | `b1_cvnn_foolber_<method>.csv`, `b1_cvnn_tgt_d3_foolber_<method>.csv` |
| `attack_a1_transfer/` | Exp 4.2 cross-model transfer | per-model white-box score CSVs |
| `attack_a1_gain/` | Exp 4.1 gain sweep | `g16_<method>.csv` |
| `attack_enroll/` | attack re-enrollment baseline | `attack/ch0.bin`, `ota_perturbations/` (first δ bank) |

Filename legend: `ch0*.bin` = victim/legit replay · `ch1*.bin`, `pert_*/delta_*.npy` = adversary
waveform / δ bank · `rx*.bin(.rxlog)` = OTA capture / decode · `score_*.csv`, `blk_*.csv` = per-frame /
per-block fingerprint scores · `ber_*.csv` = decode BER · `*.sh` = run driver.

### 7.3 Trained models — `*.pt` (download into `trained_models/`)

**Models are optional** (Tier 1 needs none). The `.pt` are used only to recompute from the classifier
(§6C, §4–§5), and even then §4.2 retrains an equivalent model from the raw enrollment — the shipped
checkpoints just skip retraining and give bit-exact reproduction.

Three checkpoints, one per architecture, all scoring `WIN=1024` complex windows over 5 classes
(`device_1…5`). Download them into the repo root only if you need the recompute path:

| Model | Size | Arch | Role |
|---|---|---|---|
| **fingerprint_cvnn.pt** | 1.72&nbsp;M | CVNN | main classifier (Experiments 1–3, and the Experiment 5 sweeps) |
| fingerprint_cnn1d.pt | 490&nbsp;K | CNN1D | Exp 4.1 real-CNN baseline |
| fingerprint_freqcvnn.pt | 1.72&nbsp;M | FreqCVNN | Exp 4.4 frequency-domain variant |

Architectures: **CVNN** = time-domain complex CNN (`fp_model.py`), **FreqCVNN** = FFT-first
(`freqcvnn.py`), **CNN1D** = real 1-D CNN baseline (`cnn1d.py`).

---

## 8. Regenerating every figure, table, and workbook

All figure scripts live in `plots/` and are self-locating (run from any directory). The `plot_exp*`
scripts carry embedded data; the `make_*` scripts read the shipped `docs/*.csv`. Each writes `.png` +
`.svg` (several also `.pdf`) under `docs/` (the `plot_*` scripts default to `plots/output/`, overridable
with `FIG_OUT`).

**Master workbook (run first — the data index):**
`plots/make_results_workbook_byexp.py` produces `docs/exp3_all_results_byexp.xlsx` (sheets A1-ML, A2-ML,
A2-transfer-raw, A3-ML, A4-ML, B1-W, B2-W).

**Experiments 1–3 (main paper figures):** `plots/plot_exp1_sidebyside.py`,
`plots/plot_exp1_ber_sidebyside.py`, `plots/plot_exp1_singlecol.py` (Exp 1) · `plots/plot_exp2_sidebyside.py`
(Exp 2) · `plots/plot_exp3_sidebyside.py`, `plots/plot_exp3_rejection.py`, `plots/plot_exp3_untargeted.py`,
`plots/plot_exp3_targeted.py` (Exp 3) · `plots/plot_tables.py` (result tables). Supporting:
`plots/make_e1_diagram.py`, `plots/make_digital_vs_ota.py`, `plots/make_sc_vs_2ch_plot.py`.

**Experiment 4 (Exp 4.1–4.4):** `plots/make_a1_compare.py`, `plots/make_cnn1d_fooling_plot.py` (Exp 4.1) ·
`plots/make_transfer_plot.py`, `plots/make_ota_multimodel_plot.py` (Exp 4.2) · `plots/make_a3_plot.py` (Exp 4.3).
Exp 4.4 is reported as a table.

**Experiment 5 (Exp 5.1, 5.2):** `plots/make_b1_combined_cvnn.py`, `plots/make_pwr_fool_ber.py` (Exp 5.1) ·
`plots/make_b2_matched_vs_unmatched.py` (Exp 5.2).

Regenerate everything:
```bash
for f in plot_exp1_sidebyside plot_exp1_ber_sidebyside plot_exp2_sidebyside \
         plot_exp3_sidebyside plot_exp3_rejection plot_tables \
         make_e1_diagram make_digital_vs_ota make_sc_vs_2ch_plot \
         make_a1_compare make_cnn1d_fooling_plot make_transfer_plot make_ota_multimodel_plot make_a3_plot \
         make_b1_combined_cvnn make_pwr_fool_ber make_b2_matched_vs_unmatched \
         make_results_workbook_byexp; do
  echo "== $f =="; python3 "plots/$f.py" || echo "  (check inputs for $f)"
done
```

---

## 9. Reading results

Score-CSV columns: `epsilon, fool_pct, landed, device_1..5 (counts), frames_decoded, bit_errors, ber,
bits_compared`.
- **fool_pct** = % off the victim · **target_pct** = % on `--target` · **landed** = plurality class.
- **BER**: when `bit_errors == 0`, report the rule-of-three 95% upper bound `< 3/bits_compared`
  (payload = 4000 bits/frame, so `bits_compared` = decoded_frames × 4000), never a bare 0.
- **Crossover** = first ε where the target becomes the plurality class.

