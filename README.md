# AdvRadio — Radio-Fingerprint Adversarial Attacks (Reproducibility Guide)

Over-the-air (OTA) adversarial attacks against a Wi-Fi (802.11 a/g) **device-fingerprinting**
neural network. A victim radio transmits; an **adversary** radio adds a crafted perturbation so the
receiver's classifier mis-identifies the victim as another enrolled device — ideally while the
802.11 link still decodes (BER ≈ 0). The study spans a software upper bound, a single-radio OTA
attack, a realistic two-radio OTA attack, four classifier variations, and five attack-configuration
variations.

This README is the **WiSec 2027 artifact entry point**: it documents the environment, every
experiment we ran, and the exact step-by-step commands to reproduce each result and figure.

> **This repo is code only.** It contains the scripts, the shipped result tables/figures (`docs/`),
> and the GNU Radio flowgraphs. Every figure and table re-renders from the shipped `docs/*.csv` with
> **no download and no hardware** (§5, §8). A small **recompute bundle** (enrollment + models, ~19 GB)
> and the trained models are hosted externally and are needed only to recompute from scratch — see the
> reproduction tiers in §3.

> **Terminology (paper vs. code).** The paper calls the **1,024-sample model input** an **I/Q frame**
> (the sliding-window input, matching our prior work — 1,024 samples here vs. 288 there). In the code and
> this guide, that same unit is a **window** (`WIN = 1024`; see `frame_to_windows` in `exp3_fp_model.py`).
> The code/README **frame** (`FRAME_LEN = 15360`; e.g. "detected frames", `frames_decoded`,
> `ch0_run3_<N>.bin`) is the full 802.11 transmission, which the paper does not name separately and which
> yields **13 windows**. In short: **paper "I/Q frame" = code/README "window"**, and a code/README
> "frame" is one 802.11 transmission (13 windows each).

> **Note on "the paper."** There is **no manuscript PDF in this repo**. The attack results are
> reproduced from the `make_fig*.py` / `make_*.py` scripts, the per-experiment workbook
> `docs/exp3_all_results_byexp.xlsx`, and the result CSVs under `docs/`. Figure numbers
> (Fig. 12/13/14, Table VII) follow the authors' internal numbering in the figure scripts. To add exact
> manuscript cross-references, point to the submission `.tex`/`.pdf` and they can be inserted.

---

## Contents

1. [What this reproduces](#1-what-this-reproduces)
2. [Repository layout](#2-repository-layout)
3. [Datasets & models](#3-datasets--models)
4. [Environment & hardware](#4-environment--hardware)
5. [Quick start — reproduce results without radios](#5-quick-start--reproduce-results-without-radios)
6. [Full capture → attack pipeline (step by step)](#6-full-capture--attack-pipeline-step-by-step)
7. [Experiment catalog (E1–E3, A1–A4, B1–B5)](#7-experiment-catalog)
8. [Regenerating every figure, table, and workbook](#8-regenerating-every-figure-table-and-workbook)
9. [The baseline classifier](#9-the-baseline-classifier)
10. [Reading results](#10-reading-results)
11. [Gotchas](#11-gotchas)
12. [Caveats, superseded artifacts, citation](#12-caveats-superseded-artifacts-citation)

---

## 1. What this reproduces

Two labeling layers run on **one** capture/craft pipeline:

**Capture/craft modalities** — *how* δ reaches the receiver:

| | Modality | What it measures | Radio? | δ scaling |
|---|---|---|---|---|
| **E1** | Digital reachability | Software upper bound: smallest ℓ∞ radius to flip the victim to each class, shortfall Δ, ρ = R/ε_phys, proxy BER | no | — |
| **E2** | Single-channel OTA | δ summed into the victim replay, radiated through the victim's own chain (ch1 silent) | yes | SCALE ≈ 13.06 |
| **E3** | Two-channel OTA | Victim replay on ch0, δ on ch1 from a **separate MIMO-locked adversary antenna** (the realistic attack) | yes | SCALE = 1 |

**Experiment variations** — *what* is varied (each has digital and/or OTA arms that run inside E1/E2/E3):

| | Varies | One-line finding | Main figure/table |
|---|---|---|---|
| **A1** | Architecture: real CNN1D vs complex CVNN | Fragility is task-, not architecture-, bound (CNN1D reproduces CVNN reachability, corr 0.987) | feeds Fig. 14 |
| **A2** | Cross-model transferability | δ crafted on CVNN transfers to CNN1D and physics SVM/RF (τ ≈ 0.90, near-black-box) | transfer figure |
| **A3** | Physics SVM/RF (feature-explicit) | Partially resists (~2× budget) but is not a defense | A3 robustness figure |
| **A4** | Representation: time vs frequency CVNN | Fragility is representation-invariant; geometry is representation-dependent (corr 0.324) | A4 ε50 figure |
| **B1** | Adversary power / received PSR sweep | Two-channel fooling, target rate, BER vs adversary gain; standoff ratios | **Table VII**, B1 power-sweep figures |
| **B2** | Matched (per-frame) vs universal (one waveform) δ | A realizable universal δ needs larger ε than per-frame matched δ | B2 matched-vs-unmatched |
| **B3** | Frame-level vs per-window objective (ℓ∞) | Under ℓ∞ the frame objective gives no benefit | B3 ℓ∞ bars |
| **B4** | — | **Dropped** (not built; not in workbook/report) | — |
| **B5** | L2 vs ℓ∞ × objective | Under L2 the frame objective concentrates energy and wins; ordering survives OTA | B5 L2-vs-ℓ∞ |

The two-channel white-box OTA sweep (`docs/a1cvnn_*_ota_ber.csv`) double-serves as both the **A1**
OTA arm and the **E3** source for Fig. 12–14 — one capture, two roles.

---

## 2. Repository layout

```
AdvRadio/                             # this repo — WiSec 2027 artifact (code only)
├── README.md                         # <- this file
├── requirements.txt                  # Python deps (torch, numpy, scipy, sklearn, matplotlib)
├── LICENSE
│
├── exp3_fp_model.py                  # CVNN + window preprocessing — single source of truth
├── exp3_cnn1d.py exp3_freqcvnn.py    # A1 (real CNN1D) and A4 (freq CVNN) classifiers
│                                       exp3_transfer_cnn1d_cvnn.py (A2), exp3_a3_feature_model.py (A3)
│
│   ── canonical pipeline scripts (see §6) ──
├── exp3_cvnn.py                      # train / fine-tune the CVNN
├── exp3_crossday_eval.py            # per-device accuracy + confusion
├── exp3_map_detected_ids.py         # recover detected frame ids
├── build_ch0_matched.py            # build the victim replay (ch0)
├── exp3_make_ota_perturbations.py  # craft δ banks (received-domain, masked)
├── run_2ch.py score_2ch.py  # OTA transmit primitive + scorer (classify + BER)
├── exp3_rx_capture.py               # OTA receiver capture (called by run_2ch.py)
├── exp3_ber_eval.py                 # true 802.11-decode BER of a capture
├── exp3_reachability.py        # E1 digital reachability
├── build_ch0_sc.py build_ch1.py  # internal: E2/E3 waveform builders
├── exp3_b5_make_ota_banks.py        # B5 L2/ℓ∞ bank builder
├── campaign.sh run_sc.sh run_2ch.sh # E2/E3 campaign drivers
├── exp3_attack_matrix.py exp3_make_perturbation.py exp3_adversary_tx.py  # helpers imported by the above
│
├── make_fig*.py  make_a*.py  make_b*.py  make_*.py   # figure generators (see §8)
├── make_results_workbook_byexp.py   # -> docs/exp3_all_results_byexp.xlsx (master index)
│
├── flowgraphs/                       # GNU Radio 3.10 flowgraphs + enrollment wrappers (OTA only, §6.1)
│   ├── lab_exp3_enroll_calib.py      #   enrollment capture driver
│   ├── enroll_d{1..5}_runs.sh        #   per-device enrollment wrappers
│   ├── wifi_tx_updated.py            #   legit transmitter (FRID-stamped frames)
│   ├── wifi_rx.py                    #   receiver (decode + log frame ids)
│   └── wifi_adversary_tx.py          #   2-channel MIMO adversary TX
│
└── docs/                             # ** all result CSVs, figures, workbook, and write-ups **
    ├── exp3_all_results_byexp.xlsx   #   master per-experiment workbook (A1-ML … B5-W)
    ├── a1cvnn_<method>_ota_ber.csv   #   E3/A1 two-channel OTA (CVNN) — Fig.12/13/14 source
    ├── a1_<method>_ota_ber.csv       #   E3/A1 two-channel OTA (CNN1D)
    ├── b1_cvnn_foolber_*.csv    #   B1 power sweep (untargeted); _tgt_d2_ = targeted
    ├── b2_*.csv b3_*.csv b5_*.csv    #   B2/B3/B5 result data
    ├── exp3_fig12_ber_ieee.{pdf,svg,png}  …  #   rendered paper figures
    └── *.md                          #   result write-ups (B1_mean_results.md, VII-B1_…, etc.)
```

**Hosted externally, not in git (§3):** the enrollment + models **recompute bundle** (~19 GB) and the
raw OTA capture sessions (provenance). Download the bundle only to recompute from scratch (Tier 2);
every figure/table regenerates from the shipped `docs/` with nothing downloaded (Tier 1).

**Required separately for OTA (§4):** the `gr-foo` and `gr-ieee802-11` GNU Radio 3.10 OOT modules, built
and installed. They are **not** vendored here; the `flowgraphs/` scripts import the installed
`ieee802_11` module.

**Navigating.** The canonical reproducible path is the §6 pipeline; the single authoritative data
index is `docs/exp3_all_results_byexp.xlsx`. Superseded/dated snapshots from the development repo were
**not** carried into this artifact (§12).

---

## 3. Datasets & models

Reproduction has tiers; **most reviewers need nothing beyond this repo.**

| Tier | Reproduces | Data to download | Size |
|---|---|---|---|
| **1 — Results** | every figure, table, and the workbook | **nothing** — the shipped `docs/*.csv` in this repo (verified: 36/36 `make_*.py` regenerate) | **0** |
| **2 — Recompute** | E1 reachability, the A-series, retraining, feature-domain crafting (no radios) | the **minimal bundle** below | **~19 GB** |
| **3 — Re-score raw OTA** | OTA fool%/BER from the raw `rx_*.bin` recordings (no radios) | the full attack capture dirs | ~90 GB |
| **4 — Re-run OTA** | the whole pipeline | everything **+ the radios (§4)** | — |

So: **clone the repo and run the scripts to reproduce every reported result.** Download data only to
recompute from scratch.

### Minimal bundle (Tier 2) — OneDrive

> **Download:** `<<< PASTE ONEDRIVE LINK HERE >>>`  *(also cited in the paper)*

```
captures/
├── enrollment/           #   per-device enrollment captures (§3.1)   ~19 GB
└── attack_assets/        #   victim replay + δ waveform (ch0_run3_107.bin, ch1_e04_s1.bin)  ~0.3 GB
models/                   #   fingerprint_{cvnn,cnn1d,freqcvnn}.pt (§3.3)  ~4 MB
```

The raw OTA capture sessions (Tier 3, ~90 GB of `rx_*.bin`) are provenance for byte-level audit and are
**available on request / on the dataset DOI**, not in the minimal bundle.

**Setting the location.** Every script reads the captures root from the `CAPTURES_ROOT` environment
variable, defaulting to `~/captures`. Point it at wherever you unpacked the bundle:
```bash
export CAPTURES_ROOT=/path/to/captures
```
Then copy the three models (§3.3) into the repo root (scripts take `--model fingerprint_cvnn.pt`).

### 3.1 Enrollment dataset — `${CAPTURES_ROOT}/enrollment/`

- **Format:** complex64 interleaved I/Q at **Fs = 5 MHz**, ~30 s per run.
- **Classes:** 5 fingerprint classes **`device_1 … device_5`** (`NUM_CLASSES = 5` in `exp3_fp_model.py`;
  `device_5` = the adversary radio, captured as a victim class).
- **Layout:** `${CAPTURES_ROOT}/enrollment/device_<N>/` with
  `clean_run_{1,2,3}.bin` (~1.24–1.28 GB each) plus sidecars
  `frame_run_*.bin`, `frame_index_run_*.csv`, `rx_frames_run_*.jsonl`, `ber_frames_run_*.jsonl`,
  `tx_ids_run_*.csv`.
- **Split (leakage-safe):** per device, **runs 1,2 = train, run 3 = held-out validation**
  (no window leakage). `${CAPTURES_ROOT}/validation/` holds separate BER/decode validation captures,
  and `${CAPTURES_ROOT}/train/` is the 5-class symlink tree the trainer builds (§6.2).
- Windowing: `WIN = 1024` complex samples tiled **non-overlapping** (stride = `WIN`) over each frame's
  payload region `[PRE_ROLL : PRE_ROLL+ACTIVE] = [500 : 13812]` in `exp3_fp_model.py` — i.e. skipping the
  preamble and SIGNAL field (samples 100–500). Each window is the paper's **I/Q frame** (the model
  input), power-normalized to unit average power.
  - **Why each 802.11 frame gives 13 I/Q frames:** the windowed payload span is `ACTIVE = 13312`
    samples, and with non-overlapping 1,024-sample windows that is exactly `13312 / 1024 = 13` windows.
    `ACTIVE` is set to `13 × 1024 = 13312` — the largest whole multiple of `WIN` that fits in the ~14,160
    available payload samples (500…14,660); any leftover payload shorter than one window is dropped so
    every window is full-length. So one 802.11 frame (15,360 samples) gives **13** I/Q frames (1,024
    samples each), and the counts follow: e.g. 6,731 Normal 802.11 frames × 13 = 87,503 I/Q frames; the
    107-frame OTA replay × 13 = 1,391 I/Q frames.

### 3.2 Attack data — `${CAPTURES_ROOT}/attack_*/` (Tier 3, provenance)

These are the **raw OTA capture sessions** — needed only to re-score the over-the-air results from the
raw `rx_*.bin` recordings (Tier 3) or re-transmit them (Tier 4). The **derived result CSVs the figures
actually read ship in `docs/`** (Tier 1), so you do *not* need these dirs to regenerate any figure. The
only part in the minimal bundle is `attack_assets/`'s small TX waveforms (`ch0_run3_107.bin`,
`ch1_e04_s1.bin`). The role-based dirs:

| Directory | Role | Key contents |
|---|---|---|
| `attack_assets/` | OTA victim/adversary waveforms (E2/E3) | `ch0_run3_107.bin` (victim replay), `ch1_e04_s1.bin` (adversary δ) |
| `attack_b1/` | B1 adversary-power sweep | `b1_cvnn_foolber_<method>.csv`, `b1_cvnn_tgt_d2_foolber_<method>.csv` |
| `attack_a1_transfer/` | A1 cross-model transfer | per-model white-box score CSVs |
| `attack_a1_gain/` | A1 gain sweep | `g16_<method>.csv` |
| `attack_enroll/` | attack re-enrollment baseline | `attack/ch0.bin`, `ota_perturbations/` (first δ bank) |

Filename legend: `ch0*.bin` = victim/legit replay · `ch1*.bin`, `pert_*/delta_*.npy` = adversary
waveform / δ bank · `rx*.bin(.rxlog)` = OTA capture / decode · `score_*.csv`, `blk_*.csv` = per-frame /
per-block fingerprint scores · `ber_*.csv` = decode BER · `*.sh` = run driver.

### 3.3 Trained models — `*.pt` (OneDrive `models/`)

> **Models are optional.** Every figure, table, and the workbook regenerate from the shipped
> `docs/*.csv` with **no model** (verified: all `make_*.py` run with no `.pt` present). The `.pt` are
> needed only to **recompute from the classifier** — the E1 quick-start (§5C), and OTA scoring/crafting
> (§6–§7). Even then they're a convenience: §6.2 retrains an equivalent model from the raw enrollment;
> the shipped checkpoints just give bit-exact reproduction and skip retraining.

Three checkpoints, one per architecture, all scoring `WIN=1024` complex windows over 5 classes
(`device_1…5`). Download them into the repo root only if you need the recompute path:

| Model | Size | Arch | Role |
|---|---|---|---|
| **fingerprint_cvnn.pt** | 1.72 M | CVNN | main classifier (E1–E3, B-series) |
| fingerprint_cnn1d.pt | 490 K | CNN1D | A1 real-CNN baseline |
| fingerprint_freqcvnn.pt | 1.72 M | FreqCVNN | A4 frequency-domain variant |

Architectures: **CVNN** = time-domain complex CNN (`exp3_fp_model.py`), **FreqCVNN** = FFT-first
(`exp3_freqcvnn.py`), **CNN1D** = real 1-D CNN baseline (`exp3_cnn1d.py`).

---

## 4. Environment & hardware

### Software

```bash
pip install -r requirements.txt     # numpy, scipy, matplotlib, scikit-learn, torch
```
For GPU training install the CUDA build of `torch` from https://pytorch.org/. **E1 (digital), all
scoring, and all figure regeneration need only Python + PyTorch** — no GNU Radio, no hardware.

**Capture / OTA only (E2/E3 and enrollment):** GNU Radio 3.10 + UHD, with the `gr-foo` and
`gr-ieee802-11` OOT modules **built and installed** (they are not vendored in this repo — obtain
`gr-ieee802-11` from https://github.com/98360679/gr-ieee802-11 and `gr-foo` from its upstream, both on
the `maint-3.10` branch):
```bash
cd gr-foo && mkdir build && cd build && cmake .. && make && sudo make install && sudo ldconfig
cd ../../gr-ieee802-11 && mkdir build && cd build && cmake .. && make && sudo make install && sudo ldconfig
```
(`gr-foo` first.) Python modules `gnuradio`, `ieee802_11`, `pmt` come from this system install, not pip.
The `flowgraphs/` scripts import `ieee802_11`.

### Hardware rig (E2/E3)

- **Victim** — USRP2/N-series `192.168.10.3`, antenna **J1**, TX gain **0.3689**, enrolled as the
  victim class (`device_5` in this rig).
- **Adversary** — USRP2/N-series `192.168.10.4`, antenna **J1**, **MIMO-cabled to the victim**; the
  only attacker.
- **Other enrolled devices** — B200mini/B205mini (`device_1 3263606`, `device_2 32593C4`,
  `device_3 3259376`, `device_4 3259368`), swapped one at a time into TX/RX; these are the
  classifier's other legitimate classes = the impersonation *targets*, not attackers.
- **Receiver** — b205mini, **`rx_gain = 0.93` everywhere** (gotcha #1).

---

## 5. Quick start — reproduce results without radios

Everything below needs only Python + the shipped CSVs in `docs/` (no hardware, no data download).

> **No models needed for the headline reproduction.** Parts **A** (all figures) and **B** (the
> workbook) regenerate every result from the shipped `docs/*.csv` with **no `.pt` and no data download**
> — verified across all `make_*.py`. Only part **C** (E1 reachability) loads a model and the enrollment
> windows, and that model is optional (see §3.3 — you can retrain via §6.2 instead).

**A. Re-render the main OTA figures** (Fig. 12 BER, Fig. 13 rejections, Fig. 14 fooling) from the
two-channel SCALE=1 data:
```bash
cd /path/to/AdvRadio
python3 make_fig1213_exp3_ieee.py     # -> docs/exp3_fig12_ber_ieee.* , docs/exp3_fig13_rejections_ieee.*
python3 make_fig14_fooling_ieee.py    # -> docs/exp3_fig14_fooling_ieee.*
```

**B. Rebuild the master results workbook** (one sheet per A1–A4, B1–B5):
```bash
python3 make_results_workbook_byexp.py   # -> docs/exp3_all_results_byexp.xlsx
```

**C. Run E1 digital reachability** (needs a model from OneDrive `models/` + the enrollment windows, no
radio):
```bash
python3 exp3_reachability.py \
  --model fingerprint_cvnn.pt \
  --base  $CAPTURES_ROOT/enrollment \
  --out   reachability_quickstart.csv
```

Full from-raw reproduction (capture → train → craft → OTA → score) is §6; per-experiment
reproduction is §7.

---

## 6. Full capture → attack pipeline (step by step)

Values in `<angle brackets>` are yours to fill: `<N>` number of
detected victim frames, `<scale>` from §6.5, `<prev>` checkpoint to fine-tune from, `<victim>` the
victim label (e.g. `device_5`), `<V>` its 1-based class index (e.g. `5`), `<target>` the class to
impersonate, `<K>` the target's 1-based index.

| Stage | Script | Radio | § |
|---|---|---|---|
| 1. Collect enrollment | `flowgraphs/enroll_d{1..5}_runs.sh` → `flowgraphs/lab_exp3_enroll_calib.py` | yes | 6.1 |
| 2. Train model | `exp3_cvnn.py` | no | 6.2 |
| 3. Validate | `exp3_crossday_eval.py` | no | 6.3 |
| 4. Build victim replay | `exp3_map_detected_ids.py`, `build_ch0_matched.py` | no | 6.4 |
| 5. Silent-adversary gate | `run_2ch.py`, `score_2ch.py` | yes | 6.5 |
| 6. Craft δ banks | `exp3_make_ota_perturbations.py` | no | 6.6 |
| E1 Reachability | `exp3_reachability.py` | no | 7 |
| E2 Single-channel | `run_sc.sh` / `campaign.sh sc_*` | yes | 7 |
| E3 Two-channel | `run_2ch.sh` / `campaign.sh ch2_*` | yes | 7 |

### 6.1 Collect enrollment — `flowgraphs/lab_exp3_enroll_calib.py`
Bring a TX radio up once, calibrate its gain to a target RX level, capture 3 clean runs/device via the
per-device wrappers (one radio init per run — the pattern that works).
Produces `$CAPTURES_ROOT/enrollment/device_N/clean_run_{1,2,3}.bin` (+ sidecars).
Key args: `--device-id N` (2–6), `--run-id`, `--tx-serial` (serial= for B-series, IP for the usrp2
victim), `--tx-antenna` (`TX/RX` or `J1`), `--target-db` (use **−30**), `--fix-gain` (victim only:
**0.3689**), `--g0`, `--date` (new every session), `--tree enrollment|validation`.
```bash
cd /path/to/AdvRadio/flowgraphs
# edit each wrapper: attacker devices --target-db -30 --g0 <start>, drop --fix-gain; victim keeps --fix-gain 0.3689
bash enroll_d1_runs.sh    # then swap radio → enroll_d2_runs.sh → d3 → d4 → d5
```
Each run ≈ 1 GB, 100+ frames, `bad_packet=no`, within tolerance of −30 dB.

### 6.2 Train / fine-tune — `exp3_cvnn.py`
**Refit every session** — a reused prior-day model fails on a re-leveled enrollment. Needs a `_5dev`
symlink tree (`device_N → device_(N+1)`, **clean_run only**). Produces `fingerprint_<tag>.pt`.
Args: `--root`, `--val-run` (3), `--epochs` (use **80**), `--init <ckpt>` (fine-tune; omit for scratch),
`--tag`, `--lr`, `--cfo-correct`, `--rebuild`.
```bash
cd /path/to/AdvRadio
ENR=$CAPTURES_ROOT/enrollment
TRN=$CAPTURES_ROOT/train; rm -rf "$TRN"
for n in 1 2 3 4 5; do mkdir -p "$TRN/device_$n"
  for r in 1 2 3; do ln -s "$ENR/device_$((n+1))/clean_run_$r.bin" "$TRN/device_$n/clean_run_$r.bin"; done; done
python3 exp3_cvnn.py --init fingerprint_cvnn.pt --val-run 3 --epochs 80 --root "$TRN" --tag cvnn
#   -> fingerprint_cvnn.pt
```

### 6.3 Validate — `exp3_crossday_eval.py`
Per-device accuracy + confusion. Want **victim = 100%** and the target class strong.
```bash
python3 exp3_crossday_eval.py --model fingerprint_cvnn.pt --test "$ENR"
```

### 6.4 Build the victim replay `ch0`
A clean baseband replay reproducing the victim frame-for-frame (base for crafting *and* the E2/E3 ch0).
```bash
A=$CAPTURES_ROOT/attack_assets; mkdir -p "$A"; VIC="$ENR/<victim>"
python3 exp3_map_detected_ids.py --capture $VIC/clean_run_3.bin --rx-frames $VIC/rx_frames_run_3.jsonl \
    --out $A/detected_ids_run3.csv --thr-mult 2.0
python3 build_ch0_matched.py --dev-dir $VIC --run 3 --ids $A/detected_ids_run3.csv \
    --period 65661 --out $A/ch0_run3_<N>.bin        # N = number of detected frames
```
`--thr-mult 2.0` **must match** §6.6 so δ[k] ↔ ch0 frame k.

### 6.5 Silent-adversary gate — `run_2ch.py` + `score_2ch.py`
Prove the unperturbed victim replay reads **the victim class** through today's rig before trusting any
campaign. Transmit ch0 + a zeros ch1 and score.
```bash
python3 - <<'PY'   # zeros ch1 + 1-row block map
import numpy as np
A="$CAPTURES_ROOT/attack_assets"; ch0=np.fromfile(f"{A}/ch0_run3_<N>.bin",dtype=np.complex64); n=len(ch0)
np.zeros(n,dtype=np.complex64).tofile(f"{A}/ch1_zeros.bin")
open(f"{A}/blk_control.csv","w").write("block,method,epsilon,phase_deg,start_sample,n_samples,score_from,score_to\n0,none,0.0,0.0,0,%d,0,%d\n"%(n,n))
PY
python3 run_2ch.py --ch0 $A/ch0_run3_<N>.bin --ch1 $A/ch1_zeros.bin --out $A/rx_control.bin \
    --secs 30 --tx-gain 0.493 --adv-gain 0.493 --rx-gain 0.93
python3 score_2ch.py --capture $A/rx_control.bin --blocks $A/blk_control.csv --ch0 $A/ch0_run3_<N>.bin \
    --model fingerprint_cvnn.pt --victim <victim> --select fool --ber --guard-frac 0 --out $A/score_control.csv
```
**Pass = landed the victim class, fool 0%, BER 0.** Else fix the rig (usually rx-gain). Then compute
**SCALE** = `ch0_active_rms / received_active_rms` (δ is crafted at received amplitude; the build
re-scales it to TX level) — see the snippet in the appendix of the per-session runlog; use it as
`<scale>` for E2 (two-channel E3 uses `SCALE=1`).

### 6.6 Craft δ banks — `exp3_make_ota_perturbations.py`
Optimize δ against the **received** victim, payload-only, MAC header masked. Produces a bank dir of
compact `.npy` deltas `(N, 13312)` + `manifest.csv`, one per method×ε.
Args: `--model`, `--ch0` (= the **received** capture), `--from-capture`, `--thr-mult 2.0` (match §6.4),
`--nper <N>`, `--mask-lead 780`, `--tgt-dst <K>` (target 1-based index) **or** `--untargeted`,
`--victim-class <V>`, `--methods` (default 6: `fgsm bim mim pgd_a0.01 pgd_a0.05 pgd_a0.10`), `--epsilons`.
```bash
RX=$ENR/<victim>/clean_run_3.bin
COMMON="--model fingerprint_cvnn.pt --ch0 $RX --from-capture --thr-mult 2.0 --nper <N> --mask-lead 780 --victim-class <V>"
MASKED="0.01 0.012 0.014 0.016 0.018 0.02 0.022 0.025 0.03 0.035 0.04 0.045 0.05 0.06 0.08 0.1"          # E3, 16 pts
FINE=$(python3 -c "print(' '.join(f'{round(0.1+0.01*i,2):g}' for i in range(41)))")                      # E2, 41 pts 0.10-0.50
python3 exp3_make_ota_perturbations.py $COMMON --tgt-dst <K> --epsilons $MASKED --out-dir $A/pert_tgt_<target>_masked
python3 exp3_make_ota_perturbations.py $COMMON --tgt-dst <K> --epsilons $FINE   --out-dir $A/pert_tgt_<target>_sc
python3 exp3_make_ota_perturbations.py $COMMON --untargeted  --epsilons $MASKED --out-dir $A/pert_unt_masked
python3 exp3_make_ota_perturbations.py $COMMON --untargeted  --epsilons $FINE   --out-dir $A/pert_unt_sc
```

---

## 7. Experiment catalog

Each experiment below lists its one-line goal, how to reproduce it, the result data, and the figure it
produces. Shared setup (model + enrollment + δ banks) comes from §6. The master data index for every
experiment is `docs/exp3_all_results_byexp.xlsx`.

### E1 — Digital reachability (no radio)
```bash
python3 exp3_reachability.py --model fingerprint_cvnn.pt --base "$ENR" --out $A/reachability.csv
```
Data: `reachability_matrix.csv`, `reachability_shortfall_curves.{npz,csv}`,
`exp3_eps_phys_empirical.{csv,npz}`, `reachability_cvnn.csv`. Figures: `make_e1_diagram.py` → `docs/e1_perturbation_scoring_ber.*`,
`make_digital_vs_ota.py` → `docs/digital_vs_ota.*`. Produces the reachability table (`tab:reach`).

### E2 — Single-channel OTA — `campaign.sh sc_tgt_d1 | sc_unt`
δ summed into the victim replay (ch1 = zeros), fine grid 0.10–0.50. `campaign.sh` reads **exported**
`A` and `CH0`.
```bash
export A=$CAPTURES_ROOT/attack_assets; export CH0=$A/ch0_run3_<N>.bin
FINE=$(python3 -c "print(','.join(f'{round(0.1+0.01*i,2):g}' for i in range(41)))")
export MODEL=fingerprint_cvnn.pt KEEP_CAPTURES=0 GUARD_FRAC=0 BER_PASSES=1 \
       PASSES_SC=2 SECS_SC=130 SCALE=<scale> METHODS="fgsm bim mim pgd_a0.01 pgd_a0.05 pgd_a0.10" EPS_GRID="$FINE"
RUN=1 TGT=<target> PERT_D1=$A/pert_tgt_<target>_sc ./campaign.sh sc_tgt_d1
RUN=1 PERT_UNT=$A/pert_unt_sc                      ./campaign.sh sc_unt
./run_sc.sh                                        # or both arms in one shot
```
Output `$A/score_sc_{tgt_d1,unt}_r<RUN>_<method>.csv`. Figure: `make_sc_vs_2ch_plot.py` →
`docs/b1_sc_vs_2ch.*`; data `docs/b1_single_channel_*_pgd05.csv`.

### E3 — Two-channel OTA — `campaign.sh ch2_tgt_d1 | ch2_unt`
Victim replay on ch0, δ on ch1, MIMO-locked, masked 16-pt grid, **SCALE=1**. The Qt GUI shows per
method. Reuses the exported `A`/`CH0`.
```bash
export MODEL=fingerprint_cvnn.pt KEEP_CAPTURES=0 GUARD_FRAC=0 BER_PASSES=1 SECS_2CH=150 \
       SCALE=1 METHODS="fgsm bim mim pgd_a0.01 pgd_a0.05 pgd_a0.10" \
       EPS_GRID="0.01,0.012,0.014,0.016,0.018,0.02,0.022,0.025,0.03,0.035,0.04,0.045,0.05,0.06,0.08,0.1"
RUN=r1 TGT=<target> PERT_D1=$A/pert_tgt_<target>_masked ./campaign.sh ch2_tgt_d1
RUN=r1 PERT_UNT=$A/pert_unt_masked                      ./campaign.sh ch2_unt
./run_2ch.sh                                            # or both arms in one shot
```
Output `$A/score_ch2_{tgt_d1,unt}_r<RUN>_<method>.csv`. **This is the source for Fig. 12/13/14**:
white-box CVNN results land in `docs/a1cvnn_<method>_ota_ber.csv`, CNN1D in `docs/a1_<method>_ota_ber.csv`.
Figures: `make_fig1213_exp3_ieee.py` (Fig.12 BER, Fig.13 rejections), `make_fig14_fooling_ieee.py`
(Fig.14 fooling), `make_fig12a_fullrange_ieee.py` / `make_exp3_ber_fullrange_ieee.py` (full-range BER).

### A1 — CNN1D vs CVNN (architecture axis)
Does complex architecture cause the fragility? Train the CNN1D baseline (`exp3_cnn1d.py`), run its
reachability, compare to CVNN.
Data: `reachability_matrix_cnn1d.csv`, `exp3_A1_cnn1d_compare.npz`, `docs/a1_*_ota_ber.csv`,
`docs/a1cvnn_*_ota_ber.csv`. Figures: `make_a1_compare.py` → `docs/a1_cnn1d_vs_cvnn.*`,
`make_a1_ota.py` → `docs/a1_cnn1d_ota.*`, `make_a1_gain.py`, `make_cnn1d_fooling_plot.py`.

### A2 — Cross-model transferability
δ crafted on CVNN applied to CNN1D and the physics SVM/RF. `exp3_transfer_cnn1d_cvnn.py`.
Data: `exp3_transfer_cnn1d_cvnn.{csv,npz}`, `exp3_ota_multimodel.csv`. Figures: `make_transfer_plot.py`
→ `docs/transfer_fooling_vs_eps.*`, `make_ota_multimodel_plot.py` → `docs/ota_multimodel_A1A4.*`.
(Workbook sheets `A2-ML`, `A2-transfer-raw`.)

### A3 — Physics feature model (SVM/RF)
Feature-explicit classifier on explicit impairment features. `exp3_a3_feature_model.py`.
Data: `exp3_a3_feature_model.{csv,npz}`. Figure: `make_a3_plot.py` → `docs/a3_feature_model_robustness.*`.

### A4 — Frequency-domain CVNN
FFT-first classifier (`exp3_freqcvnn.py`); representation-invariance of fragility.
Data: `reachability_matrix_freqcvnn.csv`, `exp3_A4_freqcvnn_compare.npz`. Figure: `make_a4_plot.py`
→ `docs/a4_time_vs_freq_eps50.*`.

### B1 — Adversary power / PSR sweep → **Table VII**
Two-channel fooling, target rate, and BER vs adversary gain and received PSR (ε=0.01, **mean over 16
blind phases**); standoff ratios. Runs on `$CAPTURES_ROOT/attack_b1`.
Data: `docs/b1_cvnn_foolber_<method>.csv` (untargeted), `docs/b1_cvnn_tgt_d2_foolber_<method>.csv`
(targeted d3), `docs/b1_2ch_corrected_scale1.csv`; tables in `docs/B1_mean_results.md`,
`docs/VII-B1_and_standoff_update.md`. Figures: `make_b1_combined_cvnn.py`,
`make_b1_fool_ber_mean_cvnn.py`, `make_b1_mean_4panel.py` → `docs/b1_mean_4panel.png`,
`make_b1_maxphase_plateau_cvnn.py`. **Use the canonical data** (the superseded earlier
runs were re-baselined — see §12).

### B2 — Matched vs universal δ
Per-frame matched δ vs one universal waveform reused on every frame (the realizable adversary).
Data: `docs/b2_matched_vs_unmatched.csv`; notes `docs/B2_answers_from_lab_machine.md`.
Figures: `make_b2_matched_vs_unmatched.py` → `docs/b2_matched_vs_unmatched.*`, `make_b2_ber.py`,
`make_b2_eps50.py`.

### B3 — Frame-level vs per-window objective (ℓ∞)
Data: `docs/b3_frame_vs_window.csv` (digital), `docs/b3_ota_linf.csv` (OTA). Figures:
`make_b3_ota.py` → `docs/b3_ota_linf.*`, `make_b3_objective_effect.py` → `docs/b3_objective_effect.*`.

### B4 — dropped
Not built; not in the workbook or report.

### B5 — L2 vs ℓ∞ × objective
Banks built by `exp3_b5_make_ota_banks.py`. Data: `docs/b5_l2_vs_linf.csv` (digital),
`docs/b5_ota_energy.csv`, `docs/b5_ota_allmethods.csv`. Figures: `make_b5_l2_vs_linf.py`,
`make_b5_ota.py`, `make_b5_ota_allmethods.py` → `docs/b5_*.*`.

---

## 8. Regenerating every figure, table, and workbook

All figure scripts read the shipped `docs/*.csv` (no hardware). Each writes `.png` + `.svg` (several
also `.pdf`) under `docs/`.

**Master workbook (run this first — the data index):**
`make_results_workbook_byexp.py` → `docs/exp3_all_results_byexp.xlsx` (sheets A1-ML, A2-ML,
A2-transfer-raw, A3-ML, A4-ML, B1-W, B2-W, B3-W, B5-W).

**E3 / paper figures:** `make_fig1213_exp3_ieee.py` (Fig.12, Fig.13) · `make_fig14_fooling_ieee.py`
(Fig.14) · `make_fig12a_fullrange_ieee.py`, `make_exp3_ber_fullrange_ieee.py` (full-range BER) ·
`make_fig12_exp3_ber_scale1.py`, `make_fig13_exp3_rejections_scale1.py` (SCALE=1 variants).

**E1/E2:** `make_e1_diagram.py`, `make_digital_vs_ota.py`, `make_sc_vs_2ch_plot.py`.

**A-series:** `make_a1_ota.py`, `make_a1_compare.py`, `make_a1_gain.py`, `make_cnn1d_fooling_plot.py`,
`make_transfer_plot.py`, `make_ota_multimodel_plot.py`, `make_a3_plot.py`, `make_a4_plot.py`.

**B-series:** `make_b1_combined_cvnn.py`, `make_b1_fool_ber{,_mean,_tgt,_tgt_mean}_cvnn.py`,
`make_b1_mean_4panel.py`, `make_b1_maxphase_plateau{,_tgt}_cvnn.py`, `make_b1_plot.py`, `make_pwr_fool_ber.py` ·
`make_b2_matched_vs_unmatched.py`, `make_b2_ber.py`, `make_b2_eps50.py` ·
`make_b3_ota.py`, `make_b3_objective_effect.py` ·
`make_b5_l2_vs_linf.py`, `make_b5_ota.py`, `make_b5_ota_allmethods.py`.

Regenerate everything:
```bash
for f in make_fig1213_exp3_ieee make_fig14_fooling_ieee make_fig12a_fullrange_ieee \
         make_e1_diagram make_digital_vs_ota make_sc_vs_2ch_plot \
         make_a1_compare make_a1_ota make_transfer_plot make_ota_multimodel_plot make_a3_plot make_a4_plot \
         make_b1_combined_cvnn make_b1_mean_4panel make_b2_matched_vs_unmatched make_b2_ber \
         make_b3_ota make_b3_objective_effect make_b5_l2_vs_linf make_b5_ota_allmethods \
         make_results_workbook_byexp; do
  echo "== $f =="; python3 "$f.py" || echo "  (check inputs for $f)"
done
```

---

## 9. The baseline classifier

The attacks target a **victim classifier** — a device-fingerprinting study of its own (4 devices ×
2 days × {time, freq} × {CNN1D, CVNN}; energy-detected 512-sample windows, power-normalized,
leakage-safe split train-on-runs-1–2 / test-on-run-3; best model is the frequency-domain CNN1D at
99.3% Day1, 95.4% Day2 frame-level). That baseline study (`radio_fingerprint/`, with `build_dataset.py`,
`train.py`, `plots_eda.py`, and `REPORT.pdf`) is a **separate codebase and is not part of this
artifact**; it is provided alongside the dataset bundle on OneDrive. This repo reproduces the **attack**
results against that classifier.

---

## 10. Reading results

Score-CSV columns: `epsilon, fool_pct, landed, device_1..5 (counts), frames_decoded, bit_errors, ber,
bits_compared`.
- **fool_pct** = % off the victim · **target_pct** = % on `--target` · **landed** = plurality class.
- **BER**: when `bit_errors == 0`, report the rule-of-three 95% upper bound `< 3/bits_compared`
  (payload = 4000 bits/frame, so `bits_compared` = decoded_frames × 4000), never a bare 0.
- **Crossover** = first ε where the target becomes the plurality class.

---

## 11. Gotchas (each cost real rig time)

1. **`rx-gain 0.93` everywhere** — a mismatch makes the *unperturbed* victim misclassify and every
   fool% a baseline artifact. The gate (§6.5) catches it in 60 s.
2. **Refit the model each session** (§6.2) — reused models fail on re-leveled enrollments.
3. **`--tgt-dst` = the target's 1-based class index** (e.g. `device_1` = 1); the bank name, `TGT`, and
   `--tgt-dst` must all refer to the same target class.
4. **Never edit a running bash script** (bash re-reads by byte offset) — edit via an atomic `mv`.
5. **Disk**: captures are ~6 GB (2-ch) / ~5 GB (SC); keep `KEEP_CAPTURES=0`; merge score CSVs by
   column name, not position.
6. **Don't interrupt a live OTA run**; the MIMO Qt GUI must stay visible.
7. **Two-channel is SCALE=1**; only single-channel (E2) applies the received→TX SCALE.

---

## 12. Caveats, superseded artifacts, citation

This artifact carries **only the canonical reproduction path**. The development repo's superseded /
duplicate files (dated campaign drivers, pre-SCALE=1 crafting and TX engines, one-off orchestration
shells, ~100 dated `*.pt` iterations) were **not** carried over. The three helper modules kept at the
repo root — `exp3_attack_matrix.py`, `exp3_make_perturbation.py`, `exp3_adversary_tx.py` — are retained
because the canonical pipeline scripts import them by name.

- **B1 figures**: a `SCALE=13.06` two-channel scaling bug (fixed to `SCALE=1`) means the superseded B1
  runs were re-baselined to the canonical set. Use the canonical B1 CSVs + `docs/b1_2ch_corrected_scale1.csv`.

**Notes:** `a1cvnn_*_ota_ber.csv` serves both A1 (OTA arm) and E3 (Fig.12–14) — one capture, two roles.
**B4 is intentionally dropped.** The attack figure numbers follow the `make_fig*` script naming, not a
rendered manuscript.

**License:** see `LICENSE`. **Citation:** add the WiSec 2027 BibTeX here once the paper is accepted.
