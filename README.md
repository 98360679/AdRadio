# AdRadio — Radio-Fingerprint Adversarial Attacks (Reproducibility Guide)

This repository accompanies a study of over-the-air (OTA) adversarial attacks on a Wi-Fi (802.11a/g)
**device-fingerprinting** neural network. A victim radio transmits while an **adversary** radio injects
a crafted perturbation, causing the receiver's classifier to misidentify the victim as another enrolled
device while the 802.11 link continues to decode (BER ≈ 0). The study comprises **five experiments**: a
software upper bound (Type I), a single-radio OTA attack (Type II), a two-radio OTA attack (Type III), an
evaluation of robustness across authentication models (Experiment 4), and an evaluation of adversary
realism (Experiment 5).

This document is the **WiSec 2027 artifact entry point**. It describes the software environment, the
experiments, and the commands required to reproduce each result and figure. The repository contains
**source code only**: the analysis scripts, the result CSVs and figures under `docs/`, and the GNU Radio
flowgraphs. Datasets and trained models are hosted externally and are organized into the reproduction
tiers described in **§3**.

> **Reproducing without SDR hardware.** The artifact is designed so Tiers 1–3 need no radios (OTA signals are provided as recordings):
> - **Tier 1** regenerates every figure, table, and the workbook from the data under `docs/`, and runs **Type I reachability** on the shipped CVNN model and window sample — Python only, no download.
> - **Tiers 2–3** recompute the model-variation and validation results, and re-score the OTA fooling-rate and BER metrics by running the classifier on the supplied `.bin` recordings.
>
> Only **Tier 4** (collecting new transmissions) requires the testbed (§5). **For this submission the dataset bundle is not yet released, so only Tier 1 can be run** (§3); the link that unlocks Tiers 2–4 will be added later. Every script runs on CPU; a GPU is optional. The tier table appears in §3.


---

## Contents

1. [What this reproduces](#1-what-this-reproduces)
2. [Quick start — reproduce results without radios](#2-quick-start--reproduce-results-without-radios)
3. [Datasets & models](#3-datasets--models)
4. [Repository layout](#4-repository-layout)
5. [Environment & hardware](#5-environment--hardware)
6. [Capture-to-attack pipeline (step by step)](#6-capture-to-attack-pipeline-step-by-step)
7. [Experiment catalog](#7-experiment-catalog)
8. [Regenerating every figure, table, and workbook](#8-regenerating-every-figure-table-and-workbook)
9. [Reading results](#9-reading-results)

---

## 1. What this reproduces

The five experiments fall into two groups. Experiments 1–3 vary how the perturbation δ reaches the
receiver, in order of increasing physical realism. Experiments 4–5 vary the classifier under attack and
the realism of the adversary.

> **Review scope:** for this submission the dataset is not yet released, so only **Tier 1** is runnable — the
> figures, tables, workbook, and Type I reachability, all from the repository with no download (§3).

| Experiment 1-3 | One-line finding | Radio? | δ scaling |
|---|---|---|---|
| **Exp 1: Type I Baseline** | Receiver-side perturbation of I/Q frames readily fools the classifier, but the BER rises sharply, making the attack easy to detect — a software upper bound | no | — |
| **Exp 2: Type II Adversary** | An adversary perturbing at the transmitter forces misidentification while the receiver decodes without bit errors | yes | SCALE ≈ 11.7 (measured, §6.5) |
| **Exp 3: Type III Adversary** | A separate adversary radio fools the classifier over the air without bit errors, but only within a narrow range; full evasion becomes detectable | yes | SCALE = 1 |

**Experiment 4: Robustness across authentication models.** These four variations test whether the
vulnerability depends on the specific classifier (Exp 4.4 is reported as a table):

| Experiment 4 | One-line finding |
|---|---|
| **Exp 4.1: a real-valued model** | The vulnerability is task-dependent, not architecture-dependent: the CNN1D reproduces the CVNN reachability (corr 0.892) |
| **Exp 4.2: transfer across models** | Perturbations crafted on the CVNN transfer to the CNN1D and to physics-based SVM/RF models (white-box/transferred ε ratio 0.90–0.92; near-black-box) |
| **Exp 4.3: explicit physical features** | Explicit physical features offer partial resistance (~2× budget) but are not a defense |
| **Exp 4.4: spectral representation** | The vulnerability is representation-invariant, though its geometry is representation-dependent (corr 0.534) |

**Experiment 5: Adversary realism.** These two variations test the attack under a more realistic
adversary. Both run through the Experiment 3 two-channel pipeline:

| Experiment 5 | One-line finding |
|---|---|
| **Exp 5.1: perturbation power** | Sweep of adversary power and measured received PSR: two-channel fooling rate, target rate, and BER versus adversary gain, with the distance-ratio reach |
| **Exp 5.2: one perturbation for every frame** | Per-frame matched δ versus a single universal waveform: a realizable universal δ requires a larger ε than a per-frame matched δ |


---

## 2. Quick start — reproduce results without radios

Parts **A**, **B**, and **C** are all **Tier 1** — none needs a download. A and B use data shipped in `docs/`;
C (reachability) runs on the shipped CVNN model and a 300-window/device sample (`docs/reachability_sample.npz`),
and optionally on the full enrollment when the bundle is available (§3). All scripts locate the repository
root automatically and therefore run from any directory.

**0. Get the code & install** (one time). Obtain the artifact — download and unzip the supplied archive (or
`git clone` the repository once de-anonymized) — then change into the extracted folder and install the
Python dependencies:
```bash
cd <extracted folder>               # wherever you unzipped it — the folder containing requirements.txt and README.md
pip install -r requirements.txt     # numpy, scipy, matplotlib, pandas, openpyxl, scikit-learn, torch
```
All commands below are relative to that folder (run them from there).

**A. Re-render the Experiment 3: Type III Adversary figures** from the shipped data:
```bash
python3 plots/plot_exp3_sidebyside.py    # fooling + BER vs ε; figures to plots/output/
python3 plots/plot_exp3_rejection.py     # frame rejections vs ε; figures to plots/output/
```

**B. Rebuild the master results workbook** (one sheet per Exp 4.1–4.4, Exp 5.1–5.2):
```bash
python3 plots/make_results_workbook_byexp.py   # -> docs/exp3_all_results_byexp.xlsx
```

**C. Run Experiment 1: Type I Classifier reachability** (no radio, no download — the CVNN model and a
300-window/device sample ship in the repo):
```bash
python3 experiment_1/exp1_reachability.py --out reachability_quickstart.csv
```
This uses `trained_models/fingerprint_cvnn.pt` and `docs/reachability_sample.npz` — the same 300 windows per
device the shipped `reachability_matrix.csv` was computed from, so it reproduces that matrix exactly. To run
on the full enrollment instead, set `CAPTURES_ROOT` to the bundle (§3) and add `--base $CAPTURES_ROOT/enrollment`.

Full reproduction from raw captures (capture, train, craft, transmit, score) is described in §6;
per-experiment reproduction is described in §7.

---

## 3. Datasets & models

Reproduction is organized into tiers. The repository alone is sufficient to reproduce every reported
result.

| Tier | Reproduces | Data to download | Size |
|---|---|---|---|
| **1 Results** | every figure, table, and the workbook, plus **Experiment 1 (Type I) reachability** | **nothing** — the `docs/*.csv`, the CVNN model, and `docs/reachability_sample.npz`, all shipped in this repo | **0** |
| **2 Recompute** | Experiment 4, retraining, feature-domain crafting, and reachability on the full enrollment (no radios) | the **minimal bundle** below | **~19&nbsp;GB** |
| **3 Rescore&nbsp;raw&nbsp;OTA** | OTA fool%/BER from the raw `rx_*.bin` recordings (no radios) | the full attack capture dirs (on request, §3.2) | ~90&nbsp;GB |
| **4 Re-run OTA** | the whole pipeline | everything **+ the radios (§5)** | — |

> ⚠️ **For this submission, only Tier 1 is runnable.** The dataset bundle is not distributed yet (below), so
> Tiers 2–4 cannot be exercised during review. Everything reported — every figure, table, the workbook, and
> **Type I (reachability)** — reproduces under **Tier 1** from the repository alone, with no download. The
> data link will be added later to unlock Tiers 2–4.

In summary, cloning the repository and running the scripts reproduces every reported result. The
datasets are required only to recompute results from scratch.

### Minimal bundle (Tier 2)

> **Download:** the minimal bundle is not distributed with this submission; a download link will be added
> to the repository later. Tier 1 — every figure, table, and the workbook — needs no download (see §2).
>
> Once the bundle is available, set `CAPTURES_ROOT` to its root (it contains `enrollment/`, `models/`, `attack_assets/`) and copy `models/*.pt` into `trained_models/` (§3.3).

```
bundle/                   #   = $CAPTURES_ROOT
├── enrollment/           #   per-device enrollment captures (§3.1)   ~19 GB
├── models/               #   fingerprint_{cvnn,cnn1d,freqcvnn}.pt  ~4 MB
└── attack_assets/        #   victim replay + δ waveform (ch0_run3_107.bin, ch1_e04_s1.bin)  ~0.3 GB
trained_models/           #   copy models/*.pt here, at the repo root (§3.3)
```

The raw OTA capture sessions (Tier 3; ~90 GB of `rx_*.bin`) serve as provenance for byte-level audit.
They are **available from the authors on request** and are not included in the minimal bundle.

**Capture location.** Each script reads the capture root from the `CAPTURES_ROOT` environment
variable, which defaults to `~/captures`. Set it to the directory into which the bundle was unpacked:
```bash
export CAPTURES_ROOT=/path/to/bundle
```
Then create the models directory and copy the three `.pt` checkpoints (§3.3) into it:
```bash
mkdir -p trained_models && cp "$CAPTURES_ROOT"/models/*.pt trained_models/
```
(scripts accept `--model trained_models/fingerprint_cvnn.pt`). This directory is not part of the repository and must be created.

### 3.1 Enrollment dataset — `${CAPTURES_ROOT}/enrollment/`

- **Format:** complex64 interleaved I/Q at **Fs = 5 MHz**, ~30 s per run.
- **Classes:** 5 fingerprint classes **`device_1 … device_5`** (`NUM_CLASSES = 5` in `fp_model.py`;
  `device_5` = the victim radio, the USRP N-series of §5).
- **Layout:** `${CAPTURES_ROOT}/enrollment/device_<N>/` with
  `clean_run_{1,2,3}.bin` (~1.24–1.28 GB each) plus sidecars
  `frame_run_*.bin`, `frame_index_run_*.csv`, `rx_frames_run_*.jsonl`, `ber_frames_run_*.jsonl`,
  `tx_ids_run_*.csv`.
- **Split (leakage-safe):** per device, **runs 1,2 = train, run 3 = held-out validation**
  (no window leakage). `${CAPTURES_ROOT}/train/` is the 5-class symlink tree the trainer builds (§6.2);
  it is not part of the bundle.
- Windowing: `WIN = 1024` complex samples tiled **non-overlapping** (stride = `WIN`) over each frame's
  payload region `[PRE_ROLL : PRE_ROLL+ACTIVE] = [500 : 13812]` in `fp_model.py` — i.e. skipping the
  preamble and SIGNAL field (samples 100–500). Each window is the paper's **I/Q frame** (the model
  input), power-normalized to unit average power.
  - **13 I/Q frames per 802.11 frame:** `ACTIVE = 13312 = 13 × 1024` (the largest whole multiple of `WIN`
    fitting the payload; any leftover shorter than one window is dropped), so each 802.11 frame yields
    **13** windows. The counts follow: 6,731 Normal frames × 13 = 87,503 I/Q frames; the 107-frame OTA
    replay × 13 = 1,391.

### 3.2 Attack data — `${CAPTURES_ROOT}/attack_*/` (Tier 3, provenance)

These directories hold the **raw OTA capture sessions**. They are needed only to re-score the
over-the-air results from the raw `rx_*.bin` recordings (Tier 3) or to retransmit them (Tier 4). The
derived result CSVs that the figures read are shipped under `docs/` (Tier 1), so these directories are
not required to regenerate any figure. Only the small TX waveforms in `attack_assets/`
(`ch0_run3_107.bin`, `ch1_e04_s1.bin`) are included in the minimal bundle. The role-based directories:

| Directory | Role | Key contents |
|---|---|---|
| `attack_assets/` | OTA victim/adversary waveforms (Exp 2/3) | `ch0_run3_107.bin` (victim replay), `ch1_e04_s1.bin` (adversary δ) |
| `attack_b1/` | Exp 5.1 adversary-power sweep | `b1_cvnn_foolber_<method>.csv`, `b1_cvnn_tgt_d2_foolber_<method>.csv` |
| `attack_a1_transfer/` | Exp 4.2 cross-model transfer | per-model white-box score CSVs |
| `attack_a1_gain/` | Exp 4.1 gain sweep | `g16_<method>.csv` |
| `attack_enroll/` | attack re-enrollment baseline | `attack/ch0.bin`, `ota_perturbations/` (first δ bank) |

Filename legend: `ch0*.bin` = victim/legit replay · `ch1*.bin`, `pert_*/delta_*.npy` = adversary
waveform / δ bank · `rx*.bin(.rxlog)` = OTA capture / decode · `score_*.csv`, `blk_*.csv` = per-frame /
per-block fingerprint scores · `ber_*.csv` = decode BER · `*.sh` = run driver.

### 3.3 Trained models — `*.pt` (download into `trained_models/`)

The main CVNN classifier (`fingerprint_cvnn.pt`) ships in the repo so Tier-1 reachability (§2 Part C) runs
with no download, together with `docs/reachability_sample.npz` (300 windows/device). The other two
checkpoints (`fingerprint_cnn1d.pt`, `fingerprint_freqcvnn.pt`) arrive with the bundle (§3). The `.pt`
checkpoints recompute results from the classifier (§2C, §6–§7); §6.2 retrains an equivalent model from the
raw enrollment, and the supplied checkpoints simply avoid retraining and give bit-exact reproduction.

Three checkpoints, one per architecture, all scoring `WIN=1024` complex windows over 5 classes
(`device_1…5`). Place them in the repository root only if the recompute path is required:

| Model | Size | Arch | Role |
|---|---|---|---|
| **fingerprint_cvnn.pt** | 1.72&nbsp;M | CVNN | main classifier (Experiments 1–3, and the Experiment 5 sweeps) |
| fingerprint_cnn1d.pt | 490&nbsp;K | CNN1D | Exp 4.1 real-CNN baseline |
| fingerprint_freqcvnn.pt | 1.72&nbsp;M | FreqCVNN | Exp 4.4 frequency-domain variant |

Architectures: **CVNN** = time-domain complex CNN (`fp_model.py`), **FreqCVNN** = FFT-first
(`freqcvnn.py`), **CNN1D** = real 1-D CNN baseline (`cnn1d.py`).

---

## 4. Repository layout

```
AdRadio/                                # WiSec 2027 artifact (code only)
├── README.md  requirements.txt  LICENSE
├── trained_models/                     # fingerprint_cvnn.pt ships here (Tier-1 reachability); cnn1d/freqcvnn gitignored, arrive with the bundle (§3)
│
├── lib/                                # shared modules (imported across every experiment)
│   ├── fp_model.py                #   CVNN input windowing — single source of truth
│   ├── cvnn.py  cnn1d.py  freqcvnn.py   # classifier architectures
│   ├── attack_matrix.py           #   attack crafting (FGSM/BIM/MIM/PGD)
│   └── extract_frames.py          #   energy-detect + window extraction
│
├── training/                           # enrollment, training, and validation of the victim
│   ├── flowgraphs/                     #   GNU Radio 3.10 — OTA capture (§6.1)
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
│   └── README.md                       #   Exp 5.1/5.2 run via experiment_3; figures in plots/ (see §8)
│
├── plots/          # ALL figure/graph generators (self-locating; run from anywhere)
│   ├── plot_exp{1,2,3}_*.py  plot_tables.py          # Exp 1/2/3 paper figures + result tables
│   ├── fig1_col.py  fig2_check.py  fig3_check.py  make_a3_plot.py  # Exp 4 (4.1, 4.2, 4.3)
│   ├── figB1.py  figB2.py  # Exp 5 (5.1, 5.2)
│   └── make_results_workbook_byexp.py               # -> docs/…xlsx master index
│
└── docs/           # all result CSVs, rendered figures, workbooks, and write-ups
```

**Running scripts.** Each entry script locates the repository root (the directory containing `lib/`),
adds `lib/` to `sys.path`, and resolves `docs/` and the `.pt` models relative to that root. A script
therefore runs correctly **from any working directory**
(for example, `python3 plots/figB1.py` or `python3 experiment_1/exp1_reachability.py`).
The shared OTA runners (`experiment_3/campaign.sh`, `experiment_2/run_sc.sh`) reference each other
by repository-relative path, so the Experiment 2 and 3 campaigns operate across the directory split.

**Hosted externally (not in testbed; §3):** the ~19 GB recompute bundle (enrollment and models) and the raw
OTA capture sessions (provenance).

**Required separately for OTA (§5):** the `gr-foo` and `gr-ieee802-11` GNU Radio 3.10 OOT modules, built
and installed. They are **not** vendored here; the `training/flowgraphs/` scripts import the installed
`ieee802_11` module.

**Navigation.** The canonical reproducible path is the §6 pipeline, and the authoritative data
index is `docs/exp3_all_results_byexp.xlsx`. Superseded and dated snapshots from the development
repository were **not** carried into this artifact.

---

## 5. Environment & hardware

### Software

```bash
pip install -r requirements.txt     # numpy, scipy, matplotlib, pandas, openpyxl, scikit-learn, torch
```
For GPU training, install the CUDA build of `torch` from https://pytorch.org/. Every step except OTA
capture — Experiment 1, all scoring, and all figure regeneration — requires only Python and PyTorch,
not GNU Radio.

**Capture and OTA only (Experiments 2–3 and enrollment).** These steps require GNU Radio 3.10 and UHD,
with the `gr-foo` and `gr-ieee802-11` OOT modules **built and installed**. These modules are not
vendored here: obtain `gr-ieee802-11` from `<anonymized-fork-url>` (our fork; link withheld for
double-blind review) and `gr-foo` from its upstream, both on the `maint-3.10` branch:
```bash
cd gr-foo && mkdir build && cd build && cmake .. && make && sudo make install && sudo ldconfig
cd ../../gr-ieee802-11 && mkdir build && cd build && cmake .. && make && sudo make install && sudo ldconfig
```
Build `gr-foo` first. The Python modules `gnuradio`, `ieee802_11`, and `pmt` come from this system
installation rather than pip; the `flowgraphs/` scripts import `ieee802_11`.

### Our hardware testbed (Experiments 2–3)

- **Victim** — USRP2/N-series `192.168.10.3`, antenna **J1**, TX gain **0.3689**, enrolled as the
  victim class (`device_5` in this testbed).
- **Adversary** — USRP2/N-series `192.168.10.4`, antenna **J1**, **MIMO-cabled to the victim**; the
  only attacker.
- **Other enrolled devices** — B200mini/B205mini (`device_1 3263606`, `device_2 32593C4`,
  `device_3 3259376`, `device_4 3259368`), swapped one at a time into TX/RX; these are the
  classifier's other legitimate classes = the impersonation *targets*, not attackers.
- **Receiver** — B205mini, with `rx_gain = 0.93` throughout. A gain mismatch causes the *unperturbed* victim to misclassify, turning every fooling rate into a baseline artifact; the §6.5 gate detects this condition.

### Re-running from scratch on your own hardware (Tier 4)

The exact radios are not required; the serial numbers and IP addresses above record only the hardware we
used. Reproducing the full pipeline on a different SDR testbed requires the following roles:

- **At least five distinct transmitters:** one victim and four enrolled impersonation **targets**. Any
  SDRs with stable, distinguishable hardware impairments are suitable; we used N-series and B20x-mini
  USRPs. Each passes through the TX chain individually during enrollment.
- **One adversary transmitter** for the two-channel Experiment 3, clock- and PPS-locked to the victim.
  A shared 10 MHz reference and PPS keep the two streams phase-coherent; we MIMO-cabled two N-series
  radios, but any UHD radios that can share a reference are suitable.
- **One receiver.**

All radios operate through **UHD and GNU Radio 3.10** (USRP family). Match the following settings for comparable results:

| setting | value |
|---|---|
| center frequency | **5.29 GHz** |
| sample rate | **5 MS/s** |
| modulation | 802.11a/g **BPSK, rate 1/2** |
| RX gain | **fixed** (we used 0.93; the value may differ on other hardware), identical for enrollment and attack |
| TX gain | **calibrated per device** to a common target RX level (via `enroll_calib.py`) |

Then follow **§6** end to end: enroll (clean runs per device), train (§6.2), validate (§6.3), build
the victim replay (§6.4), pass the silent-adversary gate (§6.5), craft the δ banks (§6.6), and run the OTA
campaigns (§7, via `run_sc.sh` and `run_2ch.sh`).

> **Absolute numbers will differ.** A device fingerprint is specific to the physical radios, so the
> classifier, its clean accuracy, and the exact PSR and fooling-rate crossings **will differ** from
> ours; the model is trained on your hardware rather than loaded from ours. What reproduces is the
> **qualitative result**: a low-power adversary fools the OTA classifier while the 802.11 link continues
> to decode. The most common pitfall is an **uncalibrated gain** that causes the clean victim to
> misclassify, which the §6.5 gate detects in roughly 60 s.

---

## 6. Capture-to-attack pipeline (step by step)

Values in `<angle brackets>` must be supplied: `<N>` the number of
detected victim frames, `<scale>` from §6.5, `<victim>` the
victim label (e.g. `device_5`), `<V>` its 1-based class index (e.g. `5`), `<target>` the class to
impersonate, and `<K>` the target's 1-based index.

| Stage | Script | Radio | § |
|---|---|---|---|
| 1. Collect enrollment | `training/flowgraphs/enroll_d{1..5}_runs.sh`, then `training/flowgraphs/enroll_calib.py` | yes | 6.1 |
| 2. Train model | `training/train_fingerprint.py` | no | 6.2 |
| 3. Validate | `training/validation_confusion.py` | no | 6.3 |
| 4. Build victim replay | `experiment_3/exp3_map_detected_ids.py`, `experiment_3/build_ch0_matched.py` | no | 6.4 |
| 5. Silent-adversary&nbsp;gate | `experiment_3/run_2ch.py`, `experiment_3/score_2ch.py` | yes | 6.5 |
| 6. Craft δ banks | `experiment_3/exp3_make_ota_perturbations.py` | no | 6.6 |
| Experiment&nbsp;1:&nbsp;Type&nbsp;I&nbsp;Baseline&nbsp;| `experiment_1/exp1_reachability.py` | no | 7 |
| Experiment&nbsp;2:&nbsp;Type&nbsp;II&nbsp;Adversary&nbsp;| `experiment_2/run_sc.sh` / `campaign.sh sc_*` | yes | 7 |
| Experiment&nbsp;3:&nbsp;Type&nbsp;III&nbsp;Adversary&nbsp;| `experiment_3/run_2ch.sh` / `campaign.sh ch2_*` | yes | 7 |

### 6.1 Collect enrollment — `training/flowgraphs/enroll_calib.py`
Bring up one TX radio, calibrate its gain to a target RX level, and capture three clean runs per device
using the per-device wrappers (one radio initialization per run).
Produces `$CAPTURES_ROOT/enrollment/device_N/clean_run_{1,2,3}.bin` (+ sidecars).
Key args: `--device-id N` (1–5), `--run-id`, `--tx-serial` (serial= for the B200/B205 radios, IP for the usrp2
victim), `--tx-antenna` (`TX/RX` or `J1`), `--target-db` (use **−30**), `--fix-gain` (victim only:
**0.3689**), `--g0`, `--date` (new every session), `--tree enrollment|validation`.
```bash
cd training/flowgraphs
# edit each wrapper: attacker devices --target-db -30 --g0 <start>, drop --fix-gain; victim keeps --fix-gain 0.3689
bash enroll_d1_runs.sh    # then swap radio, enroll_d2_runs.sh, d3, d4, d5
```
Each run is approximately 1.2 GB and contains 100 or more frames, with `bad_packet=no` and the level within tolerance of −30 dB.

### 6.2 Train — `training/train_fingerprint.py`
**Retrain every session:** a model reused from a previous day fails on a re-leveled enrollment. Training
requires a `_5dev` symlink tree (`device_N` linking to `device_N`, clean runs only) and produces
`fingerprint_<tag>.pt`.
Args: `--root`, `--val-run` (3), `--epochs` (use **80**),
`--tag`, `--lr`, `--cfo-correct`, `--rebuild`.
```bash
ENR=$CAPTURES_ROOT/enrollment
TRN=$CAPTURES_ROOT/train; rm -rf "$TRN"
for n in 1 2 3 4 5; do mkdir -p "$TRN/device_$n"
  for r in 1 2 3; do ln -s "$ENR/device_$n/clean_run_$r.bin" "$TRN/device_$n/clean_run_$r.bin"; done; done
python3 training/train_fingerprint.py --val-run 3 --epochs 80 --root "$TRN" --tag cvnn
#   -> trained_models/fingerprint_cvnn.pt
```

### 6.3 Validate — `training/validation_confusion.py`
Per-device accuracy and confusion on the held-out run 3 (leakage-safe). The victim class should reach **100%** and the target class should be strong; the overall frame-level accuracy is **95.98%**.
```bash
python3 training/validation_confusion.py --model trained_models/fingerprint_cvnn.pt --base "$ENR"
```

### 6.4 Build the victim replay `ch0`
A clean baseband replay that reproduces the victim frame for frame; it is the basis for both crafting and the Experiment 2/3 ch0.
```bash
A=$CAPTURES_ROOT/attack_assets; mkdir -p "$A"; VIC="$ENR/<victim>"
python3 experiment_3/exp3_map_detected_ids.py --capture $VIC/clean_run_3.bin --rx-frames $VIC/rx_frames_run_3.jsonl \
    --out $A/detected_ids_run3.csv --thr-mult 2.0
python3 experiment_3/build_ch0_matched.py --dev-dir $VIC --run 3 --ids $A/detected_ids_run3.csv \
    --period 65661 --out $A/ch0_run3_<N>.bin        # N = number of detected frames; 65661 samples = frame spacing in the replay
```
`--thr-mult 2.0` **must match** §6.6 so that δ[k] corresponds to ch0 frame k.

### 6.5 Silent-adversary gate — `experiment_3/run_2ch.py` + `experiment_3/score_2ch.py`
Confirm that the unperturbed victim replay is classified as **the victim class** on the current testbed
before running any campaign. Transmit ch0 together with a zero-valued ch1 and score the result.
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
A pass requires the victim class to land with a **0% fooling rate** and **0 BER**; otherwise, correct
the testbed (usually the RX gain). Then compute **SCALE** = `ch0_active_rms / received_active_rms` (δ is
crafted at the received amplitude, and the build rescales it to the TX level). Measure both RMS values
over the payload samples only; the comment above `SCALE` in `experiment_3/campaign.sh` documents the
measurement. SCALE is session-specific and must be re-measured each session. Use it as
`<scale>` for Experiment 2 (our runs measured ≈**11.7**; `experiment_3/campaign.sh` defaults to `SCALE=8.7625` only if it is unset). Two-channel Experiment 3 uses `SCALE=1`.

### 6.6 Craft δ banks — `experiment_3/exp3_make_ota_perturbations.py`
Optimize δ against the **received** victim, over the payload only and with the MAC header masked. This
produces a bank directory of compact `.npy` deltas of shape `(N, 13312)` and a `manifest.csv`, one per
method and ε.
Args: `--model`, `--ch0` (= the **received** capture), `--from-capture`, `--thr-mult 2.0` (match §6.4),
`--nper <N>`, `--mask-lead 780`, `--tgt-dst <K>` (target 1-based index) **or** `--untargeted`,
`--victim-class <V>`, `--methods` (default 6: `fgsm bim mim pgd_a0.01 pgd_a0.05 pgd_a0.10`), `--epsilons`.
`--mask-lead 780` holds δ at zero over the first 780 payload samples, where the MAC header lies, so
perturbed frames still pass the receiver's address check.
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

## 7. Experiment catalog

Each experiment below lists its objective, the reproduction command, the result data, and the figure it
produces. The shared setup (model, enrollment, and δ banks) is described in §6, and the master data
index for all experiments is `docs/exp3_all_results_byexp.xlsx`.

### Experiment 1 (Type I Baseline) — Classifier reachability (no radio)
```bash
python3 experiment_1/exp1_reachability.py --model trained_models/fingerprint_cvnn.pt --base "$ENR" --out reachability.csv
```
Data: `reachability_matrix.csv`, `reachability_cvnn.csv`. Figures: `plots/make_e1_diagram.py`,
`plots/make_digital_vs_ota.py`. Produces the reachability table.

### Experiment 2 (Type II Adversary) — `campaign.sh sc_tgt_d1 | sc_unt`
δ summed into the victim replay (ch1 = zeros), fine grid 0.10–0.50. `campaign.sh` reads **exported**
`A` and `CH0`.
```bash
export A=$CAPTURES_ROOT/attack_assets; export CH0=$A/ch0_run3_<N>.bin
FINE=$(python3 -c "print(','.join(f'{round(0.1+0.01*i,2):g}' for i in range(41)))")
export MODEL=trained_models/fingerprint_cvnn.pt KEEP_CAPTURES=0 GUARD_FRAC=0 BER_PASSES=1 \
       PASSES_SC=2 SECS_SC=200 SCALE=<scale> METHODS="fgsm bim mim pgd_a0.01 pgd_a0.05 pgd_a0.10" EPS_GRID="$FINE"
export RUN=1 TGT=<target> PERT_D1=$A/pert_tgt_<target>_sc PERT_UNT=$A/pert_unt_sc
cd experiment_2 && ./run_sc.sh        # both single-channel arms (calls ../experiment_3/campaign.sh)
```
Output `$A/score_sc_{tgt_d1,unt}_r<RUN>_<method>.csv`. Figure: `plots/make_sc_vs_2ch_plot.py`.
`SECS_SC` must cover the full grid (≈200 s for the 41-point fine grid at 130 s only ~36 of 41 ε are scored); verify the scored ε count equals the grid size.

### Experiment 3 (Type III Adversary) — `campaign.sh ch2_tgt_d1 | ch2_unt`
Victim replay on ch0, δ on ch1, MIMO-locked, over the masked 16-point grid, with **`SCALE=1`**. The Qt GUI displays progress per method.
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
These four variations test whether the vulnerability depends on the classifier (Exp 4.4 is a table). Crafting follows §6.

- **Exp 4.1 — a real-valued model (CNN1D vs CVNN).** Train the real-valued CNN1D baseline, then compare it
  with the CVNN on both the reachability matrix and the white-box over-the-air attack (fooling rate and BER
  vs ε), testing whether the vulnerability is specific to the complex-valued model:
  ```bash
  python3 lib/cnn1d.py --root "$TRN" --val-run 3 --epochs 80   # -> trained_models/fingerprint_cnn1d.pt  ($TRN from §6.2)
  ```
  Data: `reachability_matrix_cnn1d.csv`, `docs/a1_*_ota_ber.csv` (CNN1D), `docs/a1cvnn_*_ota_ber.csv` (CVNN),
  aggregated into the workbook. Figure: `plots/fig1_col.py` produces `varA1_cnn1d_vs_cvnn_col.*`.
- **Exp 4.2 — transfer across models.** δ crafted on CVNN applied to CNN1D and the physics SVM/RF.
  `experiment_4/exp3_transfer_cnn1d_cvnn.py`. Data: `exp3_transfer_cnn1d_cvnn.{csv,npz}`,
  `exp3_ota_multimodel.csv`. Figures: `plots/fig2_check.py` (varA2 digital), `plots/fig3_check.py` (varA2 OTA).
  (Workbook sheets `A2-ML`, `A2-transfer-raw`.)
- **Exp 4.3 — explicit physical features (SVM/RF).** A classifier trained on explicit impairment features.
  `experiment_4/exp3_a3_feature_model.py`. Data: `exp3_a3_feature_model.{csv,npz}`. Figure:
  `plots/make_a3_plot.py` produces `docs/a3_feature_model_robustness.*`.
- **Exp 4.4 — spectral representation (frequency-domain CVNN).** FFT-first classifier (`lib/freqcvnn.py`); representation-invariance
  of the fragility, **reported as a table** (no figure). Data: `reachability_matrix_freqcvnn.csv`.

### Experiment 5 — Adversary realism variations
Both variations run through the Experiment 3 two-channel pipeline (§6); neither has dedicated scripts.

- **Exp 5.1 — perturbation power.** Two-channel fooling rate, target rate, and BER versus adversary gain
  and measured received PSR (ε = 0.01), with the distance-ratio reach. Runs on `$CAPTURES_ROOT/attack_b1`. Data:
  `docs/b1_cvnn_foolber_<method>.csv` (untargeted), `docs/b1_cvnn_tgt_d2_foolber_<method>.csv`
  (targeted device_2). Figure: `plots/figB1.py` (varB1 power sweep).
- **Exp 5.2 — one perturbation for every frame.** Per-frame matched δ versus a single universal waveform reused on every frame
  (the realizable adversary). Data: `docs/b2_matched_vs_unmatched.csv`. Figure:
  `plots/figB2.py` produces `varB2_matched_vs_universal.*`.

---

## 8. Regenerating every figure, table, and workbook

All figure scripts reside in `plots/` and locate the repository root automatically, so they run from any
directory. The `plot_exp*` scripts carry embedded data and the `fig*` scripts read the workbook; the
`make_*` scripts read the CSVs under `docs/`. The `make_*` scripts write `.png` and `.svg` under `docs/`;
the `plot_*` and `fig*` scripts write `.png` and `.pdf` under `plots/output/`, which `FIG_OUT` overrides.

**Master workbook (run first — the data index):**
`plots/make_results_workbook_byexp.py` produces `docs/exp3_all_results_byexp.xlsx` (sheets A1-ML, A2-ML,
A2-transfer-raw, A3-ML, A4-ML, B1-W, B2-W).

**Experiments 1–3 (main paper figures):** `plots/plot_exp1_sidebyside.py`,
`plots/plot_exp1_ber_sidebyside.py`, `plots/plot_exp1_singlecol.py` (Exp 1) · `plots/plot_exp2_sidebyside.py`
(Exp 2) · `plots/plot_exp3_sidebyside.py`, `plots/plot_exp3_rejection.py`, `plots/plot_exp3_untargeted.py`,
`plots/plot_exp3_targeted.py` (Exp 3) · `plots/plot_tables.py` (result tables). Supporting:
`plots/make_e1_diagram.py`, `plots/make_digital_vs_ota.py`, `plots/make_sc_vs_2ch_plot.py`.

**Experiment 4 (Exp 4.1–4.4):** `plots/fig1_col.py` (Exp 4.1) · `plots/fig2_check.py`, `plots/fig3_check.py` (Exp 4.2) · `plots/make_a3_plot.py` (Exp 4.3).
Exp 4.4 is reported as a table.

**Experiment 5 (Exp 5.1, 5.2):** `plots/figB1.py` (Exp 5.1) · `plots/figB2.py` (Exp 5.2).

Regenerate everything:
```bash
for f in make_results_workbook_byexp \
         plot_exp1_sidebyside plot_exp1_ber_sidebyside plot_exp1_singlecol plot_exp2_sidebyside \
         plot_exp3_sidebyside plot_exp3_rejection plot_exp3_untargeted plot_exp3_targeted plot_tables \
         make_e1_diagram make_digital_vs_ota make_sc_vs_2ch_plot \
         fig1_col fig2_check fig3_check make_a3_plot \
         figB1 figB2; do
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
