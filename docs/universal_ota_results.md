# Universal perturbation — OTA results (2026-08-21)

One **universal** adversarial perturbation (a single waveform radiated identically on every
victim frame) impersonating the victim's RF fingerprint, evaluated over the air. Companion to
the per-frame attack. Crafting: `craft_universal_frames` in `exp3_make_ota_perturbations.py`
(`--universal`) — one frame-δ shared across all frames, obtained by averaging the raw
per-frame gradients **before** the sign (a mean of per-frame signs would let frames cancel).

## Setup

- **Frequency** 5.29 GHz. **GPU** NVIDIA RTX A6000. RX gain 0.93; adversary/victim TX gain 0.493.
- **Fresh enrollment** `enrollment/`, 5 devices (device_1..device_5), 3 runs
  each, attackers calibrated to −30 dB, victim device_5 at fixed gain 0.3689 (antenna J1).
- **Model** `fingerprint_cvnn.pt` (CVNN, fine-tuned from an earlier checkpoint, best-run test frame acc
  1.000; run-3 held-out 96.4%). Victim **device_5 (D5)** and target **device_1 (D1)** both
  classify at 100%; the only confusion is device_2↔device_3 (see `confusion_matrix_run3`).
- **Victim replay** `ch0_0821_d6r3.bin`, 141 frame-id-matched clean frames.
- **Silent-adversary gate PASSED**: victim replay alone lands device_5 100%, fool 0%, BER 0.
- **SCALE = 9.8229** (received→TX). Applied to single-channel only; two-channel canonical is
  SCALE=1 (δ radiated on its own channel, does not ride ch0).
- Victim D5 = device_5, target D1 = device_1. Metrics per (attack, ε): targeting/fooling %,
  plurality landing class, frames sent/decoded, bit errors, BER (real 802.11 decode).

## E2 — Single-channel OTA (δ rides victim replay), targeted, full 41-ε grid, SCALE=9.82

100%-targeting onset (device_5 → device_1):

| attack | onset ε | BER at 100% | high-ε note |
|---|---|---|---|
| pgd_a0.05 | 0.11 | 0 | 100% through 0.50 |
| pgd_a0.10 | 0.11 | 0 | 100% through 0.50 |
| bim | 0.15 | 0 | roll-off to 72% at 0.50 |
| mim | 0.18 | 0 | roll-off to 83% at 0.50 |
| pgd_a0.01 | 0.34 | 0 | slow ramp |
| fgsm | — | — | 0% at every ε (fails) |

Working attacks impersonate at **BER ≈ 0** (link intact, frames_decoded ~140/141) — stealthy.
FGSM never impersonates yet accrues the most bit errors at high ε (up to 2.3e-4 at 0.50).

## E3 — Two-channel OTA (δ on separate MIMO channel), targeted, phase-0, 21-ε grid

Phase-0 only: a single power-up fixes the inter-channel phase (shared-LO coherent MIMO); the
blind 4-phase sweep is redundant (phase 0 wins for every ε), so one phase per ε.

**Unscaled (SCALE=1) — the canonical two-channel:** 100%-targeting onset ε≈0.15 (pgd_a0.05/0.10),
0.20 (bim), 0.25 (mim); **pgd_a0.01 and fgsm fail (0%)**. BER ≈ 0 through the grid (stealthy);
mild roll-off to ~92–94% at ε=0.30. Two-channel needs a **higher budget than single-channel**
(δ on its own channel, not riding the TX-level victim).

**Scaled (SCALE=9.82) — over-amplified reference:** onset drops to ε≈0.02–0.04 (scaling lifts δ
~10×), and pgd_a0.01 is rescued (100% from 0.025). But BER **explodes** as ε rises — bim/mim/pgd
climb to BER 0.25–0.49 and `frames_decoded` collapses toward 0 (link destroyed) by ε≈0.20–0.25,
while targeting stays 100%. This is the **stealth dichotomy** made physical: the two-channel
adversary radiates on a channel the receiver's equalizer cannot invert, so buying targeting
margin corrupts the payload — impersonation succeeds to the fingerprinter while the data link
dies. The unscaled run sits in the stealthy corner; the scaled run trades stealth for jamming.

## Untargeted universal (push victim OFF device_5)

- **Single-channel (SCALE=9.82, 41-ε):** fooling onset ε≈0.11 (pgd_a0.05/0.10), 0.13 (bim),
  0.17 (mim), ~0.30 (pgd_a0.01); fgsm fails. BER ≈ 0.
- **Two-channel (SCALE=1, phase-0, 21-ε):** fooling onset ε≈0.12–0.15 (pgd_a0.05/0.10), 0.20
  (bim), 0.25 (mim); pgd_a0.01 and fgsm fail. BER ≈ 0; roll-off at 0.30.
- **Every working attack lands on device_1** (the dominant attractor) at the fooling onset —
  so "off the victim" and "onto D1" are the same event; untargeted ≡ targeted-to-D1 in outcome.

## Findings

1. **The universal perturbation works over the air** — one waveform reused on every frame
   impersonates the victim at 100% for the iterative attacks (BIM, MIM, PGD-0.05/0.10).
2. **Digital→OTA gap ≈ one ε step** (e.g. BIM single-channel digital onset 0.10 → OTA 0.15).
3. **Single- vs two-channel:** two-channel costs more budget (unscaled onset ~0.15–0.25 vs
   single-channel ~0.11–0.18), consistent with the analytical budget ratio.
4. **Stealth dichotomy confirmed OTA:** unscaled two-channel is stealthy (BER≈0); scaling the
   same attack drives BER to ~0.5 and destroys the link — impersonation without stealth.
5. **FGSM fails universally over the air** (single-step can't find a shared direction);
   **pgd_a0.01 fails at two-channel** unless over-scaled.
6. **Universal costs budget vs per-frame** (per-frame single-channel reaches 100% at ε≈0.05;
   universal needs ε≈0.11–0.18) — the price of one waveform fooling all frames.

## Artifacts

- Banks: `pert_uni_tgt_d1_sc` (41-ε), `pert_uni_tgt_d1_masked` / `_ext` (16/21-ε),
  `pert_uni_unt_sc`, `pert_uni_unt_masked_ext` (all in the attack session directory).
- Scores: `score_sc_tgt_d1_r2_*` (SC targeted, full 41-ε), `score_ch2_tgt_d1_r3_*` (2ch unscaled),
  `score_ch2_tgt_d1_r4_*` (2ch scaled), `score_sc_unt_r1_*`, `score_ch2_unt_r3_*`.
- Accounting CSV: `sc_accounting_r2.csv`. Confusion matrix: `docs/confusion_matrix_run3.*`.
- Code: `exp3_make_ota_perturbations.py --universal/--bank-frames`, `campaign.sh` PHASES,
  `exp3_universal_digital.py`, `exp3_confusion_run3.py`. Algorithm 2 in `experiment_setup.tex`.

## Workbooks (T7, one sheet per method, full score columns)

| workbook | arm | grid | note |
|---|---|---|---|
| `E2_SC_0821_tgt_d1_fine.xlsx` | single-channel targeted | 41-ε (0.10–0.50) | δ scaled (SCALE=9.82) |
| `E2_SC_0821_unt_fine.xlsx` | single-channel untargeted | 41-ε | δ scaled |
| `E3_2ch_0821_tgt_d1_masked.xlsx` | two-channel targeted | 21-ε (0.01–0.30) | SCALE=1 (canonical), phase-0 |
| `E3_2ch_0821_tgt_d1_masked_scaled.xlsx` | two-channel targeted | 21-ε | SCALE=9.82 (over-scaled reference) |
| `E3_2ch_0821_unt_masked.xlsx` | two-channel untargeted | 21-ε | SCALE=1, phase-0 |

Named to mirror the `E2_SC_/E3_2ch_*` set; source score CSVs in the attack session directory.
