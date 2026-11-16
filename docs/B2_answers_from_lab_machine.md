# B2 answers (from the lab machine)

Answered from the code and the actual delta files. Priority answers (Q1, Q4) first.

## Q1 — Representation: one waveform the length of a whole frame

It is one full-frame waveform of 13 **distinct** windows x 1024 = 13,312 active samples, and the **same** full-frame waveform is used on every frame. It is **not** a single 1024-sample waveform repeated in every window — the 13 windows differ from each other.

- `exp3_make_ota_perturbations.py` -> `craft_universal_frames` optimizes a delta of shape `(nS=13, 1, WIN=1024)` ("ONE frame-delta shared across ALL frames; per-window structure within the frame is retained"), stored as one `(13, WIN)` frame broadcast identically to every frame.
- Verified on `pert_uni_tgt_d1_masked/delta_bim_e0.05.npy` (shape `(200, 13312)`): all 200 frames identical; window 0 and window 1 differ (max |delta| ~ 0.0043).

So the B2 text should say "one waveform for every **frame**," not "every window."

## Q4 — Transmission: same waveform, same intra-frame offset, on every frame, fixed

`build_ch1.py`:
```
s = base + k * FRAME_PERIOD + PRE_ROLL      # FRAME_PERIOD = 65661, PRE_ROLL = 500
buf[s:s + ACTIVE] += d[k] * rot
```
For a universal bank `d[k]` is identical for every frame `k`, so the same waveform is added to every frame at the same offset: exactly +500 samples (`PRE_ROLL`) into each frame's active region. The offset is deterministic (`k*FRAME_PERIOD + PRE_ROLL`) and does not change frame to frame — identical placement to the matched attack, only the content differs. The threat model's time-alignment assumption still applies to B2.

## Q2 — What data is it crafted on?

- OTA banks: crafted on the received held-out run (`--from-capture` on `clean_run_3`, or against the `ch0` replay), on `nper` frames (~107), minimizing the **mean** cross-entropy over all frames and all 13 windows.
- Digital counterpart (`exp3_universal_digital.py`): crafts on the train split and scores on a held-out test split (strict separation).
- Crafted on the frames it is scored on? In the OTA path the delta is crafted on the received `clean_run_3` frames and then scored on the OTA re-capture of those same frames — same frame content, different capture, so there IS frame-content overlap in the OTA path (worth stating).

## Q3 — How is it crafted?

Loss: cross-entropy (untargeted ascends the true-class loss; targeted descends toward the chosen class). Per method (`exp3_attack_matrix.py`, `PGD_STEPS = 10`), same six as the matched attack:

| method | steps | alpha | momentum | random start |
|---|---|---|---|---|
| FGSM | 1 | eps | none | no |
| BIM | 10 | eps/10 | none | no |
| MIM | 10 | eps/10 | mu = 1.0 (L1-normalized grad) | no |
| PGD-0.01 | 10 | 0.01 | none | yes |
| PGD-0.05 | 10 | 0.05 | none | yes |
| PGD-0.10 | 10 | 0.10 | none | yes |

Each step is a sign-gradient update clamped to [-eps, eps]. The eps bound is the **same as the matched attack**: a per-sample L-infinity limit on each (real, imag) component, on the unit-RMS window domain, un-normalized by per-window mean RMS for transmission.

Header mask: the universal B2 banks (`pert_uni_*_masked*`) are masked — window 0's first 780 samples held at zero (verified, first-nonzero index 780); the plain matched per-frame bank `pert_unt_rx107` was not (first-nonzero index 0). Confirm the matched B2 arm used the same 780 mask as the universal arm; a mismatch would be a confound.

## Q5 — Frames scored, same session?

107 frames per eps point (frames_sent 107; decoded ~106-107 at low eps, falling at high eps as the link degrades). Both arms (matched, universal) and both goals (untargeted, targeted) are in the one file `docs/b2_matched_vs_unmatched.csv` (mode unt/tgt, arm mat/uni), i.e. the same capture session as the matched attack.

## Q6 — Targeted vs untargeted: separate perturbations?

Separate perturbations: untargeted (`--untargeted`) and targeted (`--tgt-dst`) are crafted in separate runs, one universal delta per (goal, method, epsilon).

## To confirm on the paper side

- Whether the matched B2 arm used the same 780-sample header mask as the universal arm (Q3).
- Whether the paper's B2 figure uses the OTA path (frame-content overlap) or the digital train/test-split path (Q2).
