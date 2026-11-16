# B1 — Adversary-dominant power sweep, OTA crossover confirmation (protocol)

Scope this round: **B1–B4 only.** This document is the run protocol for B1 OTA; execute once the
radios are up. Digital B1 is already done (exp3_adversary_power_sweep.py): the adversary-stamp
crossover is ~0 dB PSR — below it the victim is retained, above it the classifier reports the
ADVERSARY's own class. OTA confirms this on the real channel and adds BER + frame rejection, which
the digital sweep cannot produce.

## Blocker at time of writing
`uhd_find_devices` = No UHD Devices Found. The 2-channel sink uses network USRPs
`addr0=192.168.10.3, addr1=192.168.10.4`; they must be powered and reachable on the 192.168.10.x
network before any capture. Also: no fresh enrollment exists yet, and per session discipline the
model must be REFIT for today's channel (a prior-day checkpoint fails on a freshly re-leveled
enrollment). Latest payload model on disk: fingerprint_cvnn.pt — do NOT reuse it OTA.

## Testbed (from wifi_adversary_tx.py)
- 2-channel USRP sink, 5.29 GHz, LO-offset manual tune.
- **Legit Tx (victim):** ch0, antenna J1, normalized tx_gain = 0.4930 (victim device_5 fixed).
- **Adversary:** ch1, antenna J1 (J1 = +7.9 dB vs J2, measured), gain = the swept knob adv_gain.
- Qt spectrum GUI MUST be visible for the run (MIMO-run rule), never offscreen.
- **SCALE = 1** for the 2-channel build (2ch does NOT scale delta; the received->TX SCALE applies
  only to single-channel).

## Session setup (before the sweep)
1. Bring up both USRPs; confirm `uhd_find_devices` sees .3 and .4.
2. Fresh enrollment in `.../enrollment/`, all 6 devices,
   attacker devices calibrated to the -30 dB target, victim device_5 fixed at 0.3689.
3. Refit the fingerprint model for the session -> fingerprint_cvnn.pt; verify clean
   per-device accuracy before attacking.

## PSR sweep
Digital crossover ~0 dB, so bracket realized PSR **-6, -3, 0, +3, +6 dB** (5 points).
**CRITICAL — measure realized PSR, do not assume it from adv_gain.** Normalized gain maps
non-linearly to dB and the received delta level has bitten us before (PSR-scaling bug: labels were
~25 dB optimistic). Procedure per point:
1. Victim-only capture -> measure victim received power P_v.
2. Adversary-only capture at a reference adv_gain -> measure P_a; solve adv_gain for each target
   realized PSR = 10log10(P_a/P_v).
3. Run the 2-channel capture at that adv_gain; RE-MEASURE realized PSR from the capture and report
   THAT (not the nominal), same as the workbook methodology.

## Measure at each PSR (the B1 deliverables)
- **Fooling rate** (victim windows leaving the victim class).
- **Landing-class distribution** — especially the fraction that becomes the ADVERSARY's own class
  (the overwrite signal), vs victim-retained vs other.
- **BER** via the subprocess software decode (ber_rx as a subprocess, never in-process with a live
  TX flowgraph; make minimal edits to the working decode).
- **Frame rejection** (fraction failing detection / address check / CRC) — a hot adversary jams.

## Expected result (to confirm)
Below ~0 dB: victim retained, low BER (stealthy regime, matches existing E-results). Above ~0 dB:
adversary-stamp overwrite — victim relabeled as the ADVERSARY (not a chosen third device), with BER
and frame rejection rising (power buys denial/overwrite, not impersonation). Report the OTA crossover
PSR and compare to the digital ~0 dB.

## Outputs
Per-PSR CSV (realized_psr, fool_rate, frac_victim, frac_adversary, frac_other, ber, frame_reject),
the raw captures in the dated dir, and a fooling/landing/BER-vs-PSR figure. Then B2-B4 follow using
the same session's captures where possible.
