# Standoff table and B1 PSR answers (from the lab machine)

Answered from the B1 data and code. Key finding up front: the paper's 99%, saturation, and BER values match the current (0920) data, but the 50% values do not — they trace to the **superseded 0918 sweep** — and that is the source of the VII-B1 vs VIII disagreement.

## Q1 — Untargeted 50% PSR, per method (current 0920 data, eps = 0.01, max over phase)

Source `docs/b1_cvnn_foolber_0920_<method>.csv`; PSR via `g2psr` (see Q4).

| method | 50% interp (dB) | first grid at/above 50% (dB) |
|---|---|---|
| FGSM | -16.2 | -14.5 |
| BIM | -15.3 | -14.5 |
| MIM | -12.5 | -9.5 |
| PGD-0.01 | -15.2 | -14.5 |
| PGD-0.05 | -12.3 | -9.5 |
| PGD-0.10 | -12.5 | -9.5 |

If the "rate" is instead the **mean** over phase, 50% lands much higher, -5 to +1.7 dB. Neither metric gives -29 to -21.

## Q2 — Targeted 50% PSR (max over phase)

The only targeted B1 sweep is **device_2** (`b1_cvnn_tgt_d2_foolber_0920_*`), not "Tx1" (= device_1) — flag this mismatch with the paper.

| method | 50% interp (dB) | first grid at/above 50% (dB) |
|---|---|---|
| FGSM | -12.0 | -9.5 |
| BIM | -12.0 | -9.5 |
| MIM | -14.3 | -9.5 |
| PGD-0.01 | -15.4 | -14.5 |
| PGD-0.05 | -12.0 | -9.5 |
| PGD-0.10 | -12.1 | -9.5 |

## Q3 — BIM -25? FGSM -19 or -21? (priority)

- **FGSM -21 / -19 comes from the SUPERSEDED 0918 sweep**, not the current data. In `docs/b1_cvnn_gain_0918_fgsm.csv` (col `fool_max_pct`): FGSM 50% = **-21.4 interpolated, -19.5 grid**. So VII-B1 used the interpolated value (-21) and VIII used the grid point (-19.5 ~ -19): same old data, two conventions.
- **BIM -25 matches neither.** 0918 gives BIM 50% ~ -17.6; 0920 gives -15.3. Nothing available produces -25 for BIM.
- Current (0920) values: FGSM -16.2, BIM -15.3 (interp).
- So the B1 50% row is built on the retired 0918 sweep (0918 for FGSM, unknown for BIM) while the 99%/BER come from 0920 — an internal inconsistency on top of the interp-vs-grid split between the two sections.

## Q4 — How the B1 PSR axis is defined (priority)

It is the **measured received power ratio** (adversary received power over victim received power), **not** the nominal 10*log10(2*eps^2). `PSR_MEAS = {20:-4.5, 25:1.7, 30:6.3, 35:9.9, 40:10.0}` (in `make_b1_fool_ber_cvnn.py`) holds the **measured** points at adversary gains 20-40, obtained by power-subtraction: a total adversary+victim capture minus a victim-only (silent-adversary) capture gives the adversary power, ratioed to the victim received power. **Below gain 20 it is linearly extrapolated** (`g2psr(g) = -4.5 + (g - 20)`), because there the adversary sits below the receiver noise floor and the ratio cannot be measured directly. Every 50% crossing (all below gain 20) therefore sits in the **extrapolated** region, not the measured one — the absolute dB values carry that caveat.

## Q5 — 99% PSR, per method; is -9.5 a grid point?

Yes, **-9.5 dB is a grid point** (adversary gain 15). All six methods first reach 99% (max over phase) at **-9.5 to -9.7 dB**: FGSM -9.7, the other five -9.6. So "five of six reach 99-100% by -9.5" is correct, and FGSM is right at it too in the 0920 data.

## Q6 — BER crossing 1e-2: uniform or per-method?

**Uniform: every method first exceeds BER = 1e-2 at +1.7 dB** (gain 25). BER is at the floor through -4.5 dB (gain 20) and exceeds 1e-2 at the next grid point, +1.7 dB, for all six — it does not vary by method (the deltas all carry the same L-infinity budget at eps = 0.01, so the link-damage onset is common).

## Q7 — Grid points and spacing

**13 PSR points** (adversary gains -20 to +40, step 5 dB):
-44.5, -39.5, -34.5, -29.5, -24.5, -19.5, -14.5, -9.5, -4.5, 1.7, 6.3, 9.9, 10.0 dB.
Spacing is a uniform **5 dB in the -30 to -20 region** (extrapolated). Near 0 dB it follows the measured points: -4.5, then 1.7 (6.2 dB step), then 6.3 (4.6 dB step); it compresses and clamps at the +10 dB usrp2 ceiling.

## Net for the standoff table

The consistent, current numbers are 50% at **-12 to -16 dB (max over phase)**, 99% at **-9.5 dB**, BER above 1e-2 at **+1.7 dB** — all from the 0920 eps = 0.01 sweep. The -29 to -23 / -25 / -21 / -19 values in the current draft come from the superseded 0918 sweep (and for BIM cannot be reproduced at all), so both VII-B1 and the VIII standoff table should be recomputed from the 0920 data to be self-consistent.
