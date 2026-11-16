# Standoff table (Table VII) — updated values (0920 sweep, eps = 0.01, MEAN over 16 phases)

Recomputed from the current 0920 untargeted power sweep (`docs/b1_cvnn_foolber_0920_<method>.csv`),
**mean over the 16 blind phases** (matching the B1 figure), measured-points PSR axis. Standoff ratio
da/dv = 10^(-PSR / (10 n)); values above one mean the adversary may stand farther from the receiver
than the victim, below one it must stand closer.

| row | PSR (dB) | n = 2 | n = 2.5 | n = 3 | n = 3.5 |
|---|---|---|---|---|---|
| half of frames (FGSM) | -4.0 | 1.58 | 1.45 | 1.36 | 1.30 |
| half of frames (BIM) | -4.9 | 1.76 | 1.57 | 1.46 | 1.38 |
| rate saturates (FGSM/BIM/MIM) | +6.2 | 0.49 | 0.56 | 0.62 | 0.67 |
| rate saturates (PGD) | +9.4 | 0.34 | 0.42 | 0.49 | 0.54 |
| BER exceeds 1e-2 | +1.7 | 0.82 | 0.86 | 0.88 | 0.89 |

Notes:
- Under the mean, 50% is reached in a narrow band: FGSM -4.0, BIM -4.9, MIM -4.3, PGD-0.01 -5.1,
  PGD-0.05 -4.4, PGD-0.10 -4.2 dB (interpolated).
- Saturation (99%) is method-dependent under the mean: FGSM/BIM/MIM at +6.2 dB, the three PGD
  variants at +9.1 to +9.4 dB -- so two "rate saturates" rows (collapse to one if preferred).
- The "half of frames" rows have standoff above one (adversary 1.3-1.8x farther), but every
  saturation row is below one (adversary must stand closer than the victim to fully saturate),
  and the BER row is also below one -- so under the mean metric full evasion requires power
  parity or better, not a distance advantage.
- The -9.5 / +1.7 saturation and BER values of the max-metric table do not apply here; only the
  BER row (+1.7, phase-independent) is shared.
