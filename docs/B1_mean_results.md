# B1 results — MEAN over 16 blind phases (0920 CVNN OTA, eps = 0.01)

Power sweep, fixed eps = 0.01, adversary gain swept. Rates are the mean over the 16 blind
inter-channel phase offsets (the realistic blind-adversary metric). PSR via the measured-
points axis (measured at gains 20–40 by power subtraction, linearly extrapolated below).
BER is over decoded frames; '—' means no frames decoded (link failed).

### Untargeted fooling rate (%)

| gain | PSR (dB) | FGSM | BIM | MIM | PGD-0.01 | PGD-0.05 | PGD-0.10 |
|---|---|---|---|---|---|---|---|
| -20 | -44.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| -15 | -39.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| -10 | -34.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| -5 | -29.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0 | -24.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 5 | -19.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 10 | -14.5 | 9.7 | 8.3 | 2.5 | 5.0 | 0.9 | 1.2 |
| 15 | -9.5 | 36.7 | 37.6 | 32.9 | 34.9 | 33.9 | 33.7 |
| 20 | -4.5 | 48.2 | 51.0 | 49.2 | 52.2 | 49.6 | 49.3 |
| 25 | 1.7 | 70.1 | 68.7 | 75.1 | 69.1 | 66.5 | 65.3 |
| 30 | 6.3 | 99.9 | 99.0 | 99.6 | 96.0 | 94.8 | 93.5 |
| 35 | 9.9 | 100.0 | 100.0 | 100.0 | 99.9 | 99.9 | 99.9 |
| 40 | 10.0 | 100.0 | 100.0 | 100.0 | 99.9 | 99.9 | 99.9 |

### Targeted (device_2) target rate (%)

| gain | PSR (dB) | FGSM | BIM | MIM | PGD-0.01 | PGD-0.05 | PGD-0.10 |
|---|---|---|---|---|---|---|---|
| -20 | -44.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| -15 | -39.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| -10 | -34.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| -5 | -29.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0 | -24.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 5 | -19.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 10 | -14.5 | 0.0 | 0.0 | 5.3 | 5.8 | 0.1 | 0.2 |
| 15 | -9.5 | 29.6 | 30.3 | 31.5 | 34.3 | 32.2 | 30.6 |
| 20 | -4.5 | 39.4 | 43.8 | 43.8 | 48.2 | 46.4 | 46.2 |
| 25 | 1.7 | 55.4 | 58.8 | 62.4 | 62.7 | 65.1 | 63.9 |
| 30 | 6.3 | 63.3 | 71.0 | 68.6 | 72.8 | 75.8 | 73.9 |
| 35 | 9.9 | 75.4 | 80.7 | 76.2 | 97.2 | 98.4 | 98.7 |
| 40 | 10.0 | 72.2 | 81.9 | 76.9 | 97.1 | 97.8 | 98.2 |

### Untargeted BER (over decoded frames)

| gain | PSR (dB) | FGSM | BIM | MIM | PGD-0.01 | PGD-0.05 | PGD-0.10 |
|---|---|---|---|---|---|---|---|
| -20 | -44.5 | 0 | 0 | 0 | 0 | 0 | 0 |
| -15 | -39.5 | 1.2e-05 | 0 | 0 | 0 | 0 | 1.9e-05 |
| -10 | -34.5 | 0 | 9.6e-06 | 0 | 0 | 0 | 0 |
| -5 | -29.5 | 0 | 7.2e-06 | 0 | 0 | 1.9e-05 | 0 |
| 0 | -24.5 | 0 | 7.2e-06 | 0 | 0 | 0 | 0 |
| 5 | -19.5 | 0 | 4.8e-06 | 0 | 0 | 0 | 0 |
| 10 | -14.5 | 2.2e-05 | 0 | 0 | 0 | 0 | 0 |
| 15 | -9.5 | 0 | 0 | 0 | 0 | 7.2e-06 | 0 |
| 20 | -4.5 | 1.0e-04 | 6.3e-05 | 2.4e-05 | 2.5e-05 | 8.9e-05 | 8.9e-05 |
| 25 | 1.7 | 1.3e-02 | 1.5e-02 | 3.0e-02 | 4.3e-02 | 1.8e-02 | 3.2e-02 |
| 30 | 6.3 | 3.3e-01 | 4.2e-01 | 4.1e-01 | 3.7e-01 | 4.0e-01 | 4.1e-01 |
| 35 | 9.9 | 4.3e-01 | — | 4.4e-01 | — | — | — |
| 40 | 10.0 | 4.5e-01 | 4.5e-01 | — | — | 4.7e-01 | — |

### Targeted (device_2) BER (over decoded frames)

| gain | PSR (dB) | FGSM | BIM | MIM | PGD-0.01 | PGD-0.05 | PGD-0.10 |
|---|---|---|---|---|---|---|---|
| -20 | -44.5 | 0 | 0 | 0 | 0 | 0 | 0 |
| -15 | -39.5 | 0 | 0 | 0 | 1.7e-05 | 0 | 0 |
| -10 | -34.5 | 0 | 0 | 0 | 0 | 0 | 0 |
| -5 | -29.5 | 1.4e-05 | 0 | 0 | 0 | 0 | 0 |
| 0 | -24.5 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | -19.5 | 0 | 7.2e-06 | 0 | 0 | 0 | 0 |
| 10 | -14.5 | 0 | 0 | 0 | 0 | 0 | 0 |
| 15 | -9.5 | 0 | 0 | 0 | 0 | 6.7e-05 | 3.9e-04 |
| 20 | -4.5 | 1.4e-04 | 9.6e-05 | 6.3e-05 | 1.9e-04 | 9.9e-05 | 5.8e-05 |
| 25 | 1.7 | 5.4e-02 | 1.6e-02 | 5.1e-02 | 1.2e-01 | 4.2e-02 | 8.1e-02 |
| 30 | 6.3 | 3.8e-01 | 3.7e-01 | 4.0e-01 | 3.1e-01 | 3.8e-01 | 3.2e-01 |
| 35 | 9.9 | — | — | — | — | — | — |
| 40 | 10.0 | — | — | — | — | — | — |

### Standoff table (Table VII) — mean over phase

Standoff ratio da/dv = 10^(-PSR / (10 n)); above one the adversary may stand farther from
the receiver than the victim, below one it must stand closer.

| row | PSR (dB) | n = 2 | n = 2.5 | n = 3 | n = 3.5 |
|---|---|---|---|---|---|
| half of frames (FGSM) | -4.0 | 1.58 | 1.45 | 1.36 | 1.30 |
| half of frames (BIM) | -4.9 | 1.76 | 1.57 | 1.46 | 1.38 |
| rate saturates (FGSM/BIM/MIM) | +6.2 | 0.49 | 0.56 | 0.62 | 0.67 |
| rate saturates (PGD) | +9.4 | 0.34 | 0.42 | 0.49 | 0.54 |
| BER exceeds 1e-2 | +1.7 | 0.82 | 0.86 | 0.88 | 0.89 |

Notes: half of frames reached in a narrow band (FGSM -4.0, BIM -4.9, MIM -4.3, PGD-0.01 -5.1,
PGD-0.05 -4.4, PGD-0.10 -4.2 dB). Saturation (99%) is method-dependent: FGSM/BIM/MIM +6.2 dB,
PGD +9.1 to +9.4 dB. Every saturation and the BER row have standoff below one — under the mean
(blind-adversary) metric, full evasion requires power parity or closer.
