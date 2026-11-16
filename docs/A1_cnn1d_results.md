# A1 — Real-valued CNN1D: does the fragility generalize across architectures?

**Question (from experiment_variations.docx, A1).** Is the fragility we found — the CVNN's decision
boundaries lying far inside the physical distance between devices — a property of the complex-valued
architecture, or of the RF-fingerprinting task itself?

**Method.** Train a real-valued 1D CNN on the same enrollment windows and re-run the
reachability matrix and attack sweep on it, then compare. To isolate real-vs-complex, the CNN1D
(`exp3_cnn1d.py`) mirrors the CVNN layer-for-layer — same channels (32/64/128/128), kernels
(7/5/5/3), pooling, and head — but uses ordinary real convolutions over a 2-channel (real, imag)
input. Its `forward(re, im)` matches the CVNN interface, so the existing attack, targeted-PGD, and
reachability code runs unchanged: the same input convention, the same L-infinity budget, the same
six attacks. Training is identical — same cleaned data root, same drift-augmentation, same 80-epoch
AdamW/cosine schedule, run 3 held out.

**The CNN1D is an equally good fingerprinter.** Same-session validation frame accuracy 100%
(per device 103/103, 103/103, 104/104, 104/104, 103/103), matching the CVNN. This matters: the
comparison is only meaningful because both models classify the devices well.

## Result: the reachability structure is essentially architecture-independent

| Metric | CVNN | CNN1D |
|---|---|---|
| Same-session frame accuracy | 100% | 100% |
| Parameters | 422,277 | 119,013 |
| Reachability radii R (20 pairs) | — | Pearson corr with CVNN = **0.987** |
| Median \|R_cvnn − R_cnn1d\| | — | 0.0008 (max 0.025, at the noisiest cell) |
| rho = R / eps_phys, max over pairs | 0.172 | 0.136 |
| All rho ≤ 1 (physical bound holds) | yes | yes |
| Shortfall Delta_i | ≈ 0 | ≈ 0 |

Side-by-side row minima: CVNN [0.0070, 0.0125, 0.0242, 0.0055, 0.0180] vs
CNN1D [0.0070, 0.0117, 0.0250, 0.0047, 0.0188]. The victim Tx5 co-dominant Tx1/Tx4 attractor
structure is reproduced. The largest per-source radius is the same target (Tx3 = device_3) in both.

The one notable pair-difference (Tx3→Tx2, 0.122 vs 0.097) is the cell whose reachability search had
40 of 300 windows unreachable within the search bracket — the noisiest median in either matrix — so
the gap there is search noise, not an architecture effect.

## Conclusion

A completely different, real-valued classifier — trained on the same data with the same
augmentation, and an equally accurate fingerprinter — produces the same reachability radii
(corr 0.987), the same rho ≤ 1 relationship to the physical impersonation bound, and the same
near-zero optimization shortfall. **The fragility is a property of the RF-fingerprinting task, not
of the complex-valued architecture.** The paper's central reachability finding is not an artifact of
the CVNN.

Caveats: (1) the CNN1D has fewer parameters (a complex conv is four real convs); this is inherent to
the real-vs-complex control and does not affect the conclusion, since the lower-capacity model
reaches the same accuracy and the same radii. (2) Both models were trained white-box on the same
data; cross-model transferability (A2) is the separate test of whether an attack crafted on one
transfers to the other.

## Artifacts
- `fingerprint_cnn1d.pt` — the trained CNN1D checkpoint.
- `reachability_matrix_cnn1d.{npz,csv}` — CNN1D reachability matrix.
- `reachability_shortfall_curves_cnn1d.{npz,csv}` — CNN1D attack sweep / eps_50 curves.
- `exp3_A1_cnn1d_compare.npz` — both matrices, eps_phys, and both rho matrices.
- Re-run scripts (`exp3_reachability_matrix.py`, `exp3_untargeted_eps50.py`,
  `exp3_shortfall_curves.py`) gained `--arch {cvnn,cnn1d}`; default cvnn, so the CVNN path is
  unchanged.
