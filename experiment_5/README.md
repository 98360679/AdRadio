# Experiment 5 — Adversary Realism Variations (B-series)

Experiment 5 evaluates how the attack holds up under more realistic adversary models.
The two variations reported in the paper are:

- **B1 — power sweep:** attack success and BER vs the *measured received* PSR, at a fixed
  perturbation budget (ε = 0.01), as the adversary's transmit power varies.
- **B2 — matched vs. unmatched perturbation:** a per-frame-matched δ against a single
  universal (unmatched) δ applied to every frame.

Both reuse the **Experiment 3 two-channel OTA pipeline** (`../experiment_3/`, driven by
`run_2ch.sh` / `campaign.sh`) with different perturbation banks and ε/power configurations —
there are no B-specific compute scripts. The figures are generated in `../plots/`:

- `make_b1_combined_cvnn.py`  → B1 (Fig. 13)
- `make_b2_matched_vs_unmatched.py` → B2 (Fig. 14)
- `make_pwr_fool_ber.py` → B1 per-method power sweep


