# Fingerprint-classifier paragraph — code verification and reproducibility note

Verifies the "Fingerprint classifier" paragraph in `experiment_setup.tex` (lines ~100-113)
against the training code, and records the seeding / reproducibility status. Code left as-is
(per decision); this note documents behaviour, it does not change it.

## Verdict

The paragraph is an accurate description of the code. Drift-augmentation is genuinely applied —
defined AND called on every training batch, training-only, on by default. All four operations
match the paper's stated distributions exactly. The pinned checkpoint is the described CVNN.

## Drift-augmentation — faithful to the paragraph

`drift_augment` (`exp3_cvnn.py:21`) is invoked in the training loop (`exp3_cvnn.py:221`) on every
batch, gated by `--no-aug` which defaults to OFF (augmentation ON). It is applied to TRAINING
batches only; validation (`per_device_eval`) runs on raw windows.

| Paragraph claim | Code | Match |
|---|---|---|
| random global phase | `phi = (rand·2−1)·π` → uniform [−π, π] | yes |
| CFO uniform ±3000 Hz | `df = (rand·2−1)·3000` Hz, applied as `exp(j·2π·df/FS·n)` | yes |
| circular time shift up to 6 samples | `randint(−6, 7)` + gather-modulo (circular) | yes |
| AWGN, SNR uniform [8,30] dB | `snr = rand·22 + 8`, complex AWGN, variance `npow/2` per component | yes |

CFO magnitude is physically meaningful: FS = 5 MHz, so ±3000 Hz over the 1024-sample window is a
phase ramp of up to ~±220°, a real inter-run drift, not a negligible one.

## Other paragraph claims — confirmed

- Checkpoint `fingerprint_cvnn.pt` is the CVNN (complex-conv `rr/ri/ir/ii` keys),
  **423,693 params ≈ 4.2×10⁵** — matches "4.2×10⁵".
- Output layer `Linear 128→5` — NUM_CLASSES = 5, matches.
- Optimizer/schedule: AdamW, lr `6e-4`, weight decay `1e-4`, cosine annealing, cross-entropy with
  label smoothing `0.05`, batch size 256 — all match.
- "fine-tune 80 epochs, hold run 3 out": the documented training command (`README.md:126`) passes
  `--epochs 80 --val-run 3`. NOTE: the code default for `--epochs` is 40, so the 80 comes from the
  invocation, not the default — keep the exact training command on record for reproducibility.

## Seeding / reproducibility status (code left as-is)

The augmentation draws ARE governed by a seed, with three caveats worth stating in the methods:

1. **Seeded, single global stream.** `main()` sets `torch.manual_seed(0); np.random.seed(0)` once
   (`exp3_cvnn.py:198`); `torch.manual_seed` also seeds the CUDA generator, so the phase/CFO/shift/
   noise draws are reproducible. But they share the global RNG with model init, `DataLoader(shuffle=
   True)`, and dropout — so the augmentation stream is reproducible only if the entire RNG-consumption
   sequence is unchanged (same batch size, same architecture, same call order). There is no dedicated
   `Generator` isolating the augmentation.

2. **No CUDA determinism flags.** Neither `torch.backends.cudnn.deterministic = True` nor
   `torch.use_deterministic_algorithms(True)` is set. So even at fixed seed, end-to-end GPU training
   is not bit-identical run-to-run (cuDNN convolution nondeterminism, atomic-add ordering). The
   augmentation *draws* are seeded; the resulting *checkpoint* is only approximately reproducible.

3. **Seed is per entry point.** `drift_augment` sets no seed of its own; each caller seeds. The
   alternate caller `exp3_cvnn_singlerun.py` seeds separately.

Accurate methods phrasing: the augmentation uses fixed seeds (torch/numpy seed 0), so the augmentation
policy is deterministic in intent; exact checkpoint reproduction additionally depends on unchanged
RNG-call order and is subject to CUDA nondeterminism, as no deterministic-algorithm flags are set.

Optional hardening (NOT applied): add the two cudnn/deterministic flags and pass explicit
`generator=` to both the DataLoader and the augmentation draws to isolate the augmentation stream
and make a re-fit bit-reproducible.
