# Trained models

Place the three trained classifiers here (they are gitignored — hosted externally, see main README §3):

- `fingerprint_cvnn.pt`     — the victim CVNN (Experiments 1–3, and the Exp 4 reference model)
- `fingerprint_cnn1d.pt`    — real-valued 1-D CNN variation (Exp 4, A1/A2)
- `fingerprint_freqcvnn.pt` — frequency-domain CVNN variation (Exp 4, A4)

Scripts load them from this folder relative to the repo root (e.g. `--model trained_models/fingerprint_cvnn.pt`),
so no path edits are needed as long as the `.pt` files sit here.
