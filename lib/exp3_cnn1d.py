#!/usr/bin/env python3
"""exp3_cnn1d.py -- real-valued 1D-CNN fingerprint classifier (variation A1).

Real-valued control for the CVNN. Mirrors the CVNN layer structure EXACTLY -- same channel
progression (32/64/128/128), same kernels (7/5/5/3), same length-2 pooling, same real head
(128->128->nc) -- but with ordinary real Conv1d over a 2-channel (real, imag) input instead of
complex convolutions. The ONLY architectural difference is real vs complex, so re-running the
reachability matrix and attack sweep on this model isolates whether the fragility (decision
boundaries lying well inside the physical inter-device distance) is a property of the complex
architecture or of the fingerprinting task itself.

forward(re, im) matches the CVNN interface: re, im are (B,1,L) real; internally stacked to
(B,2,L). This lets the existing craft/predict/targeted-PGD/reachability code (which puts the
L_inf ball on the (re,im) representation and calls model(re,im)) run UNCHANGED -- an
apples-to-apples comparison under the identical input convention, attacks, and budget.

Training reuses the CVNN pipeline verbatim: same build() data loader, same drift_augment (random
global phase, CFO +/-3000 Hz, +/-6-sample circular shift, AWGN 8-30 dB), same AdamW/cosine/
label-smoothing schedule, val_run held out. Train on the same cleaned symlink root as the CVNN.
"""
import argparse
import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from exp3_fp_model import NUM_CLASSES
from exp3_cvnn import build, drift_augment, per_device_eval, n_params, to_reim


class CNN1DFp(nn.Module):
    """Real-valued analog of the CVNN (same depth/width/kernels/pooling/head)."""
    def __init__(self, nc=NUM_CLASSES):
        super().__init__()
        chans = [(2, 32, 7), (32, 64, 5), (64, 128, 5), (128, 128, 3)]
        blocks = []
        for ci, co, k in chans:
            blocks.append(nn.Sequential(
                nn.Conv1d(ci, co, k, padding=k // 2),
                nn.BatchNorm1d(co),
                nn.ReLU(inplace=True),
                nn.AvgPool1d(2)))
        self.features = nn.ModuleList(blocks)
        self.head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, nc))

    def forward(self, re, im):                       # re, im: (B,1,L)
        x = torch.cat([re, im], dim=1)               # (B,2,L)
        for blk in self.features:
            x = blk(x)
        z = torch.adaptive_avg_pool1d(x, 1).flatten(1)
        return self.head(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='cleaned symlink train root (clean_run_* only)')
    ap.add_argument('--val-run', type=int, default=3)
    ap.add_argument('--skip-run', type=int, default=None)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--lr', type=float, default=6e-4)
    ap.add_argument('--hop-train', type=int, default=384)
    ap.add_argument('--thr-mult', type=float, default=2.0)
    ap.add_argument('--out', default='trained_models/fingerprint_cnn1d.pt')
    ap.add_argument('--no-aug', action='store_true')
    ap.add_argument('--cfo-max', type=float, default=3000.0)
    ap.add_argument('--snr-lo', type=float, default=8.0)
    ap.add_argument('--snr-hi', type=float, default=30.0)
    ap.add_argument('--cache', default='.cnn1d_cache/train.npz')
    ap.add_argument('--rebuild', action='store_true')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0); np.random.seed(0)

    import os
    os.makedirs(os.path.dirname(a.cache), exist_ok=True)
    Xtr, ytr, Xva, yva, vfid, counts = build(a.root, a.val_run, a.hop_train, a.thr_mult,
                                             a.cache, a.rebuild, a.skip_run, False)
    print(f"  frames/device: {counts}\n  train {Xtr.shape}  val {Xva.shape}")

    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=256, shuffle=True)
    m = CNN1DFp(NUM_CLASSES).to(dev)
    print(f"  CNN1D params: {n_params(m):,}")
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    best, best_state = 0.0, None
    for ep in range(1, a.epochs + 1):
        m.train(); tot = nb = 0
        for xb, yb in dl:
            xb = xb.to(dev)
            if not a.no_aug:
                xb = drift_augment(xb, a.cfo_max, a.snr_lo, a.snr_hi)
            re, im = xb.real.unsqueeze(1), xb.imag.unsqueeze(1); yb = yb.to(dev)
            opt.zero_grad(); loss = lossf(m(re, im), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step(); tot += loss.item(); nb += 1
        sched.step()
        w, f, _, _ = per_device_eval(m, dev, Xva, yva, vfid, NUM_CLASSES)
        if ep % 5 == 0 or ep == 1:
            print(f"  epoch {ep:2d}  loss {tot/nb:.4f}  val_win {w:.4f}  val_frame {f:.4f}", flush=True)
        if w > best:
            best = w; best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state)
    w, f, per, C = per_device_eval(m, dev, Xva, yva, vfid, NUM_CLASSES)
    print(f"\n  BEST CNN1D  val_win {w:.4f}  val_frame {f:.4f}")
    print("  per-device frame acc:", {k: f"{v[0]}/{v[1]}" for k, v in per.items()})
    torch.save(m.state_dict(), a.out)
    print(f"  saved {a.out}")


if __name__ == '__main__':
    main()
