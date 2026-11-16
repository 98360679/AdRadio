#!/usr/bin/env python3
"""exp3_freqcvnn.py -- frequency-domain complex CNN fingerprint classifier (variation A4).

Same complex-valued CNN as the time-domain model, but it classifies the WINDOW'S SPECTRUM: the
first layer is an FFT. forward(re, im) still takes the time-domain (re, im) window, computes
FFT(re + j*im), unit-RMS-normalizes the spectrum, and feeds its (real, imag) to the identical
CVNN conv stack. Because the input convention and the L_inf budget stay in the time domain, the
existing craft / targeted-PGD / reachability code runs UNCHANGED (the FFT is just a differentiable
first layer) -- so A4 is directly comparable to A1: does the fingerprint and its fragility survive
a spectral representation?

Training reuses the CVNN pipeline verbatim (build split, drift_augment on the TIME-domain window
before the FFT, same 80-epoch AdamW/cosine schedule, run 3 held out).
"""
import argparse
import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from exp3_fp_model import NUM_CLASSES
from exp3_cvnn import CVNN, build, drift_augment, per_device_eval, n_params


class FreqCVNN(nn.Module):
    """CVNN on the window spectrum: FFT first, then the identical complex conv stack."""
    def __init__(self, nc=NUM_CLASSES):
        super().__init__()
        self.core = CVNN(nc)

    def forward(self, re, im):                       # re, im: (B,1,L) time-domain
        w = torch.complex(re, im)                    # (B,1,L)
        S = torch.fft.fft(w, dim=-1)                 # window spectrum
        rms = torch.sqrt((S.real ** 2 + S.imag ** 2).mean(-1, keepdim=True) + 1e-12)
        S = S / rms                                  # unit-RMS spectrum (match time-model input scale)
        return self.core(S.real, S.imag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--val-run', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--lr', type=float, default=6e-4)
    ap.add_argument('--hop-train', type=int, default=384)
    ap.add_argument('--thr-mult', type=float, default=2.0)
    ap.add_argument('--out', default='fingerprint_freqcvnn.pt')
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
                                             a.cache, a.rebuild, None, False)
    print(f"  frames/device: {counts}\n  train {Xtr.shape}  val {Xva.shape}")

    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=256, shuffle=True)
    m = FreqCVNN(NUM_CLASSES).to(dev)
    print(f"  FreqCVNN params: {n_params(m):,}")
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
    print(f"\n  BEST FreqCVNN  val_win {w:.4f}  val_frame {f:.4f}")
    print("  per-device frame acc:", {k: f"{v[0]}/{v[1]}" for k, v in per.items()})
    torch.save(m.state_dict(), a.out)
    print(f"  saved {a.out}")


if __name__ == '__main__':
    main()
