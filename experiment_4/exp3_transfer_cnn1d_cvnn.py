#!/usr/bin/env python3
"""exp3_transfer_cnn1d_cvnn.py -- cross-model transferability (variation A2).

Craft the perturbation on one model's score, then evaluate the SAME perturbation on the other
model. Both models take the identical (re, im) input under the identical L_inf budget, so a delta
crafted on one applies directly to the other -- transfer is well-defined.

For each source device we pool its windows with true labels, and at each eps craft the six-attack
ensemble (FGSM/BIM/MIM/PGDx3, untargeted, ascend true class) on the SOURCE model. The union delta
(a window is fooled if ANY attack fools it) is then scored on BOTH models:
  - on the source model    = white-box fooling
  - on the other model     = transferred fooling (same delta, different classifier)
Done in both directions (craft on CVNN, craft on CNN1D), giving the 2x2:

            eval CVNN            eval CNN1D
 craft CVNN   white-box(CVNN)     transfer(CVNN->CNN1D)
 craft CNN1D  transfer(CNN1D->CVNN) white-box(CNN1D)

Report fooling-rate-vs-eps curves and eps_50 for all four, and the transfer ratio (docx A2)
  tau_target = eps_50(white-box on target) / eps_50(delta transferred onto target)
tau ~ 1: transfer is nearly free; tau = 0.2: the transferred attack needs 5x the budget.
"""
import os as _os, sys as _sys
_R = _os.path.abspath(_os.path.dirname(__file__))
while _R != _os.path.dirname(_R) and not _os.path.isdir(_os.path.join(_R, ".git")):
    _R = _os.path.dirname(_R)
_sys.path.insert(0, _os.path.join(_R, "lib"))
_os.chdir(_R)
import argparse, csv
import numpy as np, torch
from cvnn import CVNN
from cnn1d import CNN1DFp
from fp_model import NUM_CLASSES
from attack_matrix import load_windows, to_reim_t, craft, predict, ATTACKS
import os
CAPTURES = os.environ.get('CAPTURES_ROOT', os.path.expanduser('~/captures'))

ATK = ['fgsm', 'bim', 'mim', 'pgd_a0.01', 'pgd_a0.05', 'pgd_a0.10']
DEVS = [1, 2, 3, 4, 5]


def load_model(cls, path, dev):
    m = cls(NUM_CLASSES).to(dev)
    m.load_state_dict(torch.load(path, map_location=dev, weights_only=False)); m.eval()
    return m


def interp_cross(grid, frac, thr=0.5):
    for k in range(len(grid)):
        if frac[k] >= thr:
            if k == 0:
                return float(grid[0])
            f0, f1 = frac[k - 1], frac[k]
            return float(grid[k - 1] + (thr - f0) / (f1 - f0) * (grid[k] - grid[k - 1])) if f1 != f0 else float(grid[k])
    return float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cvnn', default='trained_models/fingerprint_cvnn.pt')
    ap.add_argument('--cnn1d', default='trained_models/fingerprint_cnn1d.pt')
    ap.add_argument('--base', default=f'{CAPTURES}/enrollment')
    ap.add_argument('--maxw', type=int, default=300)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0); np.random.seed(0)
    models = {'cvnn': load_model(CVNN, a.cvnn, dev), 'cnn1d': load_model(CNN1DFp, a.cnn1d, dev)}

    # pool all devices' windows with true labels
    RE, IM, Y = [], [], []
    for d in DEVS:
        re, im = to_reim_t(load_windows(a.base, d, a.maxw), dev)
        RE.append(re); IM.append(im)
        Y.append(torch.full((re.shape[0],), d - 1, dtype=torch.long, device=dev))
    RE = torch.cat(RE); IM = torch.cat(IM); Y = torch.cat(Y)
    n = RE.shape[0]
    # clean self-accuracy sanity
    for name, m in models.items():
        acc = (predict(m, RE, IM) == Y).float().mean().item()
        print(f"clean self-accuracy {name}: {acc:.2%}  (n={n})")

    grid = np.round(np.arange(0.002, 0.0801, 0.002), 3)
    # curves[(src,eval)] = fooling rate vs eps
    curves = {(s, e): np.zeros(len(grid)) for s in models for e in models}
    print("\ncrafting union delta on each source, scoring on both models...")
    for src in models:
        for gk, eps in enumerate(grid):
            union = {e: np.zeros(n, dtype=bool) for e in models}
            for atk in ATK:
                dre, dim = craft(models[src], RE, IM, Y, float(eps), ATTACKS[atk], +1.0, dev)
                for e in models:
                    off = (predict(models[e], RE + dre, IM + dim).cpu().numpy() != Y.cpu().numpy())
                    union[e] |= off
            for e in models:
                curves[(src, e)][gk] = union[e].mean()
        print(f"  crafted on {src}: done")

    # eps_50 for the four curves
    e50 = {k: interp_cross(grid, v, 0.5) for k, v in curves.items()}
    print("\n=== eps_50 (fooling reaches 50%) ===")
    print(f"{'':14}{'eval CVNN':>12}{'eval CNN1D':>12}")
    for s in ['cvnn', 'cnn1d']:
        print(f"craft {s:8}" + "".join(f"{e50[(s,e)]:>12.4f}" for e in ['cvnn', 'cnn1d']))

    # transfer ratios: tau_target = eps50(white-box target) / eps50(transfer onto target)
    tau_cvnn = e50[('cvnn', 'cvnn')] / e50[('cnn1d', 'cvnn')]
    tau_cnn1d = e50[('cnn1d', 'cnn1d')] / e50[('cvnn', 'cnn1d')]
    print("\n=== transfer ratio tau = eps50(white-box) / eps50(transferred) ===")
    print(f"  onto CVNN  (craft CNN1D -> CVNN):  tau = {tau_cvnn:.3f}")
    print(f"  onto CNN1D (craft CVNN  -> CNN1D): tau = {tau_cnn1d:.3f}")
    print("  tau~1: transfer nearly free;  tau<1: transferred attack needs 1/tau x the budget")

    np.savez('docs/exp3_transfer_cnn1d_cvnn.npz', grid=grid,
             **{f'{s}_on_{e}': curves[(s, e)] for s in models for e in models},
             e50=np.array([[e50[(s, e)] for e in ['cvnn', 'cnn1d']] for s in ['cvnn', 'cnn1d']]),
             tau_cvnn=tau_cvnn, tau_cnn1d=tau_cnn1d)
    with open('docs/exp3_transfer_cnn1d_cvnn.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['eps'] + [f'craft_{s}_eval_{e}' for s in ['cvnn', 'cnn1d'] for e in ['cvnn', 'cnn1d']])
        for gk, eps in enumerate(grid):
            w.writerow([f'{eps:.3f}'] + [f'{curves[(s,e)][gk]:.4f}'
                       for s in ['cvnn', 'cnn1d'] for e in ['cvnn', 'cnn1d']])
    print("\nsaved exp3_transfer_cnn1d_cvnn.csv + .npz")


if __name__ == '__main__':
    main()
