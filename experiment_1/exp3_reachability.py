#!/usr/bin/env python3
"""exp3_reachability.py — reachable-set check for the 8/08 model.

Not just "did the targeted attack hit its target" (that is target_pct in the attack
matrix) but WHERE the victim actually LANDS. For each (attack, epsilon) we record the
full argmax histogram over the 5 classes, for:
  - untargeted from the victim device_5 (pushed off its own class)
  - targeted from device_5 toward each of the other 4 classes

Reachable set = the classes that ever take a plurality of the victim's windows.
Digital only — this is the perfect-delivery upper bound, no radios involved.
Labels: model classes are [device_1,device_2,device_3,device_4,device_5] (idx 0..4);
capture dir device_d maps to label d-1.  Victim = device_5 (label 4).
"""
import os as _os, sys as _sys
_R = _os.path.abspath(_os.path.dirname(__file__))
while _R != _os.path.dirname(_R) and not _os.path.isdir(_os.path.join(_R, ".git")):
    _R = _os.path.dirname(_R)
_sys.path.insert(0, _os.path.join(_R, "lib"))
_os.chdir(_R)
import numpy as np, torch, csv, time
from exp3_attack_matrix import (load_windows, craft, predict, to_reim_t,
                                ATTACKS, ber_proxy, recon)
EPSILONS = [0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
ATTACKS = {k: v for k, v in ATTACKS.items() if k in ('fgsm', 'bim', 'mim') or k.startswith('pgd')}
from exp3_cvnn import CVNN
from exp3_fp_model import NUM_CLASSES, DEVICE_NAMES
import os
CAPTURES = os.environ.get('CAPTURES_ROOT', os.path.expanduser('~/captures'))

MODEL = 'trained_models/fingerprint_cvnn.pt'
BASE = f'{CAPTURES}/enrollment'
DATA_DEVS = [1, 2, 3, 4, 5]
def lbl(d): return d - 1
VICTIM = 5
MAXW = 300
OUT = 'reachability_extgrid.csv'

# --- optional overrides so the same sweep can be run on a different epsilon grid ---
import argparse as _argparse
_ap = _argparse.ArgumentParser()
_ap.add_argument('--epsilons', help='comma-separated grid, overrides EPSILONS')
_ap.add_argument('--out', help='output csv, overrides OUT')
_ap.add_argument('--model', help='checkpoint, overrides MODEL')
_ap.add_argument('--base', help='capture root (received captures), overrides BASE')
_ap.add_argument('--max-windows', type=int, default=None, help='overrides MAXW')
_ap.add_argument('--ber-dp', type=int, default=5,
                 help='decimal places kept on the BER proxy. The default 5 floors anything '
                      'below 5e-06 to zero, which matters on a low-epsilon grid.')
_a = _ap.parse_args()
if _a.epsilons:
    EPSILONS = [float(x) for x in _a.epsilons.split(',')]
if _a.out:
    OUT = _a.out
if _a.model:
    MODEL = _a.model
if _a.base:
    BASE = _a.base
if _a.max_windows:
    MAXW = _a.max_windows
BER_DP = _a.ber_dp

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
m = CVNN(NUM_CLASSES).to(dev)
m.load_state_dict(torch.load(MODEL, map_location=dev, weights_only=False))
m.eval()
torch.manual_seed(42); np.random.seed(42)

reim, Wc = {}, {}
for d in DATA_DEVS:
    W = load_windows(BASE, d, MAXW); Wc[d] = W; reim[d] = to_reim_t(W, dev)
    pred = predict(m, *reim[d]).cpu().numpy()
    print(f"  device_{d}: {len(W)} win  clean_acc={(pred==lbl(d)).mean()*100:.1f}%", flush=True)


def hist(pred):
    """argmax histogram as a dict of class-name -> fraction."""
    n = len(pred)
    return {DEVICE_NAMES[c]: round(float((pred == c).mean()), 4) for c in range(NUM_CLASSES)}, n


rows = []
t0 = time.time(); npts = len(ATTACKS) * len(EPSILONS); cnt = 0
re_v, im_v = reim[VICTIM]

for attack, cfg in ATTACKS.items():
    for eps in EPSILONS:
        cnt += 1
        # --- untargeted: push the victim off device_5, see where it goes ---
        y = torch.full((re_v.shape[0],), lbl(VICTIM), dtype=torch.long, device=dev)
        dre, dim = craft(m, re_v, im_v, y, eps, cfg, +1.0, dev)
        pred = predict(m, re_v + dre, im_v + dim).cpu().numpy()
        h, n = hist(pred)
        fl, bt = ber_proxy(Wc[VICTIM], recon(re_v + dre, im_v + dim))
        rows.append(dict(attack=attack, mode='untargeted', epsilon=eps, intent='',
                         fool=round(float((pred != lbl(VICTIM)).mean()), 4),
                         landed=max(h, key=h.get), landed_pct=max(h.values()),
                         ber=round(fl / bt, BER_DP) if bt else '', flips=fl, bits=bt, n=n, **h))

        # --- targeted: aim at each other class, see where it ACTUALLY lands ---
        for td in DATA_DEVS:
            if td == VICTIM:
                continue
            y = torch.full((re_v.shape[0],), lbl(td), dtype=torch.long, device=dev)
            dre, dim = craft(m, re_v, im_v, y, eps, cfg, -1.0, dev)
            pred = predict(m, re_v + dre, im_v + dim).cpu().numpy()
            h, n = hist(pred)
            fl, bt = ber_proxy(Wc[VICTIM], recon(re_v + dre, im_v + dim))
            rows.append(dict(attack=attack, mode='targeted', epsilon=eps,
                             intent=DEVICE_NAMES[lbl(td)],
                             fool=round(float((pred != lbl(VICTIM)).mean()), 4),
                             landed=max(h, key=h.get), landed_pct=max(h.values()),
                             ber=round(fl / bt, BER_DP) if bt else '', flips=fl, bits=bt, n=n, **h))
        if cnt % 6 == 0 or cnt == npts:
            el = time.time() - t0
            print(f"[{cnt}/{npts}] {attack} eps={eps:.4f} "
                  f"({el/60:.1f}min, ~{el/cnt*(npts-cnt)/60:.1f} left)", flush=True)

cols = ['attack', 'mode', 'epsilon', 'intent', 'fool', 'landed', 'landed_pct', 'ber', 'flips', 'bits', 'n'] \
       + [DEVICE_NAMES[c] for c in range(NUM_CLASSES)]
with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
print(f"wrote {OUT}  ({len(rows)} rows)")

# ---- summary: the reachable set ----
print("\n=== UNTARGETED: where device_5 lands (plurality class per eps) ===")
for attack in ATTACKS:
    r = [x for x in rows if x['attack'] == attack and x['mode'] == 'untargeted']
    landed = sorted({x['landed'] for x in r if x['fool'] >= 0.5})
    print(f"  {attack:>10}: {landed if landed else 'never fooled'}")

print("\n=== TARGETED: intent vs actual landing (best eps per intent) ===")
for attack in ATTACKS:
    print(f"  {attack}")
    for intent in [DEVICE_NAMES[lbl(d)] for d in DATA_DEVS if d != VICTIM]:
        r = [x for x in rows if x['attack'] == attack and x['mode'] == 'targeted'
             and x['intent'] == intent]
        best = max(r, key=lambda x: x[intent])
        print(f"    intent {intent:>9}: best hit {best[intent]*100:5.1f}% @ eps={best['epsilon']:.4f}"
              f"   (that eps actually lands {best['landed']} {best['landed_pct']*100:.1f}%)")

reach = sorted({x['landed'] for x in rows if x['fool'] >= 0.5 and x['landed'] != 'device_5'})
print(f"\nREACHABLE SET from device_5 (digital upper bound): {reach}")
