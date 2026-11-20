#!/usr/bin/env python3
"""exp3_crossday_eval.py — evaluate a fingerprint model on a DIFFERENT day's captures.

The cross-day question: a model fine-tuned on day A's enrollment, how well does it still
identify the same five physical radios on day B? Reports per-device window accuracy, the
full confusion matrix, and the same numbers on the training day for reference, so drift is
read as a DIFFERENCE rather than an absolute.

  python3 exp3_crossday_eval.py \
      --model trained_models/fingerprint_cvnn.pt \
      --test  $CAPTURES_ROOT/validation \
      --ref   $CAPTURES_ROOT/enrollment

⚠ Read per-device accuracy next to the capture LEVEL. A device whose signal level differs
between the two days confounds fingerprint drift with SNR — see the level table in
exp3-0812-crossday-validation-capture.
"""
import os as _os, sys as _sys
_R = _os.path.abspath(_os.path.dirname(__file__))
while _R != _os.path.dirname(_R) and not _os.path.isdir(_os.path.join(_R, ".git")):
    _R = _os.path.dirname(_R)
_sys.path.insert(0, _os.path.join(_R, "lib"))
_os.chdir(_R)
import argparse, csv, os
import numpy as np, torch
from exp3_attack_matrix import load_windows, predict, to_reim_t
from exp3_cvnn import CVNN
from exp3_fp_model import NUM_CLASSES, DEVICE_NAMES

DEVS = [1, 2, 3, 4, 5]
def lbl(d): return d - 1


def evaluate(m, root, dev, maxw):
    """per-device: (n_windows, accuracy, histogram over predicted classes)"""
    out = {}
    for d in DEVS:
        W = load_windows(root, d, maxw)
        if W is None:
            out[d] = (0, None, None)
            continue
        re_t, im_t = to_reim_t(W, dev)
        pred = predict(m, re_t, im_t).cpu().numpy()
        hist = np.array([(pred == c).sum() for c in range(NUM_CLASSES)])
        out[d] = (len(pred), float((pred == lbl(d)).mean()), hist)
    return out


def show(tag, res):
    print(f'\n=== {tag} ===')
    print(f'  {"device":<10} {"n":>6} {"acc %":>8}   ' +
          '  '.join(f'{n.replace("device_","d"):>5}' for n in DEVICE_NAMES))
    accs = []
    for d in DEVS:
        n, acc, hist = res[d]
        if not n:
            print(f'  device_{d:<3} {"":>6}   NO DATA'); continue
        accs.append(acc)
        bar = '  '.join(f'{100*h/n:5.1f}' for h in hist)
        print(f'  device_{d:<3} {n:>6} {100*acc:8.1f}   {bar}')
    if accs:
        print(f'  {"MEAN":<10} {"":>6} {100*np.mean(accs):8.1f}')
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--test', required=True, help='capture root for the TEST day')
    ap.add_argument('--ref', default=None, help='capture root for the TRAINING day (reference)')
    ap.add_argument('--max-windows', type=int, default=300)
    ap.add_argument('--out', default=None, help='write per-device rows to this csv')
    a = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = CVNN(NUM_CLASSES).to(dev)
    m.load_state_dict(torch.load(a.model, map_location=dev, weights_only=False)); m.eval()
    torch.manual_seed(42); np.random.seed(42)
    print(f'model {a.model}  classes {DEVICE_NAMES}  device {dev}  max_windows {a.max_windows}')

    res_t = evaluate(m, a.test, dev, a.max_windows)
    at = show(f'TEST DAY  {a.test}', res_t)
    ar = None
    if a.ref:
        res_r = evaluate(m, a.ref, dev, a.max_windows)
        ar = show(f'TRAINING DAY  {a.ref}', res_r)
        print('\n=== cross-day drop, per device (training day minus test day) ===')
        for d in DEVS:
            nt, act, _ = res_t[d]; nr, acr, _ = res_r[d]
            if nt and nr:
                print(f'  device_{d}: {100*acr:6.1f}  ->  {100*act:6.1f}   '
                      f'{100*(act-acr):+6.1f} pts')
        if at and ar:
            print(f'  MEAN     : {100*np.mean(ar):6.1f}  ->  {100*np.mean(at):6.1f}   '
                  f'{100*(np.mean(at)-np.mean(ar)):+6.1f} pts')

    if a.out:
        with open(a.out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['day', 'device', 'n_windows', 'accuracy_pct']
                       + [f'pred_{n}' for n in DEVICE_NAMES])
            for tag, res in [('test', res_t)] + ([('ref', res_r)] if a.ref else []):
                for d in DEVS:
                    n, acc, hist = res[d]
                    if n:
                        w.writerow([tag, f'device_{d}', n, round(100*acc, 2)]
                                   + [int(x) for x in hist])
        print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
