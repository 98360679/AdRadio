#!/usr/bin/env python3
"""validation_confusion.py — clean-classifier validation accuracy + confusion.

Reproduces the victim classifier's baseline (pre-attack) accuracy reported in the
paper: the held-out run-3 validation of the CVNN, under the leakage-safe split
(runs 1-2 train, run 3 validate). Each 802.11 frame is classified by majority vote
over its 13 I/Q frames (1024-sample windows); we report the frame-level accuracy
(the classifier's actual decision) and the per-I/Q-frame (window-level) accuracy.

  python3 validation_confusion.py            # defaults: trained_models/fingerprint_cvnn.pt, run 3
  python3 validation_confusion.py --model trained_models/fingerprint_cvnn.pt \
          --base $CAPTURES_ROOT/enrollment --run 3 --out docs/validation_confusion.csv

Prints the confusion matrix and writes it (frame- and window-level) to --out.
"""
import os as _os, sys as _sys
_R = _os.path.abspath(_os.path.dirname(__file__))
while _R != _os.path.dirname(_R) and not _os.path.isdir(_os.path.join(_R, ".git")):
    _R = _os.path.dirname(_R)
_sys.path.insert(0, _os.path.join(_R, "lib"))
_os.chdir(_R)
import os, csv, argparse, numpy as np, torch
from cvnn import CVNN, iq_to_complex
from fp_model import frame_to_windows, NUM_CLASSES, WIN, DEVICE_NAMES
from attack_matrix import to_reim_t, predict
from extract_frames import extract_frames_for_file

CAPTURES = os.environ.get('CAPTURES_ROOT', os.path.expanduser('~/captures'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='trained_models/fingerprint_cvnn.pt')
    ap.add_argument('--base', default=f'{CAPTURES}/enrollment',
                    help='enrollment root with device_<N>/clean_run_<run>.bin')
    ap.add_argument('--run', type=int, default=3, help='held-out validation run (default 3)')
    ap.add_argument('--out', default='docs/validation_confusion.csv')
    a = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = CVNN(NUM_CLASSES).to(dev)
    m.load_state_dict(torch.load(a.model, map_location=dev, weights_only=False)); m.eval()
    names = DEVICE_NAMES
    Cf = np.zeros((NUM_CLASSES, NUM_CLASSES), int)   # frame-level (majority vote)
    Cw = np.zeros((NUM_CLASSES, NUM_CLASSES), int)   # window / I/Q-frame level
    for ti, name in enumerate(names):
        frames = extract_frames_for_file(f'{a.base}/{name}/clean_run_{a.run}.bin',
                                         floor_pct=20.0, thr_mult=2.0)[0]
        for fr in frames:
            W = iq_to_complex(frame_to_windows(fr, hop=WIN), cfo_correct=False)
            re, im = to_reim_t(W, dev); p = predict(m, re, im).cpu().numpy()
            for pw in p:
                Cw[ti, pw] += 1
            Cf[ti, np.bincount(p, minlength=NUM_CLASSES).argmax()] += 1

    accf = np.trace(Cf) / max(1, Cf.sum()); accw = np.trace(Cw) / max(1, Cw.sum())
    print(f"=== VALIDATION (held-out run {a.run}) — {os.path.basename(a.model)} ===")
    hdr = "  true\\pred  " + "  ".join(f"{n.replace('device_','d'):>5}" for n in names) + "    n    acc%"
    print("frame-level (majority vote over 13 I/Q frames):"); print(hdr)
    for i, n in enumerate(names):
        tot = Cf[i].sum()
        print(f"  {n:<9}" + "  ".join(f"{Cf[i,j]:5d}" for j in range(NUM_CLASSES))
              + f"  {tot:4d}  {100*Cf[i,i]/max(1,tot):5.1f}")
    print(f"  FRAME-level accuracy  = {100*accf:.2f}%  ({Cf.sum()} frames)")
    print(f"  WINDOW-level accuracy = {100*accw:.2f}%  ({Cw.sum()} I/Q frames)")

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([f'# held-out run-{a.run} validation confusion for {os.path.basename(a.model)} '
                    f'(frame-level overall {100*accf:.2f}%, window-level {100*accw:.2f}%)'])
        for lvl, C, acc in (('frame', Cf, accf), ('window', Cw, accw)):
            w.writerow([f'# {lvl}-level'])
            w.writerow(['true\\pred'] + names + ['total', 'correct', 'accuracy'])
            for i, n in enumerate(names):
                w.writerow([n] + list(C[i]) + [int(C[i].sum()), int(C[i, i]),
                                               f'{C[i,i]/max(1,C[i].sum()):.4f}'])
            w.writerow(['TOTAL'] + [int(C[:, j].sum()) for j in range(NUM_CLASSES)]
                       + [int(C.sum()), int(np.trace(C)), f'{acc:.4f}'])
    print(f"wrote {a.out}")


if __name__ == '__main__':
    main()
