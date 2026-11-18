#!/usr/bin/env python3
"""B3 — frame-level objective vs per-window (DIGITAL, reuses E1 enrollment windows).

Baseline (per-window): each of the 13 payload windows is optimized independently (F.cross_entropy on
per-window logits) -- the current attack. The frame decision is a downstream majority vote (>=7/13).
Frame-level: ONE delta over the frame's 13 windows optimized against the frame decision itself, using
the mean-logit soft-vote (mean over the 13 windows' logits) -- so budget concentrates on the windows
that move the vote and gives up on the rest once >=7 flip.

Measures FRAME-fooling (majority vote != victim, untargeted; == target, targeted) vs eps and reports
eps_50 for both objectives, all 6 methods. Prediction: frame-level lowers eps_50.
"""
import argparse, csv
import numpy as np, torch
import torch.nn.functional as F
from exp3_extract_frames import extract_frames_for_file
from exp3_fp_model import NUM_CLASSES, DEVICE_NAMES, WIN, PRE_ROLL, ACTIVE
from exp3_cvnn import CVNN, iq_to_complex, to_reim
from exp3_attack_matrix import ATTACKS
import os
CAPTURES = os.environ.get('CAPTURES_ROOT', os.path.expanduser('~/captures'))

STARTS = list(range(PRE_ROLL, PRE_ROLL + ACTIVE - WIN + 1, WIN))   # 13 windows/frame
NWIN = len(STARTS)


def load_frames(root, device, max_frames):
    import glob, os
    files = sorted(glob.glob(os.path.join(root, f'device_{device}', 'clean_run_*.bin')))
    out = []
    for f in files:
        for fr in extract_frames_for_file(f, floor_pct=20.0, thr_mult=2.0)[0]:
            w = iq_to_complex(np.stack([fr[s:s + WIN] for s in STARTS]), cfo_correct=False)  # (13,WIN)
            out.append(np.asarray(w, np.complex64))
            if len(out) >= max_frames:
                return np.stack(out)                     # (F,13,WIN)
    return np.stack(out) if out else None


def _alpha(cfg, eps):
    return eps if cfg['alpha'] == 'eps' else (eps/cfg['steps'] if cfg['alpha'] == 'eps/steps' else float(cfg['alpha']))


def craft(re, im, y, eps, cfg, sign, model, dev, frame_level, mask):
    """re,im: (F,13,1,WIN). y: (F,) frame labels. frame_level: mean-logit frame loss vs per-window.
    mask: (13,1,WIN) 0/1 or None (window 0 held at zero). Returns dre,dim (F,13,1,WIN)."""
    Fn = re.shape[0]; steps = cfg['steps']; alpha = _alpha(cfg, eps); mu = cfg['momentum']
    if cfg['random_start']:
        dre = (torch.rand_like(re)*2-1)*eps; dim = (torch.rand_like(im)*2-1)*eps
    else:
        dre = torch.zeros_like(re); dim = torch.zeros_like(im)
    if mask is not None:
        dre = dre*mask; dim = dim*mask
    gre = torch.zeros_like(re); gim = torch.zeros_like(im)
    for _ in range(steps):
        dre.requires_grad_(True); dim.requires_grad_(True)
        RE = (re+dre).reshape(Fn*NWIN, 1, WIN); IM = (im+dim).reshape(Fn*NWIN, 1, WIN)
        logits = model(RE, IM).reshape(Fn, NWIN, NUM_CLASSES)         # (F,13,5)
        if frame_level:
            loss = F.cross_entropy(logits.mean(1), y)                 # soft-vote frame loss
        else:
            loss = F.cross_entropy(logits.reshape(Fn*NWIN, NUM_CLASSES),
                                   y.repeat_interleave(NWIN))         # per-window loss
        g_re, g_im = torch.autograd.grad(loss, [dre, dim])
        with torch.no_grad():
            if mu is not None:
                l1 = (g_re.abs().flatten(1).sum(1)+g_im.abs().flatten(1).sum(1)).clamp_min(1e-12).view(-1,1,1,1)
                gre = mu*gre + g_re/l1; gim = mu*gim + g_im/l1
                s_re, s_im = gre.sign(), gim.sign()
            else:
                s_re, s_im = g_re.sign(), g_im.sign()
            dre = (dre.detach()+sign*alpha*s_re).clamp(-eps, eps)
            dim = (dim.detach()+sign*alpha*s_im).clamp(-eps, eps)
            if mask is not None:
                dre = dre*mask; dim = dim*mask
    return dre.detach(), dim.detach()


@torch.no_grad()
def frame_vote(model, re, im):
    """re,im: (F,13,1,WIN) -> per-frame majority-vote class (F,)."""
    Fn = re.shape[0]
    votes = model(re.reshape(Fn*NWIN, 1, WIN), im.reshape(Fn*NWIN, 1, WIN)).argmax(1).reshape(Fn, NWIN)
    out = []
    for v in votes:
        out.append(int(torch.bincount(v, minlength=NUM_CLASSES).argmax()))
    return torch.tensor(out, device=re.device)


def eps50(eps, met):
    eps, met = np.array(eps), np.array(met)
    for i in range(1, len(eps)):
        if met[i-1] < 50 <= met[i]:
            return round(float(eps[i-1]+(50-met[i-1])*(eps[i]-eps[i-1])/(met[i]-met[i-1])), 4)
    return (float(eps[0]) if len(met) and met[0] >= 50 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=f'{CAPTURES}/enrollment')
    ap.add_argument('--model', default='fingerprint_cvnn.pt')
    ap.add_argument('--victim', type=int, default=6)             # device number
    ap.add_argument('--tgt-dst', type=int, default=4)            # targeted -> device_3
    ap.add_argument('--max-frames', type=int, default=120)
    ap.add_argument('--mask-lead', type=int, default=1, help='mask N leading windows (MAC header). 0=none')
    ap.add_argument('--epsilons', default=None)
    ap.add_argument('--out', default='docs/b3_frame_vs_window.csv')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    vcls = DEVICE_NAMES.index(f'device_{a.victim}')
    tcls = DEVICE_NAMES.index(f'device_{a.tgt_dst}')
    m = CVNN(NUM_CLASSES).to(dev); m.load_state_dict(torch.load(a.model, map_location=dev, weights_only=False)); m.eval()

    fr = load_frames(a.root, a.victim, a.max_frames)
    W = torch.from_numpy(fr)                                     # (F,13,WIN) complex
    re = W.real.unsqueeze(2).float().to(dev); im = W.imag.unsqueeze(2).float().to(dev)   # (F,13,1,WIN)
    Fn = re.shape[0]
    clean = frame_vote(m, re, im)
    print(f"loaded {Fn} victim frames; clean frame-vote == device_{a.victim}: {(clean==vcls).float().mean()*100:.1f}%")

    mask = None
    if a.mask_lead > 0:
        mk = torch.ones(NWIN, 1, WIN, device=dev); mk[:a.mask_lead] = 0.0
        mask = mk
        print(f"masking {a.mask_lead} lead window(s) (MAC header)")

    eps_grid = [float(x) for x in a.epsilons.split(',')] if a.epsilons else \
        [0.005,0.01,0.015,0.02,0.025,0.03,0.04,0.05,0.06,0.08,0.10,0.13,0.16,0.20]
    modes = [('untargeted', vcls, +1.0), ('targeted', tcls, -1.0)]

    rows = []; summ = []
    for mode, ycls, sign in modes:
        y = torch.full((Fn,), ycls, dtype=torch.long, device=dev)
        for meth, cfg in ATTACKS.items():
            fool = {'window': [], 'frame': []}
            for eps in eps_grid:
                for objn, fl in [('window', False), ('frame', True)]:
                    dre, dim = craft(re, im, y, eps, cfg, sign, m, dev, fl, mask)
                    pv = frame_vote(m, re+dre, im+dim)
                    if mode == 'untargeted':
                        f = (pv != vcls).float().mean().item()*100
                    else:
                        f = (pv == tcls).float().mean().item()*100
                    fool[objn].append(f)
                    rows.append([meth, mode, objn, eps, round(f, 2)])
            ew, ef = eps50(eps_grid, fool['window']), eps50(eps_grid, fool['frame'])
            summ.append((meth, mode, ew, ef))
            print(f"{meth:11} {mode:11} eps50 window={ew} frame={ef} "
                  f"gain={'--' if not (ew and ef) else f'{ew/ef:.2f}x'}")

    with open(a.out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['method', 'mode', 'objective', 'eps', 'frame_fool_pct']); w.writerows(rows)
    print(f"\nwrote {a.out}")
    print("\n=== eps_50 (window -> frame, efficiency gain) ===")
    for meth, mode, ew, ef in summ:
        g = f"{ew/ef:.2f}x" if (ew and ef) else '--'
        print(f"  {meth:11} {mode:11} {str(ew):>8} -> {str(ef):>8}  ({g})")


if __name__ == '__main__':
    main()
