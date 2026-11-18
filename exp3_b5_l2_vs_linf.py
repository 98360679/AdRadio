#!/usr/bin/env python3
"""B5 — L2 vs L-infinity budget (DIGITAL), and B3 revisited under L2.

L-inf bounds every SAMPLE (|delta_i|<=eps); L2 bounds the total ENERGY (||delta||_2<=eps). Sadeghi
and the Adesina survey argue L2 is the natural wireless norm (energy is what the amplifier spends).

Two questions:
  (1) L2 vs L-inf efficiency: which reaches frame-fooling at lower delta ENERGY (per-frame RMS)?
  (2) B3 under L2: does the frame-level (soft-vote) objective now BEAT per-window? B3 showed no gain
      under L-inf because per-sample budget can't be concentrated; L2 shares one energy pool across
      the 13 windows, so concentrating on the cheapest ~7 should finally lower the energy needed.

Both norms crafted for {per-window, frame-level} objectives; fooling reported vs realized delta RMS
so the two norms are compared on a common energy axis. Reuses B3's frame machinery.
"""
import argparse, csv
import numpy as np, torch
import torch.nn.functional as F
from exp3_fp_model import NUM_CLASSES, DEVICE_NAMES, WIN
from exp3_cvnn import CVNN
from exp3_attack_matrix import ATTACKS
from exp3_b3_frame_objective import load_frames, frame_vote, NWIN
import os
CAPTURES = os.environ.get('CAPTURES_ROOT', os.path.expanduser('~/captures'))


def craft(re, im, y, radius, cfg, sign, model, dev, frame_level, mask, norm):
    """L-inf or L2 constrained delta over each frame's 13 windows. re,im: (F,13,1,WIN)."""
    Fn = re.shape[0]; steps = cfg['steps']; mu = cfg['momentum']
    if norm == 'linf':
        alpha = radius if cfg['alpha'] == 'eps' else (radius/steps if cfg['alpha'] == 'eps/steps' else float(cfg['alpha']))
    else:  # L2 step: whole-radius for single step, else standard 2.5*eps/steps
        alpha = radius if steps == 1 else 2.5 * radius / steps

    def project(dre, dim):
        if norm == 'linf':
            return dre.clamp(-radius, radius), dim.clamp(-radius, radius)
        # L2 ball over the full per-frame delta (re & im together), per frame
        n = torch.sqrt((dre**2 + dim**2).flatten(1).sum(1)).clamp_min(1e-12)     # (F,)
        f = (radius / n).clamp(max=1.0).view(-1, 1, 1, 1)
        return dre * f, dim * f

    if cfg['random_start']:
        dre = (torch.rand_like(re)*2-1)*radius; dim = (torch.rand_like(im)*2-1)*radius
    else:
        dre = torch.zeros_like(re); dim = torch.zeros_like(im)
    if mask is not None: dre, dim = dre*mask, dim*mask
    dre, dim = project(dre, dim)
    gre = torch.zeros_like(re); gim = torch.zeros_like(im)
    for _ in range(steps):
        dre.requires_grad_(True); dim.requires_grad_(True)
        RE = (re+dre).reshape(Fn*NWIN, 1, WIN); IM = (im+dim).reshape(Fn*NWIN, 1, WIN)
        logits = model(RE, IM).reshape(Fn, NWIN, NUM_CLASSES)
        loss = F.cross_entropy(logits.mean(1), y) if frame_level else \
               F.cross_entropy(logits.reshape(Fn*NWIN, NUM_CLASSES), y.repeat_interleave(NWIN))
        g_re, g_im = torch.autograd.grad(loss, [dre, dim])
        with torch.no_grad():
            if norm == 'linf':
                if mu is not None:
                    l1 = (g_re.abs().flatten(1).sum(1)+g_im.abs().flatten(1).sum(1)).clamp_min(1e-12).view(-1,1,1,1)
                    gre = mu*gre + g_re/l1; gim = mu*gim + g_im/l1
                    s_re, s_im = gre.sign(), gim.sign()
                else:
                    s_re, s_im = g_re.sign(), g_im.sign()
                dre = dre.detach()+sign*alpha*s_re; dim = dim.detach()+sign*alpha*s_im
            else:  # L2: step along the unit-normalized gradient (per frame)
                gn = torch.sqrt((g_re**2+g_im**2).flatten(1).sum(1)).clamp_min(1e-12).view(-1,1,1,1)
                if mu is not None:
                    gre = mu*gre + g_re/gn; gim = mu*gim + g_im/gn
                    hn = torch.sqrt((gre**2+gim**2).flatten(1).sum(1)).clamp_min(1e-12).view(-1,1,1,1)
                    ur, ui = gre/hn, gim/hn
                else:
                    ur, ui = g_re/gn, g_im/gn
                dre = dre.detach()+sign*alpha*ur; dim = dim.detach()+sign*alpha*ui
            if mask is not None: dre, dim = dre*mask, dim*mask
            dre, dim = project(dre, dim)
    return dre.detach(), dim.detach()


def rms(dre, dim, mask):
    """per-frame delta RMS over the UNMASKED payload samples (the energy metric)."""
    e = (dre**2 + dim**2)
    if mask is not None:
        e = e * mask
        denom = mask.sum() * dre.shape[0] / dre.shape[0]  # active samples per frame
        n = (mask.sum())
    else:
        n = e.shape[1]*e.shape[2]*e.shape[3]
    return torch.sqrt(e.flatten(1).sum(1) / n).mean().item()


def interp_x_at_50(xs, ys):
    """smallest x (energy) at which fooling crosses 50%, along the sorted-by-x curve."""
    o = np.argsort(xs); xs, ys = np.array(xs)[o], np.array(ys)[o]
    for i in range(1, len(xs)):
        if ys[i-1] < 50 <= ys[i]:
            return round(float(xs[i-1]+(50-ys[i-1])*(xs[i]-xs[i-1])/(ys[i]-ys[i-1])), 5)
    return (float(xs[0]) if len(ys) and ys[0] >= 50 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=f'{CAPTURES}/enrollment')
    ap.add_argument('--model', default='fingerprint_cvnn.pt')
    ap.add_argument('--victim', type=int, default=6)
    ap.add_argument('--tgt-dst', type=int, default=4)
    ap.add_argument('--max-frames', type=int, default=120)
    ap.add_argument('--mask-lead', type=int, default=1)
    ap.add_argument('--methods', nargs='+', default=['fgsm', 'bim', 'pgd_a0.05'])
    ap.add_argument('--out', default='docs/b5_l2_vs_linf.csv')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    vcls = DEVICE_NAMES.index(f'device_{a.victim}'); tcls = DEVICE_NAMES.index(f'device_{a.tgt_dst}')
    m = CVNN(NUM_CLASSES).to(dev); m.load_state_dict(torch.load(a.model, map_location=dev, weights_only=False)); m.eval()

    fr = load_frames(a.root, a.victim, a.max_frames)
    W = torch.from_numpy(fr)
    re = W.real.unsqueeze(2).float().to(dev); im = W.imag.unsqueeze(2).float().to(dev)
    Fn = re.shape[0]
    print(f"loaded {Fn} frames; clean device_{a.victim}: {(frame_vote(m,re,im)==vcls).float().mean()*100:.0f}%")
    mask = None
    if a.mask_lead > 0:
        mask = torch.ones(NWIN, 1, WIN, device=dev); mask[:a.mask_lead] = 0.0

    # radius grids per norm (L2 radius ~ sqrt(active_dims)*rms; scaled to land the energy-50 crossover)
    GR = {'linf': [0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.05,0.06,0.08,0.10,0.13,0.16],
          'l2':   [1,2,3,4,5,6,8,10,13,16,20,26,32,40]}
    rows = []; summ = []
    for mode, ycls, sign in [('untargeted', vcls, +1.0), ('targeted', tcls, -1.0)]:
        y = torch.full((Fn,), ycls, dtype=torch.long, device=dev)
        for meth in a.methods:
            cfg = ATTACKS[meth]
            for norm in ['linf', 'l2']:
                for objn, fl in [('window', False), ('frame', True)]:
                    xs, ys = [], []
                    for radius in GR[norm]:
                        dre, dim = craft(re, im, y, radius, cfg, sign, m, dev, fl, mask, norm)
                        pv = frame_vote(m, re+dre, im+dim)
                        fool = ((pv != vcls) if mode == 'untargeted' else (pv == tcls)).float().mean().item()*100
                        e = rms(dre, dim, mask)
                        xs.append(e); ys.append(fool)
                        rows.append([meth, mode, norm, objn, radius, round(e, 5), round(fool, 2)])
                    summ.append((meth, mode, norm, objn, interp_x_at_50(xs, ys)))

    with open(a.out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['method','mode','norm','objective','radius','delta_rms','frame_fool_pct']); w.writerows(rows)
    print(f"wrote {a.out}\n")
    print("=== energy-eps50 (delta RMS at 50% frame-fool) — lower = more energy-efficient ===")
    print(f"{'method':11}{'mode':11}{'Linf-win':>10}{'Linf-frm':>10}{'L2-win':>9}{'L2-frm':>9}")
    bykey = {(me,mo,no,ob): e for (me,mo,no,ob,e) in summ}
    for me in a.methods:
        for mo in ['untargeted','targeted']:
            def g(no,ob):
                v=bykey.get((me,mo,no,ob)); return f'{v:.4f}' if v else '--'
            print(f"{me:11}{mo:11}{g('linf','window'):>10}{g('linf','frame'):>10}{g('l2','window'):>9}{g('l2','frame'):>9}")


if __name__ == '__main__':
    main()
