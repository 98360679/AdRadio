#!/usr/bin/env python3
"""Make OTA delta banks for the B5 confirmation: L-inf/L2 x per-window/frame-level, crafted against
ch0_run3_107 on unit-RMS windows (like exp3_make_ota_perturbations) then un-normalized to the transmit
domain. One bank per config; radii become the 'epsilon' labels so build_ch1.py can bundle them.

Reuses exp3_b5_l2_vs_linf.craft (L-inf clip / per-frame L2 projection, per-window vs frame-level loss).
"""
import argparse, os, csv
import numpy as np, torch
from exp3_extract_frames import extract_frames_for_file
from exp3_fp_model import NUM_CLASSES, DEVICE_NAMES, WIN, PRE_ROLL, ACTIVE
from exp3_cvnn import CVNN
from exp3_attack_matrix import ATTACKS
from exp3_b5_l2_vs_linf import craft
from exp3_b3_frame_objective import frame_vote, NWIN
import os
CAPTURES = os.environ.get('CAPTURES_ROOT', os.path.expanduser('~/captures'))

STARTS = list(range(PRE_ROLL, PRE_ROLL + ACTIVE - WIN + 1, WIN))   # 13


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ch0', default=f'{CAPTURES}/attack_assets/ch0_run3_107.bin')
    ap.add_argument('--model', default='fingerprint_cvnn.pt')
    ap.add_argument('--method', default='pgd_a0.05')
    ap.add_argument('--victim', type=int, default=5)
    ap.add_argument('--nper', type=int, default=107)
    ap.add_argument('--mask-lead', type=int, default=1)
    ap.add_argument('--norm', required=True, choices=['linf', 'l2'])
    ap.add_argument('--objective', required=True, choices=['window', 'frame'])
    ap.add_argument('--radii', required=True, help='comma list of radii (L-inf eps or L2 radius)')
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    vcls = DEVICE_NAMES.index(f'device_{a.victim}')
    m = CVNN(NUM_CLASSES).to(dev); m.load_state_dict(torch.load(a.model, map_location=dev, weights_only=False)); m.eval()
    cfg = ATTACKS[a.method]; fl = (a.objective == 'frame')
    os.makedirs(a.out_dir, exist_ok=True)

    # load nper frames from ch0 (transmit domain), 13 windows each
    ch0 = np.fromfile(a.ch0, dtype=np.complex64); FP = 65661
    Wr = np.stack([np.stack([ch0[k*FP+s:k*FP+s+WIN] for s in STARTS]) for k in range(a.nper)])  # (nper,13,WIN)
    rms = np.sqrt((np.abs(Wr)**2).mean(-1, keepdims=True)) + 1e-9                                # (nper,13,1)
    Wn = (Wr / rms).astype(np.complex64)                                                          # unit-RMS
    W = torch.from_numpy(Wn)
    re = W.real.unsqueeze(2).float().to(dev); im = W.imag.unsqueeze(2).float().to(dev)            # (nper,13,1,WIN)
    rms_t = torch.from_numpy(rms).unsqueeze(2).float().to(dev)                                     # (nper,13,1,1)
    y = torch.full((a.nper,), vcls, dtype=torch.long, device=dev)                                 # untargeted -> off victim
    mask = None
    if a.mask_lead > 0:
        mask = torch.ones(NWIN, 1, WIN, device=dev); mask[:a.mask_lead] = 0.0

    man = []
    for radius in [float(x) for x in a.radii.split(',')]:
        dre, dim = craft(re, im, y, radius, cfg, +1.0, m, dev, fl, mask, a.norm)   # normalized-domain delta
        # digital fooling on normalized windows (sanity)
        pv = frame_vote(m, re+dre, im+dim); dfool = (pv != vcls).float().mean().item()*100
        # un-normalize to transmit domain and save per-frame delta over ACTIVE
        d = (dre.squeeze(2) + 1j*dim.squeeze(2)).cpu().numpy() * rms.reshape(a.nper, NWIN, 1)   # (nper,13,WIN) complex
        d = d.reshape(a.nper, ACTIVE).astype(np.complex64)                                       # (nper,13312)
        drms = float(np.sqrt(np.mean(np.abs(d)**2)))
        lab = f"{radius:g}"
        np.save(os.path.join(a.out_dir, f"delta_{a.method}_e{lab}.npy"), d)
        man.append(dict(method=a.method, epsilon=lab, radius=radius, norm=a.norm, objective=a.objective,
                        digital_fool_pct=round(dfool, 1), delta_tx_rms=round(drms, 5), realized_psr_db=0.0))
        print(f"  r={radius:g}  digital_fool={dfool:5.1f}%  tx_rms={drms:.5f}  -> delta_{a.method}_e{lab}.npy")
    with open(os.path.join(a.out_dir, 'manifest.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(man[0].keys())); w.writeheader(); w.writerows(man)
    print(f"wrote {len(man)} deltas + manifest -> {a.out_dir}")


if __name__ == '__main__':
    main()
