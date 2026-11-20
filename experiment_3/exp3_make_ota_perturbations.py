#!/usr/bin/env python3
"""
exp3_make_ota_perturbations.py — craft OTA-transmittable delta from the attack matrix.

For every (attack method, epsilon) it crafts a TARGETED delta (Dev6->Dev1) against the
victim replay ch0, using the same FGSM/BIM/MIM/PGD craft() as exp3_attack_matrix, then
stores the delta COMPACTLY (active region only, 100 frames x ACTIVE complex64 ~= 11.5 MB).
The full single-channel (ch0+delta) and 2-channel (delta) OTA waveforms are materialised
on demand by exp3_expand_ota_perturbation.py.

delta is crafted on the unit-RMS (re,im) window (L_inf<=eps, matrix convention) and then
UN-normalised (x per-window RMS) so it lives in ch0's amplitude domain and can be added to
ch0 directly (single-channel) or transmitted alone (2-channel).

Records per (method,eps): digital targeted success (ch0+delta -> Dev1) and realised PSR.

Usage:
  python3 exp3_make_ota_perturbations.py \
    --model trained_models/fingerprint_cvnn.pt \
    --ch0 $CAPTURES_ROOT/attack_enroll/attack/ch0.bin \
    --out-dir $CAPTURES_ROOT/attack_enroll/ota_perturbations \
    --tgt-src 5 --tgt-dst 1
"""
import os as _os, sys as _sys
_R = _os.path.abspath(_os.path.dirname(__file__))
while _R != _os.path.dirname(_R) and not _os.path.isdir(_os.path.join(_R, ".git")):
    _R = _os.path.dirname(_R)
_sys.path.insert(0, _os.path.join(_R, "lib"))
_os.chdir(_R)
import argparse, os, csv, time
import numpy as np, torch
from exp3_cvnn import CVNN, iq_to_complex, to_reim
from exp3_fp_model import PRE_ROLL, ACTIVE, WIN, NUM_CLASSES, DEVICE_NAMES, frame_to_windows
from exp3_attack_matrix import craft, ATTACKS, EPSILONS

FRAME_PERIOD = 65661
FRAME_LEN = PRE_ROLL + ACTIVE            # 500 + 13312 = 13812 (window span)
STARTS = list(range(PRE_ROLL, PRE_ROLL + ACTIVE - WIN + 1, WIN))   # 13 windows


def frames_from_ch0(ch0, nper):
    return [ch0[i * FRAME_PERIOD: i * FRAME_PERIOD + FRAME_LEN + WIN].copy() for i in range(nper)]


@torch.no_grad()
def frame_pred(model, frame, dev):
    w = iq_to_complex(np.stack([frame[s:s + WIN] for s in STARTS]), cfo_correct=False)
    re, im = to_reim(torch.from_numpy(w), dev)
    votes = model(re, im).argmax(1)
    return int(torch.bincount(votes, minlength=NUM_CLASSES).argmax())


def craft_universal_frames(model, re, im, y, eps, cfg, sign_mult, dev, nS, mask_win=None):
    """UNIVERSAL delta: ONE frame-delta of shape (nS,1,WIN) shared across ALL frames (the single
    waveform the adversary radiates on every frame). re/im are (N=nper*nS,1,WIN) ordered
    frame-major. The delta is repeated over frames each step, so grad of the mean loss w.r.t. the
    shared delta equals the mean per-frame grad -> sign(grad)==sign(mean raw grad). Per-window
    structure within the frame is retained; only window 0's MAC header is masked (mask_win)."""
    import torch.nn.functional as F
    steps = cfg['steps']
    alpha = eps if cfg['alpha'] == 'eps' else (eps / steps if cfg['alpha'] == 'eps/steps' else float(cfg['alpha']))
    mu = cfg['momentum']
    nper = re.shape[0] // nS
    dre = torch.zeros(nS, 1, WIN, device=dev); dim = torch.zeros(nS, 1, WIN, device=dev)
    if cfg['random_start']:
        dre = (torch.rand_like(dre) * 2 - 1) * eps; dim = (torch.rand_like(dim) * 2 - 1) * eps
    if mask_win is not None:
        dre = dre * mask_win; dim = dim * mask_win
    gre = torch.zeros_like(dre); gim = torch.zeros_like(dim)
    for _ in range(steps):
        dre.requires_grad_(True); dim.requires_grad_(True)
        DRE = dre.repeat(nper, 1, 1); DIM = dim.repeat(nper, 1, 1)          # (N,1,WIN)
        loss = F.cross_entropy(model(re + DRE, im + DIM), y)                 # mean over all frames*windows
        g_re, g_im = torch.autograd.grad(loss, [dre, dim])                  # (nS,1,WIN) = summed over frames
        with torch.no_grad():
            if mu is not None:
                l1 = (g_re.abs().sum() + g_im.abs().sum()).clamp_min(1e-12)
                gre = mu * gre + g_re / l1; gim = mu * gim + g_im / l1
                s_re, s_im = gre.sign(), gim.sign()
            else:
                s_re, s_im = g_re.sign(), g_im.sign()
            dre = (dre.detach() + sign_mult * alpha * s_re).clamp(-eps, eps)
            dim = (dim.detach() + sign_mult * alpha * s_im).clamp(-eps, eps)
            if mask_win is not None:
                dre = dre * mask_win; dim = dim * mask_win
    return dre.detach(), dim.detach()          # (nS,1,WIN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--arch', choices=['cvnn', 'cnn1d', 'freqcvnn'], default='cvnn',
                    help='surrogate architecture to CRAFT on (white-box). Default cvnn.')
    ap.add_argument('--ch0', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--methods', nargs='+', default=list(ATTACKS.keys()))
    ap.add_argument('--epsilons', nargs='+', type=float, default=EPSILONS)
    ap.add_argument('--tgt-src', type=int, default=5)
    ap.add_argument('--tgt-dst', type=int, default=1)
    ap.add_argument('--nper', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--mask-lead', type=int, default=0,
                    help='hold delta at zero over the first N samples of the active window, i.e. '
                         'frame samples PRE_ROLL..PRE_ROLL+N. The 802.11 MAC header lives there '
                         '(measured 8/15: perturbing samples 500-1220 alone rejected 86/88 frames '
                         'on the address check while causing only 2 bit errors). 780 protects it.')
    ap.add_argument('--untargeted', action='store_true',
                    help='push the victim OFF its own class instead of toward --tgt-dst')
    ap.add_argument('--universal', action='store_true',
                    help='craft ONE frame-delta shared across all frames (a universal perturbation: '
                         'the single waveform the adversary radiates every frame) instead of a '
                         'per-frame delta. The bank rows are then all identical.')
    ap.add_argument('--bank-frames', type=int, default=0,
                    help='universal only: write this many identical bank rows (>= ch0 frame count). '
                         'Valid because the universal delta is identical every frame (not tiling a '
                         'per-frame bank). 0 keeps the crafted frame count.')
    ap.add_argument('--victim-class', type=int, default=5,
                    help='1-based model class index of the victim (5 = device_5 in the 5-dev model)')
    ap.add_argument('--from-capture', action='store_true',
                    help='take frame positions from the burst detector instead of assuming a '
                         'fixed FRAME_PERIOD — required for a live capture (non-periodic strobe)')
    ap.add_argument('--thr-mult', type=float, default=4.0, help='burst-detector threshold (--from-capture)')
    ap.add_argument('--frame-period', type=int, default=None,
                    help='sample spacing between frames in ch0. Default is the replay-bundle '
                         'FRAME_PERIOD; a frame-id-matched ch0 uses its own burst_len.')
    a = ap.parse_args()
    global FRAME_PERIOD
    if a.frame_period:
        FRAME_PERIOD = a.frame_period
    os.makedirs(a.out_dir, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    from exp3_cnn1d import CNN1DFp
    from exp3_freqcvnn import FreqCVNN
    ARCH = {'cvnn': CVNN, 'cnn1d': CNN1DFp, 'freqcvnn': FreqCVNN}
    m = ARCH[a.arch](NUM_CLASSES).to(dev)
    m.load_state_dict(torch.load(a.model, map_location=dev, weights_only=False)); m.eval()
    print(f"crafting on arch={a.arch} model={a.model}")

    if a.from_capture:
        # live capture: the strobe is NOT periodic, so detect the bursts and keep their offsets
        from exp3_extract_frames import extract_frames_for_file
        det, starts = extract_frames_for_file(a.ch0, floor_pct=20.0, thr_mult=a.thr_mult)[:2]
        # frames_from_ch0 slices FRAME_LEN+WIN; the detector returns FRAME_LEN, so re-slice
        ch0 = np.fromfile(a.ch0, dtype=np.complex64)
        keep = [s for s in starts if s + FRAME_LEN + WIN <= len(ch0)][:a.nper]
        frames = [ch0[s: s + FRAME_LEN + WIN].copy() for s in keep]
        a.nper = len(frames)
        starts_used = np.array(keep, dtype=np.int64)
        print(f"ch0: {len(ch0)} samples -> {len(frames)} detected frames (non-periodic capture)")
    else:
        ch0 = np.fromfile(a.ch0, dtype=np.complex64)
        frames = frames_from_ch0(ch0, a.nper)
        starts_used = np.arange(a.nper, dtype=np.int64) * FRAME_PERIOD
        print(f"ch0: {len(ch0)} samples -> {len(frames)} frames")
    mode = 'untargeted' if a.untargeted else f"targeted Dev{a.tgt_src}->class{a.tgt_dst}"
    print(f"mode: {mode}")

    # build the raw windows + per-window rms for ALL frames once
    Wr = np.stack([np.stack([fr[s:s + WIN] for s in STARTS]) for fr in frames])   # (nper,14,WIN) complex
    rms = np.sqrt((np.abs(Wr) ** 2).mean(-1, keepdims=True)) + 1e-9                 # (nper,14,1)
    Wn = (Wr / rms).astype(np.complex64)                                           # unit-RMS
    N = a.nper * len(STARTS)
    re = torch.from_numpy(Wn.reshape(N, WIN).real).unsqueeze(1).float().to(dev)
    im = torch.from_numpy(Wn.reshape(N, WIN).imag).unsqueeze(1).float().to(dev)
    vcls = a.victim_class - 1
    ycls = vcls if a.untargeted else a.tgt_dst - 1
    sign = +1.0 if a.untargeted else -1.0          # +1 = ascend own-class loss, -1 = descend target loss
    y = torch.full((N,), ycls, dtype=torch.long, device=dev)
    rms_t = torch.from_numpy(rms.reshape(N, 1, 1)).float().to(dev)

    # active-region power per frame (for PSR)
    act_pow = np.array([np.abs(fr[PRE_ROLL:PRE_ROLL + ACTIVE]) ** 2 for fr in frames]).sum(1)

    mask_t = None
    if a.mask_lead > 0:
        nS = len(STARTS)
        mk = np.ones((a.nper, nS, WIN), dtype=np.float32)
        mk[:, 0, :min(a.mask_lead, WIN)] = 0.0        # lead falls inside window 0 (hop == WIN)
        assert a.mask_lead <= WIN, 'mask_lead spans past window 0; needs multi-window masking'
        mask_t = torch.from_numpy(mk.reshape(a.nper * nS, WIN)).unsqueeze(1).to(dev)
        print(f"mask: delta held at zero over frame samples "
              f"{PRE_ROLL}..{PRE_ROLL + a.mask_lead} ({a.mask_lead} samples, MAC header)")

    manifest = []
    n_cfg = len(a.methods) * len(a.epsilons); c = 0; t0 = time.time()
    for method in a.methods:
        cfg = ATTACKS[method]
        for eps in a.epsilons:
            c += 1
            if a.universal:
                nS = len(STARTS)
                # window-0 header mask (nS,1,WIN): protect only the frame's MAC header
                mw = None
                if a.mask_lead > 0:
                    mk = np.ones((nS, 1, WIN), dtype=np.float32); mk[0, 0, :min(a.mask_lead, WIN)] = 0.0
                    mw = torch.from_numpy(mk).to(dev)
                dre, dim = craft_universal_frames(m, re, im, y, eps, cfg, sign, dev, nS, mask_win=mw)
                d_norm = torch.complex(dre.squeeze(1), dim.squeeze(1)).cpu().numpy()   # (nS,WIN)
                mean_rms = rms.reshape(a.nper, nS, 1).mean(0).reshape(nS, 1)           # per-window mean RMS
                d_raw_frame = (d_norm * mean_rms).astype(np.complex64)                 # (nS,WIN) ONE frame
                d_raw = np.broadcast_to(d_raw_frame[None], (a.nper, nS, WIN))          # identical every frame
            else:
                dre, dim = craft(m, re, im, y, eps, cfg, sign, dev, mask=mask_t)   # delta on normalised windows
                d_norm = torch.complex(dre.squeeze(1), dim.squeeze(1))        # (N,WIN)
                d_raw = (d_norm * rms_t.squeeze(1)).cpu().numpy().reshape(a.nper, len(STARTS), WIN)  # un-normalise
            # assemble per-frame active-region delta (ACTIVE samples starting at PRE_ROLL)
            act = np.zeros((a.nper, ACTIVE), dtype=np.complex64)
            for i, s in enumerate(STARTS):
                act[:, s - PRE_ROLL: s - PRE_ROLL + WIN] = d_raw[:, i, :]
            if a.mask_lead > 0:
                act[:, :a.mask_lead] = 0            # must already hold; guards a mask/geometry drift
            # validate + PSR
            hits = 0; dpow = (np.abs(act) ** 2).sum(1)
            land = np.zeros(NUM_CLASSES, dtype=int)
            for j, fr in enumerate(frames):
                pf = fr.copy(); pf[PRE_ROLL:PRE_ROLL + ACTIVE] += act[j]
                p = frame_pred(m, pf, dev)
                land[p] += 1
                hits += (p != vcls) if a.untargeted else (p == a.tgt_dst - 1)
            psr = 10 * np.log10((dpow / np.maximum(act_pow, 1e-12)).mean())
            tag = f"{method}_e{eps:g}"
            save_act = act
            if a.universal and a.bank_frames > a.nper:
                save_act = np.broadcast_to(act[0][None], (a.bank_frames, ACTIVE)).copy()  # identical rows
            np.save(os.path.join(a.out_dir, f"delta_{tag}.npy"), save_act)
            row = dict(method=method, epsilon=eps, mode='untargeted' if a.untargeted else 'targeted',
                       tgt_src=a.tgt_src, tgt_dst='' if a.untargeted else a.tgt_dst,
                       digital_pct=round(100 * hits / a.nper, 2),
                       landed=DEVICE_NAMES[int(land.argmax())],
                       landed_pct=round(100 * land.max() / a.nper, 2),
                       realized_psr_db=round(float(psr), 2), file=f"delta_{tag}.npy")
            row.update({DEVICE_NAMES[i]: int(land[i]) for i in range(NUM_CLASSES)})
            manifest.append(row)
            if c % 8 == 0 or c == n_cfg:
                el = time.time() - t0
                lbl_pct = 'fool' if a.untargeted else 'tgt'
                print(f"[{c}/{n_cfg}] {tag}: {lbl_pct}%={100*hits/a.nper:.0f} "
                      f"lands {row['landed']} {row['landed_pct']:.0f}% PSR={psr:.1f}dB "
                      f"({el/60:.1f}min, ~{el/c*(n_cfg-c)/60:.1f}min left)", flush=True)

    with open(os.path.join(a.out_dir, 'manifest.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys())); w.writeheader(); w.writerows(manifest)
    # also stash ch0 path + the frame offsets the delta was crafted at (the expander must
    # place delta back at exactly these offsets — they are NOT periodic for a live capture)
    with open(os.path.join(a.out_dir, 'ch0_path.txt'), 'w') as f:
        f.write(a.ch0 + "\n")
    np.save(os.path.join(a.out_dir, 'frame_starts.npy'), starts_used)
    print(f"saved {len(manifest)} compact deltas + manifest.csv -> {a.out_dir}")


if __name__ == '__main__':
    main()
