#!/usr/bin/env python3
"""
attack_matrix.py — full white-box adversarial attack matrix vs the Experiment-3 CVNN.

Follows the Experiment-1 docx methodology (28 epsilons 0.0001..0.10, PGD 10 steps,
per-input L_inf perturbation, accuracy + targeted-% + empirical BPSK-BER proxy) and
ADDS BIM and MIM.

Attacks (6 configs):
  FGSM            single signed step at epsilon
  BIM             iterative (10 steps), alpha=eps/steps, no random start, no momentum
  MIM             BIM + momentum (mu=1.0), gradient L1-normalised per step
  PGD(alpha=0.01) iterative, random start, absolute alpha, clip to eps
  PGD(alpha=0.05) "
  PGD(alpha=0.10) "

Modes:
  untargeted  — ascend CE on the true class, over ALL devices  (overall accuracy + BER)
  targeted    — descend CE on the target class, source device only (acc + target% + BER)

Input = unit-RMS complex 1024-window fed to the CVNN as (re, im). delta is L_inf<=eps
on (re, im) — the canonical white-box RX-side injection.

Usage:
  python3 attack_matrix.py \
    --model trained_models/fingerprint_cvnn.pt \
    --data-root $CAPTURES_ROOT/attack_enroll/day1/exp1 \
    --out-csv attack_matrix.csv --out-report attack_matrix_tables.txt \
    --tgt-src 6 --tgt-dst 1 --max-windows 300
"""
import argparse, os, glob, csv, time
import numpy as np, torch, torch.nn.functional as F
from cvnn import CVNN, iq_to_complex, to_reim
from fp_model import frame_to_windows, NUM_CLASSES, WIN
from extract_frames import extract_frames_for_file

EPSILONS = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
            0.001,  0.002,  0.003,  0.004,  0.005,  0.006,  0.007,  0.008,  0.009,
            0.01,   0.02,   0.03,   0.04,   0.05,   0.06,   0.07,   0.08,   0.09, 0.10]
PGD_STEPS = 10
FFT_LEN = 64
WIN_SYMBOLS = WIN // FFT_LEN                 # 16
NULL_BINS = [0, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]
ACTIVE_BINS = [i for i in range(FFT_LEN) if i not in NULL_BINS]   # 52

# attack config -> dict(steps, alpha_mode, momentum, random_start)
#   alpha_mode: 'eps'      -> alpha = eps                (FGSM single step)
#               'eps/steps'-> alpha = eps/steps          (BIM, MIM)
#               float      -> absolute alpha             (PGD)
ATTACKS = {
    'fgsm':       dict(steps=1,         alpha='eps',       momentum=None, random_start=False),
    'bim':        dict(steps=PGD_STEPS, alpha='eps/steps', momentum=None, random_start=False),
    'mim':        dict(steps=PGD_STEPS, alpha='eps/steps', momentum=1.0,  random_start=False),
    'pgd_a0.01':  dict(steps=PGD_STEPS, alpha=0.01,        momentum=None, random_start=True),
    'pgd_a0.05':  dict(steps=PGD_STEPS, alpha=0.05,        momentum=None, random_start=True),
    'pgd_a0.10':  dict(steps=PGD_STEPS, alpha=0.10,        momentum=None, random_start=True),
}


_SAMPLE_NPZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'docs', 'reachability_sample.npz')


def _sample_windows(device, max_windows):
    """Fallback windows from the shipped reachability sample when no captures are present."""
    if not os.path.exists(_SAMPLE_NPZ):
        return None
    z = np.load(_SAMPLE_NPZ)
    key = f'device_{device}'
    return z[key][:max_windows].astype(np.complex64) if key in z.files else None


def load_windows(root, device, max_windows):
    """Return (N, WIN) complex64 unit-RMS windows for a device (capped at max_windows).

    Falls back to docs/reachability_sample.npz (300 windows/device) when the enrollment
    captures are absent — e.g. before the dataset bundle is downloaded (README §3)."""
    files = sorted(glob.glob(os.path.join(root, f'device_{device}', 'clean_run_*.bin')))
    if not files:
        return _sample_windows(device, max_windows)
    chunks, n = [], 0
    for f in files:
        frames = extract_frames_for_file(f, floor_pct=20.0, thr_mult=2.0)[0]
        for fr in frames:
            w = iq_to_complex(frame_to_windows(fr, hop=WIN), cfo_correct=False)  # (nwin, WIN) complex
            chunks.append(np.asarray(w, np.complex64)); n += len(w)
            if n >= max_windows:
                break
        if n >= max_windows:
            break
    if not chunks:
        return _sample_windows(device, max_windows)
    return np.concatenate(chunks, 0)[:max_windows]


def to_reim_t(W, dev):
    t = torch.from_numpy(W)
    return t.real.unsqueeze(1).float().to(dev), t.imag.unsqueeze(1).float().to(dev)  # (N,1,WIN)


@torch.no_grad()
def predict(model, re, im, bs=2048):
    out = []
    for i in range(0, re.shape[0], bs):
        out.append(model(re[i:i+bs], im[i:i+bs]).argmax(1))
    return torch.cat(out)


def craft(model, re, im, y, eps, cfg, sign_mult, dev, bs=512, mask=None):
    """Return (dre, dim) L_inf<=eps perturbations. sign_mult: +1 untargeted (ascend true),
    -1 targeted (descend target). y already holds true (untargeted) or target (targeted) labels.

    mask: optional (N,1,WIN) 0/1 tensor. Zeroed positions are held at zero at EVERY step, so
    the attack is optimised under the constraint rather than clipped after the fact. Used to
    keep delta off the 802.11 MAC header (see --mask-lead in exp3_make_ota_perturbations.py):
    corrupting it makes every frame fail the scorer's address check and vanish from the tables."""
    steps = cfg['steps']
    if cfg['alpha'] == 'eps':
        alpha = eps
    elif cfg['alpha'] == 'eps/steps':
        alpha = eps / steps
    else:
        alpha = float(cfg['alpha'])
    mu = cfg['momentum']
    DRE = torch.empty_like(re); DIM = torch.empty_like(im)
    for b in range(0, re.shape[0], bs):
        rb, ib, yb = re[b:b+bs], im[b:b+bs], y[b:b+bs]
        mb = None if mask is None else mask[b:b+bs]
        if cfg['random_start']:
            dre = (torch.rand_like(rb) * 2 - 1) * eps
            dim = (torch.rand_like(ib) * 2 - 1) * eps
            if mb is not None:
                dre = dre * mb; dim = dim * mb
        else:
            dre = torch.zeros_like(rb); dim = torch.zeros_like(ib)
        gre = torch.zeros_like(rb); gim = torch.zeros_like(ib)
        for _ in range(steps):
            dre.requires_grad_(True); dim.requires_grad_(True)
            logits = model(rb + dre, ib + dim)
            loss = F.cross_entropy(logits, yb)
            g_re, g_im = torch.autograd.grad(loss, [dre, dim])
            with torch.no_grad():
                if mu is not None:
                    l1 = (g_re.abs().flatten(1).sum(1) + g_im.abs().flatten(1).sum(1)).clamp_min(1e-12).view(-1, 1, 1)
                    gre = mu * gre + g_re / l1
                    gim = mu * gim + g_im / l1
                    s_re, s_im = gre.sign(), gim.sign()
                else:
                    s_re, s_im = g_re.sign(), g_im.sign()
                dre = (dre.detach() + sign_mult * alpha * s_re).clamp(-eps, eps)
                dim = (dim.detach() + sign_mult * alpha * s_im).clamp(-eps, eps)
                if mb is not None:
                    dre = dre * mb; dim = dim * mb
        DRE[b:b+bs] = dre.detach(); DIM[b:b+bs] = dim.detach()
    return DRE, DIM


def ber_proxy(Wc, Wp):
    """BPSK subcarrier sign-flip proxy (matches the docx 'empirical BER')."""
    n = min(len(Wc), len(Wp))
    if n == 0:
        return 0, 0
    c = Wc[:n].reshape(n, WIN_SYMBOLS, FFT_LEN)
    p = Wp[:n].reshape(n, WIN_SYMBOLS, FFT_LEN)
    cf = np.fft.fft(c, axis=-1)[..., ACTIVE_BINS]
    pf = np.fft.fft(p, axis=-1)[..., ACTIVE_BINS]
    flips = (np.sign(cf.real) != np.sign(pf.real))
    return int(flips.sum()), int(cf.size)


def recon(re, im):
    return (re.squeeze(1).cpu().numpy() + 1j * im.squeeze(1).cpu().numpy()).astype(np.complex64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out-csv', required=True)
    ap.add_argument('--out-report', required=True)
    ap.add_argument('--devices', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument('--epsilons', nargs='+', type=float, default=EPSILONS)
    ap.add_argument('--attacks', nargs='+', default=list(ATTACKS.keys()))
    ap.add_argument('--modes', nargs='+', default=['untargeted', 'targeted'])
    ap.add_argument('--tgt-src', type=int, default=5)
    ap.add_argument('--tgt-dst', type=int, default=1)
    ap.add_argument('--max-windows', type=int, default=300)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={dev}  model={a.model}")
    m = CVNN(NUM_CLASSES).to(dev)
    m.load_state_dict(torch.load(a.model, map_location=dev, weights_only=False)); m.eval()

    # --- load clean windows ---
    reim, Wc, clean_acc = {}, {}, {}
    for d in a.devices:
        W = load_windows(a.data_root, d, a.max_windows)
        if W is None:
            print(f"  Dev{d}: NO DATA"); continue
        Wc[d] = W
        reim[d] = to_reim_t(W, dev)
        pred = predict(m, *reim[d]).cpu().numpy()
        clean_acc[d] = float((pred == d - 1).mean())
        print(f"  Dev{d}: {len(W)} windows  clean_acc={clean_acc[d]*100:.2f}%")
    overall_clean = np.mean([clean_acc[d] for d in Wc])
    print(f"  overall clean acc = {overall_clean*100:.2f}%")

    rows = []
    src, dst = a.tgt_src, a.tgt_dst
    n_pts = len(a.attacks) * len([mo for mo in a.modes]) * len(a.epsilons)
    cnt = 0; t0 = time.time()

    for attack in a.attacks:
        cfg = ATTACKS[attack]
        for mode in a.modes:
            sign_mult = +1.0 if mode == 'untargeted' else -1.0
            for eps in a.epsilons:
                cnt += 1
                if mode == 'untargeted':
                    tot_flip = tot_bit = 0; corr = tot = 0
                    per_dev = {}
                    for d in Wc:
                        re, im = reim[d]
                        y = torch.full((re.shape[0],), d - 1, dtype=torch.long, device=dev)
                        dre, dim = craft(m, re, im, y, eps, cfg, sign_mult, dev)
                        pred = predict(m, re + dre, im + dim).cpu().numpy()
                        acc = float((pred == d - 1).mean())
                        fl, bt = ber_proxy(Wc[d], recon(re + dre, im + dim))
                        per_dev[d] = acc; corr += (pred == d - 1).sum(); tot += len(pred)
                        tot_flip += fl; tot_bit += bt
                    acc_overall = corr / tot
                    ber = tot_flip / tot_bit if tot_bit else float('nan')
                    rows.append(dict(attack=attack, mode=mode, epsilon=eps, scope='overall',
                                     acc=acc_overall, target_pct='', ber=ber,
                                     err_bits=tot_flip, tot_bits=tot_bit,
                                     **{f'dev{d}_acc': per_dev.get(d, '') for d in a.devices}))
                else:
                    if src not in Wc:
                        continue
                    re, im = reim[src]
                    y = torch.full((re.shape[0],), dst - 1, dtype=torch.long, device=dev)
                    dre, dim = craft(m, re, im, y, eps, cfg, sign_mult, dev)
                    pred = predict(m, re + dre, im + dim).cpu().numpy()
                    acc = float((pred == src - 1).mean())          # source still classified as source
                    tpct = float((pred == dst - 1).mean())         # pushed to target
                    fl, bt = ber_proxy(Wc[src], recon(re + dre, im + dim))
                    ber = fl / bt if bt else float('nan')
                    rows.append(dict(attack=attack, mode=f'targeted_{src}to{dst}', epsilon=eps,
                                     scope=f'dev{src}', acc=acc, target_pct=tpct, ber=ber,
                                     err_bits=fl, tot_bits=bt))
                if cnt % 10 == 0 or cnt == n_pts:
                    el = time.time() - t0
                    print(f"[{cnt}/{n_pts}] {attack} {mode} eps={eps:.4f}  "
                          f"({el/60:.1f}min, ~{el/cnt*(n_pts-cnt)/60:.1f}min left)", flush=True)

    # --- CSV ---
    keys = ['attack', 'mode', 'epsilon', 'scope', 'acc', 'target_pct', 'ber', 'err_bits', 'tot_bits'] \
           + [f'dev{d}_acc' for d in a.devices]
    with open(a.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    print(f"saved {len(rows)} rows -> {a.out_csv}")

    # --- docx-style text tables ---
    with open(a.out_report, 'w') as f:
        f.write("Experiment-3 CVNN — adversarial attack matrix\n")
        f.write(f"model: {a.model}\noverall clean acc: {overall_clean*100:.2f}%\n")
        f.write(f"targeted: Dev{src} -> Dev{dst}   epsilons: {len(a.epsilons)}   PGD steps: {PGD_STEPS}\n\n")
        for attack in a.attacks:
            for mode in a.modes:
                sub = [r for r in rows if r['attack'] == attack and str(r['mode']).startswith(mode.split('_')[0])]
                if not sub:
                    continue
                f.write(f"=== {attack.upper()} — {mode} ===\n")
                if mode == 'untargeted':
                    f.write(f"{'eps':>8} {'acc%':>7} {'BER':>10} {'errbits':>8} {'totbits':>9}\n")
                    for r in sub:
                        f.write(f"{r['epsilon']:>8.4f} {r['acc']*100:>7.2f} {r['ber']:>10.2e} "
                                f"{r['err_bits']:>8d} {r['tot_bits']:>9d}\n")
                else:
                    f.write(f"{'eps':>8} {'acc%':>7} {'target%':>8} {'BER':>10} {'totbits':>9}\n")
                    for r in sub:
                        f.write(f"{r['epsilon']:>8.4f} {r['acc']*100:>7.2f} {r['target_pct']*100:>8.2f} "
                                f"{r['ber']:>10.2e} {r['tot_bits']:>9d}\n")
                f.write("\n")
    print(f"saved tables -> {a.out_report}")


if __name__ == '__main__':
    main()
