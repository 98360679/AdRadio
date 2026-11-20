#!/usr/bin/env python3
"""build_ch0_sc.py — SINGLE-CHANNEL sweep: fold u+delta into ch0 so the victim
radiates the perturbation through its OWN chain. ch1 is silent.

No inter-channel phase exists in this configuration, so there is no blind 4-phase grid —
one block per epsilon (still 2 ch0 passes, the first absorbing the block boundary).
delta is scaled by the SAME factor ch0 was scaled by, so PSR is preserved.
"""
import os as _os, sys as _sys
_R = _os.path.abspath(_os.path.dirname(__file__))
while _R != _os.path.dirname(_R) and not _os.path.isdir(_os.path.join(_R, ".git")):
    _R = _os.path.dirname(_R)
_sys.path.insert(0, _os.path.join(_R, "lib"))
_os.chdir(_R)
import argparse, csv, os, numpy as np
from fp_model import PRE_ROLL, ACTIVE

FRAME_PERIOD = 65661   # overridden by --frame-period


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pert-dir', required=True)
    ap.add_argument('--ch0', required=True, help='dense TX-level ch0, one pass')
    ap.add_argument('--method', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--index-out', required=True)
    ap.add_argument('--zeros-out', required=True, help='silent ch1 file (one pass)')
    ap.add_argument('--passes', type=int, default=2)
    ap.add_argument('--frame-period', type=int, default=None,
                    help='frame spacing in ch0 (a frame-id-matched ch0 uses its burst_len)')
    ap.add_argument('--epsilons', default=None,
                    help='comma list restricting the sweep to a SUBSET of the bank. Mirrors '
                         'build_ch1.py so both experiments can share one EPS_GRID.')
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--headroom', type=float, default=None,
                    help='target peak for u+delta (e.g. 0.95). ONE global factor is derived from '
                         'the loudest epsilon and applied to every block, so relative amplitude '
                         'across the sweep — and therefore PSR — is preserved. The classifier '
                         'normalises each window to unit RMS, so a common factor is invisible to it.')
    a = ap.parse_args()
    global FRAME_PERIOD
    if a.frame_period:
        FRAME_PERIOD = a.frame_period

    u = np.fromfile(a.ch0, dtype=np.complex64)
    L = len(u)
    nfr = L // FRAME_PERIOD
    man = list(csv.DictReader(open(os.path.join(a.pert_dir, 'manifest.csv'))))
    eps_list = [float(r['epsilon']) for r in man if r['method'] == a.method]
    if a.epsilons:
        want = [float(x) for x in a.epsilons.split(',')]
        eps_list = [e for e in eps_list if any(abs(e - w) < 1e-12 for w in want)]
        missing = [w for w in want if not any(abs(e - w) < 1e-12 for e in eps_list)]
        if missing:
            raise SystemExit(f'epsilons not in the bank: {missing}')
    assert eps_list, f"no rows for method {a.method}"

    np.zeros(L, dtype=np.complex64).tofile(a.zeros_out)     # silent ch1
    blk = a.passes * L
    print(f"ch0 pass {L} ({L/5e6:.3f}s), {nfr} frames; {len(eps_list)} eps x {a.passes} passes "
          f"= {len(eps_list)*blk/5e6:.0f}s, {len(eps_list)*blk*8/1e9:.2f} GB")

    gfac = 1.0
    if a.headroom:
        pk = 0.0
        for eps in eps_list:
            d = np.load(os.path.join(a.pert_dir, f"delta_{a.method}_e{eps:g}.npy"))
            if len(d) < nfr:
                raise SystemExit(
                    f"BANK TOO SHORT: {len(d)} deltas for {nfr} frame slots. Frames "
                    f"{len(d)}..{nfr - 1} would be transmitted UNPERTURBED, capping every "
                    f"result at {100 * len(d) / nfr:.1f}% and reading as a 'structural "
                    f"ceiling' -- this is exactly what produced the phantom ~70% ceiling in "
                    f"all 8/12-8/13 results. Re-craft with --nper {nfr}. Do NOT tile the bank: "
                    f"delta[k] is crafted against ch0 frame k, so reusing it on frame k+len(d) "
                    f"applies a perturbation optimised for different IQ.")
            one = u.copy()
            for k in range(nfr):
                s0 = k * FRAME_PERIOD + PRE_ROLL
                one[s0:s0 + ACTIVE] += (d[k] * a.scale).astype(np.complex64)
            pk = max(pk, float(np.abs(one).max()))
        gfac = a.headroom / pk
        print(f"global headroom factor {gfac:.4f} (worst peak {pk:.3f} -> {a.headroom})")

    rows = []
    with open(a.out, 'wb') as f:
        for i, eps in enumerate(eps_list):
            d = np.load(os.path.join(a.pert_dir, f"delta_{a.method}_e{eps:g}.npy"))
            if len(d) < nfr:
                raise SystemExit(f"BANK TOO SHORT: {len(d)} deltas for {nfr} frame slots "
                                 f"-- re-craft with --nper {nfr} (tiling is NOT valid)")
            one = u.copy()
            for k in range(nfr):
                s = k * FRAME_PERIOD + PRE_ROLL
                one[s:s + ACTIVE] += (d[k] * a.scale).astype(np.complex64)
            one = (one * gfac).astype(np.complex64)
            pk = np.abs(one).max()
            if pk > 1.0:                       # the DAC clips at 1.0; record it, do not silently ride it
                print(f"  WARNING eps={eps:g}: peak {pk:.3f} > 1.0 — clipping in the victim chain")
            np.tile(one, a.passes).tofile(f)
            rows.append(dict(block=i, method=a.method, epsilon=eps, phase_deg=0.0,
                             start_sample=i * blk, n_samples=blk,
                             score_from=i * blk + (a.passes - 1) * L, score_to=(i + 1) * blk,
                             peak=round(float(pk), 4)))
            print(f"  eps={eps:g} peak={pk:.3f} ({i+1}/{len(eps_list)})", flush=True)

    with open(a.index_out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out} ({os.path.getsize(a.out)/1e9:.2f} GB) + {a.index_out}")


if __name__ == '__main__':
    main()
