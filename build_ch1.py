#!/usr/bin/env python3
"""build_ch1.py — pre-concatenate the whole per-method epsilon x phase sweep into ONE
long ch1 file, so the 2-channel session runs at a FIXED power-up phase with no live file
swapping (a swap stalls the TX; a session-per-point redraws the inter-channel phase).

Layout, per (epsilon, phase) point:  2 consecutive passes of ch0's length.
  pass 0 = settle/boundary guard (discarded at scoring)
  pass 1 = scored
ch0 is played with repeat=True, so ch0's period divides every block boundary exactly and
the two channels stay aligned for the whole session.

Blind 4-phase: each epsilon is emitted at psi in {0, 90, 180, 270} deg. The inter-channel
RF phase is redrawn at every power-up, so the attack is scored as max-over-psi.
"""
import argparse, os, numpy as np
from exp3_fp_model import PRE_ROLL, ACTIVE

FRAME_PERIOD = 65661   # overridden by --frame-period
PHASES_DEG = [0.0, 90.0, 180.0, 270.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pert-dir', required=True, help='dense delta bank (delta_<method>_e<eps>.npy)')
    ap.add_argument('--ch0', required=True)
    ap.add_argument('--method', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--index-out', required=True, help='csv describing every block, for scoring')
    ap.add_argument('--passes', type=int, default=2, help='ch0 passes per point (first is discarded)')
    ap.add_argument('--frame-period', type=int, default=None,
                    help='frame spacing in ch0 (a frame-id-matched ch0 uses its burst_len)')
    ap.add_argument('--phases', default=None,
                    help='comma list of psi in degrees, overriding the blind-4-phase set. Use a '
                         'FINE grid (e.g. 5 deg steps) at ONE epsilon to map where the live window '
                         'sits for this power-up.')
    ap.add_argument('--epsilons', default=None,
                    help='comma list to restrict the sweep to a SUBSET of the bank, e.g. for a '
                         'short gain-sweep probe. Default: every epsilon in the manifest.')
    ap.add_argument('--scale', type=float, default=1.0,
                    help='same factor ch0 was scaled by to reach transmit level — u and delta '
                         'MUST be scaled equally or the PSR labels are wrong')
    a = ap.parse_args()
    global FRAME_PERIOD
    if a.frame_period:
        FRAME_PERIOD = a.frame_period

    L = os.path.getsize(a.ch0) // 8                 # complex64 samples in one ch0 pass
    import csv
    man = list(csv.DictReader(open(os.path.join(a.pert_dir, 'manifest.csv'))))
    global PHASES_DEG
    if a.phases:
        PHASES_DEG = [float(x) for x in a.phases.split(',')]
    eps_list = [float(r['epsilon']) for r in man if r['method'] == a.method]
    if a.epsilons:
        want = [float(x) for x in a.epsilons.split(',')]
        eps_list = [e for e in eps_list if any(abs(e - w) < 1e-12 for w in want)]
        missing = [w for w in want if not any(abs(e - w) < 1e-12 for e in eps_list)]
        if missing:
            raise SystemExit(f'epsilons not in the bank: {missing}')
    assert eps_list, f"no rows for method {a.method}"
    nfr = L // FRAME_PERIOD                          # frames per pass

    blk = a.passes * L
    print(f"ch0 pass = {L} samples ({L/5e6:.3f}s), {nfr} frames/pass")
    print(f"{len(eps_list)} epsilons x {len(PHASES_DEG)} phases x {a.passes} passes "
          f"= {len(eps_list)*len(PHASES_DEG)*blk/5e6:.0f}s airtime, "
          f"{len(eps_list)*len(PHASES_DEG)*blk*8/1e9:.1f} GB")

    rows = []
    blk_i = 0
    with open(a.out, 'wb') as f:
        for eps in eps_list:
            d = np.load(os.path.join(a.pert_dir, f"delta_{a.method}_e{eps:g}.npy"))  # (nper, ACTIVE)
            if len(d) < nfr:
                raise SystemExit(
                    f"BANK TOO SHORT: {len(d)} deltas for {nfr} frame slots. Frames "
                    f"{len(d)}..{nfr - 1} would be transmitted UNPERTURBED, capping every "
                    f"result at {100 * len(d) / nfr:.1f}% and reading as a 'structural "
                    f"ceiling' -- this is exactly what produced the phantom ~70% ceiling in "
                    f"all 8/12-8/13 results. Re-craft with --nper {nfr}. Do NOT tile the bank: "
                    f"delta[k] is crafted against ch0 frame k, so reusing it on frame k+len(d) "
                    f"applies a perturbation optimised for different IQ.")
            for psi in PHASES_DEG:
                rot = (a.scale * np.exp(1j * np.deg2rad(psi))).astype(np.complex64)
                buf = np.zeros(blk, dtype=np.complex64)
                for p in range(a.passes):
                    base = p * L
                    for k in range(nfr):
                        s = base + k * FRAME_PERIOD + PRE_ROLL
                        buf[s:s + ACTIVE] += d[k] * rot
                buf.tofile(f)
                rows.append(dict(block=blk_i, method=a.method, epsilon=eps, phase_deg=psi,
                                 start_sample=blk_i * blk, n_samples=blk,
                                 score_from=blk_i * blk + (a.passes - 1) * L,
                                 score_to=(blk_i + 1) * blk))
                blk_i += 1
            print(f"  eps={eps:g} done ({blk_i}/{len(eps_list)*len(PHASES_DEG)} blocks)", flush=True)

    with open(a.index_out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out} ({os.path.getsize(a.out)/1e9:.2f} GB) + {a.index_out} ({len(rows)} blocks)")


if __name__ == '__main__':
    main()
