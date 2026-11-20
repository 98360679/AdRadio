#!/usr/bin/env python3
"""score_2ch.py — recover the epsilon x phase blocks from a 2-channel OTA capture
and score where the victim's frames landed.

Block recovery: ch0 is a dense periodic replay, so the first detected frame in the capture
is TX sample 0. Every later frame's offset from that anchor gives its block directly.
Only the SECOND pass of each block is scored (the first absorbs the boundary).

Blind 4-phase: the inter-channel RF phase is fixed for the session but unknown, so each
epsilon is reported as max-over-psi (matching how the sweep was built).
"""
import os as _os, sys as _sys
_R = _os.path.abspath(_os.path.dirname(__file__))
while _R != _os.path.dirname(_R) and not _os.path.isdir(_os.path.join(_R, ".git")):
    _R = _os.path.dirname(_R)
_sys.path.insert(0, _os.path.join(_R, "lib"))
_os.chdir(_R)
import argparse
import json
import re, csv, os, numpy as np, torch
from extract_frames import extract_frames_for_file
from fp_model import PRE_ROLL, ACTIVE, WIN, NUM_CLASSES, DEVICE_NAMES
from cvnn import CVNN, iq_to_complex, to_reim

STARTS = list(range(PRE_ROLL, PRE_ROLL + ACTIVE - WIN + 1, WIN))


@torch.no_grad()
def frame_pred(model, frame, dev):
    w = iq_to_complex(np.stack([frame[s:s + WIN] for s in STARTS]), cfo_correct=False)
    re, im = to_reim(torch.from_numpy(w), dev)
    votes = model(re, im).argmax(1)
    return int(torch.bincount(votes, minlength=NUM_CLASSES).argmax())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture', required=True)
    ap.add_argument('--blocks', required=True, help='block index csv from build_ch1.py')
    ap.add_argument('--ch0', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--arch', choices=['cvnn', 'cnn1d', 'freqcvnn'], default='cvnn',
                    help='classifier architecture (default cvnn)')
    ap.add_argument('--target', default='device_2')
    ap.add_argument('--victim', default='device_5')
    ap.add_argument('--out', required=True)
    ap.add_argument('--thr-mult', type=float, default=4.0)
    ap.add_argument('--pert-dir', default=None, help='delta bank, for the realized PSR column')
    ap.add_argument('--method', default=None, help='method name, for the PSR lookup')
    ap.add_argument('--ber', action='store_true',
                    help='decode each best-phase block (subprocess) for frames decoded / bit errors / BER')
    ap.add_argument('--frames-per-pass', type=int, default=None,
                    help='frames TRANSMITTED per (eps,psi) point. Auto-read from '
                         '<ch0>.frames.json when present. This is a property of the waveform, '
                         'not something the scorer can measure - frames_scored is always a '
                         'SUBSET of it, reduced by --guard-frac.')
    ap.add_argument('--decode-passes', type=int, default=1,
                    help='BER decode passes per epsilon, unioned by frame id. The decoder '
                         'is non-deterministic and one pass loses ~20%% of frames even on a '
                         'clean file, so frames_missing is meaningless at 1. 3 is the tested '
                         'value. Costs decode CPU only. (default %(default)s)')
    ap.add_argument('--guard-frac', type=float, default=0.0,
                    help='skip frames in the first fraction of each block. Use when the block is '
                         'a single ch0 pass, so there is no discarded settle pass absorbing the '
                         'delta transition at the block boundary.')
    ap.add_argument('--select', choices=['target', 'fool'], default='target',
                    help='which metric picks the best phase per epsilon — use "fool" for '
                         'untargeted runs, where there is no intended class')
    a = ap.parse_args()

    L = os.path.getsize(a.ch0) // 8
    blocks = list(csv.DictReader(open(a.blocks)))
    npass = int(blocks[0]['n_samples']) // L
    n_sent = a.frames_per_pass
    if n_sent is None:
        for cand in (a.ch0 + '.frames.json', re.sub(r'_[a-z0-9.]+\.bin$', '.bin', a.ch0) + '.frames.json'):
            if os.path.exists(cand):
                n_sent = len(json.load(open(cand))['kept']); break
    blk = int(blocks[0]['n_samples'])
    tgt_i = DEVICE_NAMES.index(a.target); vic_i = DEVICE_NAMES.index(a.victim)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    from cnn1d import CNN1DFp
    from freqcvnn import FreqCVNN
    ARCH = {'cvnn': CVNN, 'cnn1d': CNN1DFp, 'freqcvnn': FreqCVNN}
    m = ARCH[a.arch](NUM_CLASSES).to(dev)
    m.load_state_dict(torch.load(a.model, map_location=dev, weights_only=False)); m.eval()

    frames, starts = extract_frames_for_file(a.capture, floor_pct=20.0, thr_mult=a.thr_mult)[:2]
    if not frames:
        print("NO FRAMES DETECTED in the capture — nothing to score."); return 1
    anchor = starts[0]
    print(f"capture: {len(frames)} frames, anchor at sample {anchor}, "
          f"block={blk} ({npass} passes of {L})")

    agg = {}
    span = {}                      # (eps,psi) -> (lo,hi) capture samples of the scored pass
    used = 0
    for fr, s in zip(frames, starts):
        off = s - anchor
        if off < 0:
            continue
        b = off // blk
        if b >= len(blocks):
            continue
        if (off % blk) // L != npass - 1:      # score the last pass only
            continue
        if a.guard_frac and (off % blk) < a.guard_frac * blk:
            continue                           # boundary guard when there is no settle pass
        p = frame_pred(m, fr, dev)
        r = blocks[b]
        key = (float(r['epsilon']), float(r['phase_deg']))
        d = agg.setdefault(key, np.zeros(NUM_CLASSES, dtype=int))
        d[p] += 1
        used += 1
        lo = anchor + b * blk + (npass - 1) * L
        span[key] = (lo, lo + L)
    print(f"scored {used} frames across {len(agg)} (eps, phase) points")

    psr = {}
    if a.pert_dir and a.method:
        for r in csv.DictReader(open(os.path.join(a.pert_dir, 'manifest.csv'))):
            if r['method'] == a.method:
                psr[float(r['epsilon'])] = float(r['realized_psr_db'])

    rows = []
    for (eps, psi), d in sorted(agg.items()):
        n = int(d.sum())
        rows.append(dict(epsilon=eps, phase_deg=psi,
                         frames_sent=n_sent if n_sent else '', frames_scored=n,
                         target_pct=round(100 * d[tgt_i] / n, 2) if n else '',
                         fool_pct=round(100 * (n - d[vic_i]) / n, 2) if n else '',
                         landed=DEVICE_NAMES[int(d.argmax())],
                         landed_pct=round(100 * d.max() / n, 2) if n else '',
                         **{DEVICE_NAMES[i]: int(d[i]) for i in range(NUM_CLASSES)}))

    # pick the best phase per epsilon, then decode ONLY those blocks (one subprocess each)
    by_eps = {}
    for r in rows:
        by_eps.setdefault(r['epsilon'], []).append(r)
    selkey = 'target_pct' if a.select == 'target' else 'fool_pct'
    best_of = {e: max(v, key=lambda r: r[selkey] if r[selkey] != '' else -1)
               for e, v in by_eps.items()}

    if a.ber:
        import subprocess, tempfile      # json is imported at module level; re-importing it
                                        # HERE would make it local to main() and shadow the
                                        # module-level use above (UnboundLocalError)
        BEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp3_ber_eval.py')
        for eps in sorted(best_of):
            b = best_of[eps]
            lo, hi = span[(eps, b['phase_deg'])]
            seg = np.fromfile(a.capture, dtype=np.complex64, count=hi - lo, offset=lo * 8)
            tmp = tempfile.mktemp(suffix='_seg.bin'); seg.tofile(tmp)
            outj = tmp + '.json'
            cmd = ['python3', BEV, '--file', tmp, '--json', outj]
            sidecar = a.ch0 + '.frames.json'
            if os.path.exists(sidecar):     # ground-truth ids -> true BER + loss account
                cmd += ['--ids-json', sidecar]
            if a.decode_passes > 1:
                cmd += ['--decode-passes', str(a.decode_passes)]
            try:
                subprocess.run(cmd, capture_output=True, timeout=1200)
                j = json.load(open(outj)) if os.path.exists(outj) else {}
            except Exception:
                j = {}
            finally:
                for p_ in (tmp, outj):
                    if os.path.exists(p_):
                        os.remove(p_)
            b['frames_decoded'] = j.get('frames_ours_aligned', 0)
            b['bit_errors'] = j.get('bit_errors', '')
            b['ber'] = j.get('ber', '')
            b['bits_compared'] = j.get('known_bits_compared', '')
            # id-derived loss account. frames_decoded counts what the decoder found
            # in an arithmetically-placed window and can exceed frames_sent; these
            # are matched against the ch0 id set, so they actually add up.
            b['frames_unique'] = j.get('frames_unique', '')
            b['frames_missing'] = j.get('frames_missing', '')
            b['frames_duplicate'] = j.get('frames_duplicate', '')
            b['ids_repaired'] = j.get('ids_repaired', '')
            b['legacy_ber'] = j.get('legacy_ber', '')
            b['frames_first_pass'] = j.get('frames_first_pass', '')
            b['frames_dumped'] = j.get('frames_dumped', '')
            b['frames_rejected_short'] = j.get('frames_rejected_short', '')
            b['frames_rejected_addr'] = j.get('frames_rejected_addr', '')
            b['frames_by_addr'] = j.get('frames_by_addr', '')
            b['frames_by_fill'] = j.get('frames_by_fill', '')
            print(f"  ber eps={eps:g}: decoded {b['frames_decoded']} "
                  f"biterr {b['bit_errors']} ber {b['ber']}", flush=True)

    with open(a.out, 'w', newline='') as f:
        cols = list(rows[0].keys())
        for extra in ('frames_decoded', 'frames_unique', 'frames_missing',
                      'frames_duplicate', 'ids_repaired',
                      'frames_first_pass', 'frames_dumped',
                      'frames_rejected_short', 'frames_rejected_addr',
                      'frames_by_addr', 'frames_by_fill',
                      'bit_errors', 'ber', 'bits_compared', 'legacy_ber'):
            if any(extra in r for r in rows) and extra not in cols:
                cols.append(extra)          # guard: the BER pass already put these on the rows
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

    print(f"\n=== 2-channel OTA, targeted {a.target}, blind 4-phase (max over psi) ===")
    hdr = f"{'eps':>7} {'PSR dB':>8} {'sent':>6} {'scored':>7} {'decoded':>8} {'fool%':>7} {'bit_err':>9} {'BER':>10} {'landed':>9}"
    print(hdr)
    for eps in sorted(best_of):
        b = best_of[eps]
        ber = b.get('ber', '')
        ber_s = f"{ber:.2e}" if isinstance(ber, float) else '-'
        print(f"{eps:7.3f} {psr.get(eps, float('nan')):8.1f} "
              f"{b['frames_sent'] or '-':>6} {b['frames_scored']:7d} "
              f"{b.get('frames_decoded', '-'):>8} {b['fool_pct']:7.1f} "
              f"{b.get('bit_errors', '-'):>9} {ber_s:>10} {b['landed']:>9}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
