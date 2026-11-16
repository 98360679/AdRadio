#!/usr/bin/env python3
"""build_ch0_matched.py — build the replay ch0 the CORRECT way.

The receiver logs a frame_id for every frame it decodes (rx_frames_run_N.jsonl). Those ids
are carried back to the transmit side and used to cut the matching CLEAN baseband frames out
of frame_run_N.bin, via the offsets in frame_index_run_N.csv. The result is a waveform of
known-good frames that has never been through the air.

Do NOT replay a received capture (clean_run_N.bin). That file already carries the victim's
hardware signature, the channel, and that reception's bit errors; replaying it transmits all
three again and stamps the fingerprint a second time. Measured over the air, the two differ:

    received capture   72 frames   28 bit errors   BER 9.74e-05   80.6% error-free
    frame-id matched  141 frames    5 bit errors   BER 1.49e-05   94%   error-free

The matched file is also already at transmit amplitude (peak ~1.0), so it needs no scaling —
the scale factor a received capture required was itself a symptom of using the wrong source.
"""
import argparse, csv, json, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dev-dir', required=True, help='enrollment device_N directory')
    ap.add_argument('--run', type=int, required=True, help='run id with a frame_run/frame_index pair')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ids', default=None,
                    help='csv with a frame_id column giving the EXACT frames, IN ORDER '
                         '(from exp3_map_detected_ids.py). Required when the delta bank was '
                         'crafted --from-capture, so ch0 matches it frame-for-frame.')
    ap.add_argument('--repeat', type=int, default=1, help='concatenate the matched set this many times')
    ap.add_argument('--period', type=int, default=None,
                    help='place each matched frame at k*period with silence between. The frames '
                         'sit ~15.7k apart in frame_run, a ~93%% duty cycle, and the burst '
                         'detector estimates its floor as a low percentile of the file - with no '
                         'gaps that percentile lands INSIDE the signal and detection fails. '
                         'Pad to the 65661 bundle spacing (~24%% duty) to keep the detector honest.')
    a = ap.parse_args()

    D = a.dev_dir
    idx_p = f'{D}/frame_index_run_{a.run}.csv'
    tx_p = f'{D}/frame_run_{a.run}.bin'
    rx_p = f'{D}/rx_frames_run_{a.run}.jsonl'
    for p in (idx_p, tx_p, rx_p):
        if not os.path.exists(p):
            raise SystemExit(f'missing {p} — this run has no TX-side frame record; pick a run that does')

    idx = {int(r['frame_id']): (int(r['burst_offset']), int(r['burst_len']))
           for r in csv.DictReader(open(idx_p))}
    rx_ids = [json.loads(l)['frame_id'] for l in open(rx_p)]   # reported either way
    if a.ids:
        # Explicit id list, IN ORDER, from exp3_map_detected_ids.py. Use this when the
        # deltas were crafted --from-capture: crafting selects frames by burst-detector
        # offset while rx_frames lists every decoded id, and the two sets differ (103 vs
        # 134 on 8_15 device_5 run 3). delta[k] is crafted against frame k, so ch0 must
        # hold exactly the crafted frames in exactly that order or every delta lands on
        # the wrong frame.
        want = [int(r['frame_id']) for r in csv.DictReader(open(a.ids))]
        keep = [f for f in want if f in idx]
        missing = [f for f in want if f not in idx]
        if missing:
            raise SystemExit(f'{len(missing)} id(s) from {a.ids} are not in the tx index '
                             f'(first few {missing[:5]}) — ch0 would not match the delta bank')
        print(f'  id list: {len(keep)} frames from {os.path.basename(a.ids)} (order preserved)')
    else:
        keep = [f for f in rx_ids if f in idx]
    if not keep:
        raise SystemExit('no rx frame_id matched the tx index')

    base = np.fromfile(tx_p, dtype=np.complex64)
    if a.period:
        blen = idx[keep[0]][1]
        if a.period < blen:
            raise SystemExit(f'--period {a.period} is shorter than a burst ({blen})')
        one = np.zeros(len(keep) * a.period, dtype=np.complex64)
        for k, f in enumerate(keep):
            o, l = idx[f]
            one[k * a.period: k * a.period + l] = base[o:o + l]
    else:
        one = np.concatenate([base[o:o + l] for f in keep for o, l in [idx[f]]]).astype(np.complex64)
    out = np.tile(one, a.repeat) if a.repeat > 1 else one
    out.tofile(a.out)

    per = int(np.median(np.diff([idx[f][0] for f in keep]))) if len(keep) > 1 else idx[keep[0]][1]
    m = np.abs(out)
    print(f'tx-side frames indexed : {len(idx)}')
    print(f'rx decoded frame_ids   : {len(rx_ids)}')
    print(f'matched (kept)         : {len(keep)}')
    sp = a.period or idx[keep[0]][1]
    print(f'frame spacing written  : {sp} samples ({sp/5e6*1e3:.2f} ms), '
          f'duty {100*idx[keep[0]][1]/sp:.0f}%')
    print(f'wrote {a.out}: {len(out)} samples = {len(out)/5e6:.3f} s, {len(keep)*a.repeat} frames')
    print(f'  peak {m.max():.3f}  rms {np.sqrt((m**2).mean()):.4f}   (already at transmit level)')
    with open(a.out + '.frames.json', 'w') as f:
        json.dump(dict(run=a.run, kept=keep, repeat=a.repeat,
                       burst_len=idx[keep[0]][1], source=tx_p), f)


if __name__ == '__main__':
    main()
