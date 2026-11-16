#!/usr/bin/env python3
"""exp3_map_detected_ids.py — recover the frame_id of every burst the DETECTOR finds
in a received capture, so crafting and the replay can be built from the SAME frames.

Why this exists
---------------
Crafting must run against the RECEIVED signal (that is where the victim's fingerprint
lives), but the replay ch0 must be built from the TRANSMIT-side frame_run (replaying a
reception re-radiates that reception's channel and stamps the fingerprint twice).
Those two paths select frames differently:

    exp3_make_ota_perturbations.py --from-capture   burst-detector offsets, time order
    build_ch0_matched.py                            frame_id matched against rx_frames

and they disagree: the detector finds 103 bursts on a device_5 run while the
decoder logs 134 frame_ids. Threshold sweeping does NOT reconcile them - detection
plateaus at 103-104 from thr_mult 2.0 down to 1.2 and only reaches 119 at settings that
start admitting noise. The extra frames are not marginal detections; the detector simply
does not segment them.

delta[k] is crafted against frame k specifically, so a mismatched pairing applies each
perturbation to the wrong frame (this is the same reason tiling a short bank is invalid).

What it does
------------
Decodes each detected burst and reads its frame_id from the payload
(b'FRID' + uint32_be(frame_id) + 'x' fill, exp3_wifi_tx.py:169), emitting the ids IN
DETECTOR ORDER. Feed that list to build_ch0_matched.py --ids so ch0 holds exactly the
frames the deltas were crafted against, in the same order.

Bursts whose id cannot be recovered are reported and dropped from BOTH sides, so the
correspondence stays exact rather than silently drifting.

Run:
  python3 exp3_map_detected_ids.py --capture .../device_5/clean_run_3.bin \
      --out detected_ids_d6_run3.csv
"""
import argparse
import os
import sys

import numpy as np

from exp3_extract_frames import extract_frames_for_file
from exp3_fp_model import PRE_ROLL, ACTIVE, WIN

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp3_ber_eval as B

FRAME_LEN = PRE_ROLL + ACTIVE


def ids_from_segment(seg, known=None):
    """Decode one burst and return its frame_id, or None."""
    import tempfile
    tmp = tempfile.mktemp(suffix='_burst.bin')
    seg.astype(np.complex64).tofile(tmp)
    saved = os.dup(1)
    dump = tmp + '.txt'
    try:
        with open(dump, 'w') as f:
            os.dup2(f.fileno(), 1)
            try:
                tb = B.ber_rx(tmp)
                tb.run()
            finally:
                sys.stdout.flush()
                os.dup2(saved, 1)
                os.close(saved)
        frames = [m for m in B.parse_frames(open(dump).read()) if B._is_ours(m)]
        if not frames:
            return None
        raw = np.frombuffer(frames[0][B.MAC_HEADER + B.ID_LO:B.MAC_HEADER + B.ID_HI],
                            dtype=np.uint8)
        fid = int.from_bytes(raw.tobytes(), 'big')
        if known and fid not in known:                # repair against the known set
            kb = np.stack([np.frombuffer(int(k).to_bytes(4, 'big'), dtype=np.uint8)
                           for k in sorted(known)])
            d = B._POPCOUNT[np.bitwise_xor(kb, raw)].sum(axis=1)
            fid = sorted(known)[int(d.argmin())]
        return fid
    finally:
        for p in (tmp, dump):
            if os.path.exists(p):
                os.remove(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture', required=True, help='received clean_run_N.bin')
    ap.add_argument('--rx-frames', default=None,
                    help='rx_frames_run_N.jsonl — its ids constrain/repair the decode')
    ap.add_argument('--out', required=True, help='csv: index,start_sample,frame_id')
    ap.add_argument('--floor-pct', type=float, default=20.0)
    ap.add_argument('--thr-mult', type=float, default=2.0)
    ap.add_argument('--pad', type=int, default=2000,
                    help='extra samples each side of a burst, so the decoder can sync')
    a = ap.parse_args()

    known = None
    if a.rx_frames and os.path.exists(a.rx_frames):
        import json
        ids = []
        for line in open(a.rx_frames):
            line = line.strip()
            if not line:
                continue
            try:
                v = json.loads(line).get('frame_id')
                if v is not None:
                    ids.append(int(v))
            except Exception:
                pass
        known = set(ids)
        print(f"  known ids from rx_frames: {len(known)}")

    x = np.fromfile(a.capture, dtype=np.complex64)
    _, starts = extract_frames_for_file(a.capture, floor_pct=a.floor_pct,
                                        thr_mult=a.thr_mult)[:2]
    starts = [s for s in starts if s + FRAME_LEN + WIN <= len(x)]
    print(f"  {os.path.basename(a.capture)}: {len(x)} samples, {len(starts)} bursts detected")

    rows, miss = [], 0
    for i, s in enumerate(starts):
        lo = max(0, s - a.pad)
        hi = min(len(x), s + FRAME_LEN + WIN + a.pad)
        fid = ids_from_segment(x[lo:hi], known)
        if fid is None:
            miss += 1
            print(f"    burst {i:3d} @ {s}: NO ID (dropped)")
            continue
        rows.append((i, int(s), fid))
        if (i + 1) % 20 == 0:
            print(f"    ... {i+1}/{len(starts)} bursts, {len(rows)} ids so far", flush=True)

    dup = len(rows) - len({r[2] for r in rows})
    with open(a.out, 'w') as f:
        f.write('index,start_sample,frame_id\n')
        for i, s, fid in rows:
            f.write(f'{i},{s},{fid}\n')
    print(f"\n  recovered {len(rows)}/{len(starts)} ids  (dropped {miss}, duplicate ids {dup})")
    print(f"  ids span {min(r[2] for r in rows)}..{max(r[2] for r in rows)}")
    print(f"  wrote {a.out}")
    if dup:
        print("  WARNING duplicate frame_ids — pairing would be ambiguous; inspect before use")


if __name__ == '__main__':
    main()
