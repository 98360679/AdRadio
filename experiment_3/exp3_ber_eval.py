#!/usr/bin/env python3
"""
exp3_ber_eval.py — true bit-error-rate of an OTA capture (perturbed or clean)
─────────────────────────────────────────────────────────────────────────────

The fingerprint eval (exp3_rx_eval.py) tells us whether the adversarial
perturbation fools DEVICE IDENTIFICATION. This tool answers the orthogonal
question for the same capture: does the perturbation also damage the LEGITIMATE
LINK — i.e. what is the bit-error rate of the WiFi payload underneath?

Why a custom harness (and not exp3_wifi_rx.py):
  decode_mac DROPS every frame whose CRC fails (decode_mac.cc: "checksum wrong
  -- dropping"), so the stock 'out' port only ever yields CRC-OK frames — you
  can measure packet success from it, but never the bit errors on the broken
  frames. decode_mac's print_output() however dumps the decoded MAC bytes for
  EVERY completed frame, BEFORE the CRC check, whenever the block is built with
  debug=True. We run the identical sync/equalize/decode chain with debug=True,
  capture that per-frame byte dump, and compare against the known transmitted
  payload bit-for-bit.

Ground truth (exp3_wifi_tx.py:169, verbatim):
      payload_bytes = b'FRID' + struct.pack('>I', frame_id) + b'x' * (PDU_LENGTH - 8)
  so the 500-byte MSDU is EXACTLY
      [0:4]   b'FRID' magic
      [4:8]   uint32 big-endian frame_id
      [8:500] 0x78 ('x') fill
  The decoded 802.11 MAC frame is [24B MAC header][500B MSDU][4B FCS]; the MSDU
  starts at offset 24, and "our" frames are dst 42:42:.. / src 23:23:.. .

  The reference payload is therefore FULLY DETERMINED by the frame id — every one
  of the 500 bytes is known a priori. We recover the id from bytes 4:8 (repairing
  it against the known id set when it is itself corrupted) and compare bit-for-bit
  against the reconstructed truth. That gives a TRUE ber, and — because the ids
  identify WHICH frames arrived — a real sent/received/missing/duplicate account.

  HISTORY — why this replaced the previous method. The old analyse() had no ground
  truth: it took the per-position MAJORITY BYTE across the decoded frames as the
  reference and counted deviations from it. Two consequences, both measured on
  ch0_matched65k.bin (a CLEAN transmit-side file that must score exactly 0):
    * It reported a 6-bit-error floor (ber 1.11e-05) on every capture we ever ran.
      Those 6 "errors" are payload byte 6 — the frame id's HIGH byte, 0x00 for ids
      250-255 and 0x01 for 256-401. With a 0.5 mode threshold the 0x01 majority was
      declared "constant", so the 6 low-id frames were scored as 1-bit errors each.
      The floor was the id being mistaken for fill; the true clean BER is 0.
    * Measuring against the majority of the survivors means a common-mode error
      hitting every frame reads as a PERFECT link. At high epsilon, where only
      ~34 of 136 frames decode, "ber 0.00e+00" only ever meant those few agreed
      with each other — not that they were correct.
  The legacy numbers are still computed and reported under legacy_* so previously
  published results stay comparable.

Run:
  python3 exp3_ber_eval.py --file /media/.../perturbed/capture_1.bin
  python3 exp3_ber_eval.py --file <clean.bin> --tag clean --json out.json
"""

import os
import re
import sys
import json
import argparse
import tempfile
import contextlib

sys.path.append(os.environ.get('GRC_HIER_PATH', os.path.expanduser('~/.grc_gnuradio')))

import numpy as np
from gnuradio import gr, blocks, fft
from gnuradio.fft import window
import ieee802_11


# ── frame geometry ─────────────────────────────────────────────────────
PDU_LENGTH = 500         # MSDU length in bytes
MAC_HEADER = 24          # bytes before the MSDU in the decoded 802.11 frame
MAGIC = b'FRID'          # MSDU bytes 0:4  (exp3_wifi_tx.py:169)
ID_LO, ID_HI = 4, 8      # MSDU bytes 4:8 — uint32 big-endian frame_id
FILL = 0x78              # MSDU bytes 8:500 — b'x'. This is the TRANSMITTED byte;
                         # the descrambler recovers it exactly, as it does 'FRID'.

# RX chain constants — copied verbatim from exp3_wifi_rx.py (the known-good chain)
SAMP_RATE    = 5e6
FREQ         = 2.45e9
WINDOW_SIZE  = 48
SYNC_LENGTH  = 320
CHAN_EST     = 0         # LS
EQ_BANDWIDTH = 200000

_POPCOUNT = np.array([bin(i).count('1') for i in range(256)], dtype=np.int64)


class ber_rx(gr.top_block):
    """Stock gr-ieee802-11 receive chain on a file, decode_mac in debug mode."""
    def __init__(self, path, chan_est=CHAN_EST):
        gr.top_block.__init__(self, "Exp3 BER RX", catch_exceptions=True)
        window_size, sync_length = WINDOW_SIZE, SYNC_LENGTH

        src = blocks.file_source(gr.sizeof_gr_complex, path, False)

        self.sync_short = ieee802_11.sync_short(0.56, 2, False, False)
        self.sync_long = ieee802_11.sync_long(sync_length, False, False)
        self.equalizer = ieee802_11.frame_equalizer(
            ieee802_11.Equalizer(chan_est), FREQ, EQ_BANDWIDTH, False, False)
        # debug=True -> print_output() dumps EVERY completed frame, pre-CRC-drop.
        self.decode = ieee802_11.decode_mac(False, True)

        self.fftb = fft.fft_vcc(64, True, window.rectangular(64), True, 1)
        self.s2v = blocks.stream_to_vector(gr.sizeof_gr_complex, 64)
        self.mult = blocks.multiply_vcc(1)
        self.mavg_c = blocks.moving_average_cc(window_size, 1, 4000, 1)
        self.mavg_f = blocks.moving_average_ff(window_size + 16, 1, 4000, 1)
        self.div = blocks.divide_ff(1)
        self.delay16 = blocks.delay(gr.sizeof_gr_complex, 16)
        self.delay_sync = blocks.delay(gr.sizeof_gr_complex, sync_length)
        self.conj = blocks.conjugate_cc()
        self.mag2 = blocks.complex_to_mag_squared(1)
        self.mag = blocks.complex_to_mag(1)

        self.connect((self.mag, 0), (self.div, 0))
        self.connect((self.mag2, 0), (self.mavg_f, 0))
        self.connect((self.conj, 0), (self.mult, 1))
        self.connect((self.delay_sync, 0), (self.sync_long, 1))
        self.connect((self.delay16, 0), (self.conj, 0))
        self.connect((self.delay16, 0), (self.sync_short, 0))
        self.connect((self.div, 0), (self.sync_short, 2))
        self.connect((self.mavg_f, 0), (self.div, 1))
        self.connect((self.mavg_c, 0), (self.mag, 0))
        self.connect((self.mavg_c, 0), (self.sync_short, 1))
        self.connect((self.mult, 0), (self.mavg_c, 0))
        self.connect((self.s2v, 0), (self.fftb, 0))
        self.connect((self.fftb, 0), (self.equalizer, 0))
        self.connect((self.equalizer, 0), (self.decode, 0))
        self.connect((self.sync_long, 0), (self.s2v, 0))
        self.connect((self.sync_short, 0), (self.delay_sync, 0))
        self.connect((self.sync_short, 0), (self.sync_long, 0))
        self.connect((src, 0), (self.mag2, 0))
        self.connect((src, 0), (self.delay16, 0))
        self.connect((src, 0), (self.mult, 0))


# ── parse decode_mac's debug byte-dump ─────────────────────────────────
# Format emitted by decode_mac_impl::print_output() (debug=True):
#   "psdu size<N>"
#   <N hex bytes, space/newline separated>
#   <ascii rendering>
_PSDU_RE = re.compile(r'psdu size(\d+)')
_HEXTOK_RE = re.compile(r'\b[0-9a-f]{2}\b')


def parse_frames(dump_text):
    """Yield decoded MAC frames (bytes) from a decode_mac debug dump."""
    lines = dump_text.splitlines()
    i = 0
    while i < len(lines):
        m = _PSDU_RE.search(lines[i])
        if not m:
            i += 1
            continue
        n = int(m.group(1))
        # collect hex tokens from following lines until we have n bytes
        toks = []
        j = i + 1
        while j < len(lines) and len(toks) < n:
            row = _HEXTOK_RE.findall(lines[j])
            if not row:                 # blank or ascii line -> hex block ended
                if toks:
                    break
                j += 1
                continue
            toks.extend(row)
            j += 1
        if len(toks) >= n:
            yield bytes(int(t, 16) for t in toks[:n])
        i = j


# A frame of ours whose MAC header is destroyed still carries 492 fill bytes at
# payload[8:500]. Random traffic matches FILL about 1 byte in 256 (expected 1.9 of
# 492, sd 1.4), so 60 is ~40 sigma above noise; a real frame stays above it up to
# ~23% BER, by which point it is destroyed anyway. Gating on the PAYLOAD instead of
# only the header is the point: the header addresses are the first thing to go, and
# the address-only gate was silently discarding up to 48 damaged frames per block --
# preferentially the most damaged ones, which biased every high-epsilon BER low.
FILL_MATCH_MIN = 60


def _addr_ok(mac):
    hdr = np.frombuffer(mac[:MAC_HEADER], dtype=np.uint8)
    return int((hdr[4:10] == 0x42).sum()) + int((hdr[10:16] == 0x23).sum()) >= 8


def _fill_ok(mac):
    fill = np.frombuffer(mac[MAC_HEADER + ID_HI:MAC_HEADER + PDU_LENGTH], dtype=np.uint8)
    return int((fill == FILL).sum()) >= FILL_MATCH_MIN


def _is_ours(mac):
    """Ours by MAC address OR by surviving payload structure (see FILL_MATCH_MIN)."""
    if len(mac) < MAC_HEADER + PDU_LENGTH:
        return False
    return _addr_ok(mac) or _fill_ok(mac)


def truth_payload(frame_id):
    """The exact 500 bytes exp3_wifi_tx.py transmits for this id."""
    ref = np.full(PDU_LENGTH, FILL, dtype=np.uint8)
    ref[0:4] = np.frombuffer(MAGIC, dtype=np.uint8)
    ref[ID_LO:ID_HI] = np.frombuffer(
        int(frame_id).to_bytes(4, 'big'), dtype=np.uint8)
    return ref


def _match_ids(P, known_ids):
    """Recover each frame's id from bytes 4:8, repairing it against known_ids.

    A corrupted id would otherwise be reconstructed into a WRONG reference, which
    silently hides the very errors we are counting. Since all references differ
    only in bytes 4:8, nearest-reference is nearest on those four bytes alone.
    Returns (ids, n_repaired); ids[i] is None only when known_ids is empty.
    """
    raw = [int.from_bytes(P[i, ID_LO:ID_HI].tobytes(), 'big') for i in range(len(P))]
    if not known_ids:
        return raw, 0
    known = sorted(set(int(x) for x in known_ids))
    kbytes = np.stack([np.frombuffer(k.to_bytes(4, 'big'), dtype=np.uint8)
                       for k in known])                       # (K, 4)
    ids, n_rep = [], 0
    kset = set(known)
    for i, r in enumerate(raw):
        if r in kset:
            ids.append(r)
            continue
        d = _POPCOUNT[np.bitwise_xor(kbytes, P[i, ID_LO:ID_HI])].sum(axis=1)
        ids.append(known[int(d.argmin())])                    # ties -> smallest id
        n_rep += 1
    return ids, n_rep


def classify_dumped(frames):
    """Split what decode_mac emitted into ours / rejected, with the REASON.

    _is_ours() applies two tests and previously reported only the survivors, so a
    frame of ours corrupted past the address check was discarded by OUR filter and
    never appeared in any count. rejected_addr is the number that matters: those are
    candidate frames of ours destroyed badly enough to lose their MAC header.
    (Ambient WiFi and false syncs also land in rejected_* and cannot be separated
    from destroyed frames by header alone.)
    """
    n_short = n_rej = n_addr = n_fill = 0
    for m in frames:
        if len(m) < MAC_HEADER + PDU_LENGTH:
            n_short += 1
        elif _addr_ok(m):
            n_addr += 1                  # header intact
        elif _fill_ok(m):
            n_fill += 1                  # header destroyed, payload still recognisably ours
        else:
            n_rej += 1                   # foreign traffic / false sync
    return {'frames_dumped': len(frames), 'frames_ours': n_addr + n_fill,
            'frames_by_addr': n_addr, 'frames_by_fill': n_fill,
            'frames_rejected_short': n_short, 'frames_rejected_addr': n_rej}


def dedupe_by_id(frames, known_ids=None):
    """Merge frames from several decode passes, keeping one instance per frame id.

    The gr-ieee802-11 chain is NON-DETERMINISTIC on identical input: GNU Radio's
    scheduler splits the stream on different buffer boundaries each run, and frames
    straddling a boundary are lost by sync_short/sync_long. Measured on the CLEAN
    ch0 file (no channel, no noise): 105/109/110 of 136 frames over three runs, with
    ZERO ids missing from all three and only ~86 ids common to any pair. One pass
    therefore under-reports arrivals by ~20% and cannot be read as channel loss.

    Selection is FIRST-SEEN, deliberately: picking the lowest-error instance of each
    id would bias BER downward by construction. Ids are repaired to the known set
    before deduping so a corrupted id cannot masquerade as an extra frame.
    """
    ours = [f for f in frames if _is_ours(f)]
    if not known_ids:
        keyed, out = set(), []
        for f in ours:
            k = f[MAC_HEADER + ID_LO:MAC_HEADER + ID_HI]
            if k not in keyed:
                keyed.add(k)
                out.append(f)
        return out
    known = sorted(set(int(x) for x in known_ids))
    kset = set(known)
    kbytes = np.stack([np.frombuffer(k.to_bytes(4, 'big'), dtype=np.uint8)
                       for k in known])
    seen, out = set(), []
    for f in ours:
        raw = np.frombuffer(f[MAC_HEADER + ID_LO:MAC_HEADER + ID_HI], dtype=np.uint8)
        fid = int.from_bytes(raw.tobytes(), 'big')
        if fid not in kset:
            fid = known[int(_POPCOUNT[np.bitwise_xor(kbytes, raw)].sum(axis=1).argmin())]
        if fid not in seen:
            seen.add(fid)
            out.append(f)
    return out


def analyse(frames, stable_frac=0.5, known_ids=None):
    """True BER against the id-derived ground truth, plus the legacy mode metric."""
    payloads = [np.frombuffer(m[MAC_HEADER:MAC_HEADER + PDU_LENGTH], dtype=np.uint8)
                for m in frames if _is_ours(m)]
    n_aligned = len(payloads)
    if n_aligned == 0:
        return {'frames_dumped': len(frames), 'frames_ours_aligned': 0}

    P = np.stack(payloads)                          # (F, 500)
    F = P.shape[0]

    # ── TRUE metric: every one of the 500 bytes is known from the frame id ──
    ids, n_repaired = _match_ids(P, known_ids)
    REF = np.stack([truth_payload(i) for i in ids])            # (F, 500)
    tdiff = np.bitwise_xor(P, REF)
    t_biterr_per_frame = _POPCOUNT[tdiff].sum(axis=1)
    t_total_bits = int(PDU_LENGTH * 8 * F)
    t_total_biterr = int(t_biterr_per_frame.sum())
    t_clean = int((t_biterr_per_frame == 0).sum())

    # ── accounting: the ids say WHICH frames arrived, so loss is measurable ──
    seen = [int(i) for i in ids]
    uniq = set(seen)
    exp_ids = set(int(x) for x in known_ids) if known_ids else set()
    acct = {
        'frames_expected': len(exp_ids) if exp_ids else None,
        'frames_unique': len(uniq),
        'frames_duplicate': F - len(uniq),
        'frames_missing': (len(exp_ids - uniq) if exp_ids else None),
        'frames_unknown_id': (len(uniq - exp_ids) if exp_ids else None),
        'ids_repaired': n_repaired,
    }
    # per-position majority byte (reference) and how many frames agree with it
    ref = np.empty(PDU_LENGTH, dtype=np.uint8)
    ref_count = np.empty(PDU_LENGTH, dtype=np.int64)
    for k in range(PDU_LENGTH):
        vals, cnts = np.unique(P[:, k], return_counts=True)
        j = int(cnts.argmax())
        ref[k] = vals[j]
        ref_count[k] = cnts[j]
    # constant-in-ground-truth positions: one value dominates. Varying id digits
    # (≈uniform over 10 values) fall below the threshold and are excluded.
    stable = ref_count >= max(2, int(stable_frac * F))
    stable_idx = np.where(stable)[0]

    diff = np.bitwise_xor(P[:, stable_idx], ref[stable_idx])   # (F, n_stable)
    biterr_per_frame = _POPCOUNT[diff].sum(axis=1)
    byteerr_per_frame = (diff != 0).sum(axis=1)

    total_bits = int(stable_idx.size * 8 * F)
    total_biterr = int(biterr_per_frame.sum())
    n_clean = int((biterr_per_frame == 0).sum())

    out = {
        'frames_dumped': len(frames),
        'frames_ours_aligned': n_aligned,
        # TRUE metrics — ground truth reconstructed from the frame id
        'frames_error_free': t_clean,
        'known_bits_compared': t_total_bits,
        'bit_errors': t_total_biterr,
        'ber': (t_total_biterr / t_total_bits) if t_total_bits else float('nan'),
        'fer': (1 - t_clean / n_aligned) if n_aligned else float('nan'),
        'worst_frame_bit_errors': int(t_biterr_per_frame.max()),
        'mean_frame_bit_errors': float(t_biterr_per_frame.mean()),
        # legacy majority-vote metrics, kept so old results stay comparable
        'legacy_bits_compared': total_bits,
        'legacy_bit_errors': total_biterr,
        'legacy_ber': (total_biterr / total_bits) if total_bits else float('nan'),
        'legacy_frames_error_free': n_clean,
        'legacy_stable_positions': int(stable_idx.size),
        'legacy_excluded_positions': int(PDU_LENGTH - stable_idx.size),
        'legacy_reference_fill_byte': int(np.bincount(ref[stable_idx]).argmax()),
        'legacy_worst_frame_byte_errors': int(byteerr_per_frame.max()),
    }
    out.update(acct)
    return out


def main():
    p = argparse.ArgumentParser(description="True BER of an OTA WiFi capture")
    p.add_argument('--file', required=True, help='complex64 capture')
    p.add_argument('--tag', default='capture')
    p.add_argument('--json', default=None, help='write metrics JSON here')
    p.add_argument('--save-dump', default=None, help='keep the raw decode dump')
    p.add_argument('--chan-est', type=int, default=CHAN_EST,
                   help='channel estimator: 0=LS 1=LMS 2=COMB 3=STA (default %(default)s)')
    p.add_argument('--decode-passes', type=int, default=1,
                   help='decode the file N times and UNION the recovered frame ids. The '
                        'gr-ieee802-11 chain is non-deterministic (GNU Radio buffer '
                        'boundaries move between runs), so one pass loses ~20%% of frames '
                        'even on a clean file. 3 recovered 136/136 in testing. Costs decode '
                        'CPU only, no extra transmit. (default %(default)s)')
    p.add_argument('--ids-json', default=None,
                   help="<ch0>.frames.json sidecar. Its 'kept' list is the set of frame "
                        "ids actually in the replayed waveform, which makes missing/duplicate "
                        "counts meaningful and lets a corrupted id be repaired to the nearest "
                        "real one. Without it, ids are trusted as decoded.")
    a = p.parse_args()

    known_ids = None
    if a.ids_json:
        with open(a.ids_json) as f:
            known_ids = json.load(f)['kept']
        print(f"  ground-truth ids: {len(known_ids)} from {os.path.basename(a.ids_json)}")

    nsamp = os.path.getsize(a.file) // 8
    print(f"[{a.tag}] {a.file}")
    print(f"  {nsamp} samples ({nsamp/SAMP_RATE*1e3:.1f} ms @ {SAMP_RATE/1e6:.1f} MHz)")
    print("  running gr-ieee802-11 decode (debug dump) ... this streams the whole file")

    def decode_once(dump_path):
        # decode_mac's print_output goes to C-level std::cout -> redirect fd 1 to a file
        sys.stdout.flush()
        saved = os.dup(1)
        with open(dump_path, 'w') as f:
            os.dup2(f.fileno(), 1)
            try:
                tb = ber_rx(a.file, chan_est=a.chan_est)
                tb.run()                # file_source(repeat=False) -> ends at EOF
            finally:
                sys.stdout.flush()
                os.dup2(saved, 1)
                os.close(saved)
        with open(dump_path) as f:
            return list(parse_frames(f.read()))

    all_frames, recovery, split = [], [], None
    for p in range(a.decode_passes):
        dp = (a.save_dump + (f'.{p}' if p else '')) if a.save_dump \
            else tempfile.mktemp(suffix=f'_ber_dump{p}.txt')
        got = decode_once(dp)
        if split is None:
            split = classify_dumped(got)      # pass 1 only; later passes re-dump the same frames
        all_frames += got
        recovery.append(len(dedupe_by_id(all_frames, known_ids)))
        if not a.save_dump and os.path.exists(dp):
            os.remove(dp)
        if a.decode_passes > 1:
            print(f"  decode pass {p+1}/{a.decode_passes}: "
                  f"{recovery[-1]} unique frames so far", flush=True)
    frames = dedupe_by_id(all_frames, known_ids)
    res = analyse(frames, known_ids=known_ids)
    res['decode_passes'] = a.decode_passes
    res['recovery_curve'] = recovery
    res['frames_first_pass'] = recovery[0]
    res.update(split)
    res['frames_recovered_extra'] = recovery[-1] - recovery[0]
    dump_path = a.save_dump
    res['tag'] = a.tag
    res['file'] = a.file
    res['n_samples'] = nsamp

    print("\n" + "=" * 60)
    print(f"BER eval — {a.tag}")
    print("=" * 60)
    print(f"  frames decoded (dumped) : {res['frames_dumped']}")
    print(f"  ours, payload-aligned   : {res['frames_ours_aligned']}")
    if not res['frames_ours_aligned']:
        print("  (no frames matched our MAC header — nothing to score)")
        print("=" * 60)
    else:
        if res['decode_passes'] > 1:
            print(f"  decode passes           : {res['decode_passes']}"
                  f"   recovery {' -> '.join(str(x) for x in res['recovery_curve'])}"
                  f"   (+{res['frames_recovered_extra']} beyond pass 1)")
        if res.get('frames_expected'):
            print(f"  frames expected (ch0)   : {res['frames_expected']}")
            print(f"  unique ids received     : {res['frames_unique']}"
                  f"   missing {res['frames_missing']}"
                  f"   duplicate {res['frames_duplicate']}")
            if res['frames_unknown_id']:
                print(f"  ids not in ch0          : {res['frames_unknown_id']}")
        if res['ids_repaired']:
            print(f"  ids repaired to nearest : {res['ids_repaired']}")
        print(f"  decode_mac dumped       : {res['frames_dumped']}"
              f"   ours {res['frames_ours']}"
              f"   (addr {res['frames_by_addr']} + fill-only {res['frames_by_fill']})"
              f"   rejected: short {res['frames_rejected_short']},"
              f" addr-destroyed {res['frames_rejected_addr']}")
        print(f"  error-free frames       : {res['frames_error_free']}"
              f"  ({100*res['frames_error_free']/res['frames_ours_aligned']:.1f}%)")
        print(f"  bits compared (all 500B): {res['known_bits_compared']}")
        print(f"  bit errors              : {res['bit_errors']}")
        print(f"  >>> BER  (true)         : {res['ber']:.3e}")
        print(f"  >>> FER (frame err rate): {res['fer']:.3f}")
        print(f"  worst/mean bit errors   : {res['worst_frame_bit_errors']} / "
              f"{res['mean_frame_bit_errors']:.2f}  per frame")
        print(f"  [legacy mode-vote BER   : {res['legacy_ber']:.3e} over "
              f"{res['legacy_stable_positions']}/500 positions]")
        print("=" * 60)

    if a.json:
        with open(a.json, 'w') as f:
            json.dump(res, f, indent=2)
        print(f"  wrote {a.json}")
    if not a.save_dump and dump_path and os.path.exists(dump_path):
        os.remove(dump_path)


if __name__ == '__main__':
    main()
