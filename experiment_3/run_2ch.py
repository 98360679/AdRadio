#!/usr/bin/env python3
"""run_2ch.py — one resident 2-channel OTA session: victim on ch0 replays the dense
ch0, adversary on ch1 radiates the pre-concatenated epsilon x phase sweep.

Fixed power-up phase for the whole sweep (one session, no live file swap). RX runs as a
continuous capture; blocks are recovered by anchoring on the first frame in the capture.
The Qt GUI stays visible during transmit.
"""
import argparse, os, sys, time, subprocess

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # AdvRadio repo root (parent of experiment_3/)
HERE = os.path.join(BASE, 'training', 'flowgraphs')   # GNU Radio flowgraphs (wifi_adversary_tx.py) live here
sys.path.insert(0, HERE)


def patch(py, **kv):
    import re
    s = open(py).read()
    for k, v in kv.items():
        s = re.sub(r'(self\.%s = %s = ).*' % (k, k), r'\g<1>%s' % v, s, count=1)
    open(py, 'w').write(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ch0', required=True)
    ap.add_argument('--ch1', required=True)
    ap.add_argument('--out', required=True, help='RX capture file')
    ap.add_argument('--secs', type=float, required=True, help='ch1 airtime seconds')
    ap.add_argument('--tx-gain', type=float, default=0.493)
    ap.add_argument('--adv-gain', type=float, default=None,
                    help='adversary (ch1) normalized gain. Default None = follow tx_gain, which is '
                         'how the flowgraph ships (adv_gain = tx_gain). Set it to DECOUPLE the two.')
    ap.add_argument('--rx-gain', type=float, default=0.93,
                    help='MUST match the enrollment receiver gain (wifi_rx.py rx_gain=0.93). At 0.85 the silent-adversary control classifies as device_2/device_1, not the victim.')
    ap.add_argument('--lead', type=float, default=6.0, help='RX head start before TX begins')
    a = ap.parse_args()

    tx_py = os.path.join(HERE, 'wifi_adversary_tx.py')
    # epsilon is the LIVE ch1 scaling knob — the sweep is already baked into ch1, so it
    # must be exactly 1.0 or every PSR label is wrong. adv_gain follows tx_gain.
    patch(tx_py, frame_file='"%s"' % a.ch0, pert_file='"%s"' % a.ch1,
          epsilon='1.0', tx_gain='%.4f' % a.tx_gain)
    # adv_gain ships as "= tx_gain"; a literal here decouples the adversary from the victim
    patch(tx_py, adv_gain='%.4f' % a.adv_gain if a.adv_gain is not None else 'tx_gain')
    adv = a.tx_gain if a.adv_gain is None else a.adv_gain
    print(f"[2ch] ch0={a.ch0}\n[2ch] ch1={a.ch1}\n"
          f"[2ch] tx_gain={a.tx_gain} adv_gain={adv} epsilon=1.0", flush=True)

    # RX first, so the TX turn-on edge is inside the capture and can anchor the blocks
    rx_secs = a.secs + a.lead + 8
    rx = subprocess.Popen(['python3', '-u', os.path.join(BASE, 'exp3_rx_capture.py'),
                           '--out', a.out, '--secs', str(rx_secs), '--gain', str(a.rx_gain)],
                          stdout=open(a.out + '.rxlog', 'w'), stderr=subprocess.STDOUT)
    t0 = time.time()
    while not (os.path.exists(a.out) and os.path.getsize(a.out) > 1e6):
        if rx.poll() is not None:
            print("RX died:\n" + open(a.out + '.rxlog').read()); return 1
        time.sleep(0.1)
    print(f"[2ch] RX capturing -> {a.out}", flush=True)
    time.sleep(a.lead)

    # GUI VISIBLE — the user watches the spectrum during transmit
    from PyQt5 import Qt
    import importlib
    TXM = importlib.import_module('wifi_adversary_tx')
    qapp = Qt.QApplication(sys.argv)
    tb = TXM.wifi_adversary_tx()
    tb.start(); tb.show()
    print(f"[2ch] TX streaming ({a.secs:.0f}s sweep), GUI up", flush=True)

    t_tx = time.time()
    while time.time() - t_tx < a.secs + 3:
        qapp.processEvents()
        time.sleep(0.05)

    tb.stop(); tb.wait()
    print("[2ch] TX done, waiting for RX to finish", flush=True)
    rx.wait()
    sz = os.path.getsize(a.out) / 1e9
    print(f"[2ch] capture complete: {sz:.2f} GB, {time.time()-t0:.0f}s wall", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
