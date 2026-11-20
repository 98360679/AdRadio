#!/usr/bin/env python3
"""
enroll_calib.py — per-device enrollment capture with TX-gain calibration.

EFFICIENT design (devices take >60s to init): the TX flowgraph is brought up ONCE and
kept resident; tx_gain is changed LIVE (set_normalized_gain) between probes, so we pay the
slow radio init once per device instead of once per gain step. The TX DSP runs in GNU
Radio's own scheduler threads (via start()), concurrent with this control loop; the fast
b205mini RX runs as short subprocess probes.

Loop for ONE device:
  bring TX up (resident) -> wait init -> {set gain live; 5s RX probe; read rx_power
  'signal dB'; secant-adjust} until |signal-target|<=tol -> lock gain -> 30s RX clean_run
  capture (TX still resident at locked gain) -> record gain -> caller prompts to swap.

  python3 enroll_calib.py --device-id 1 --tx-serial 325426C

The capture tree (enrollment|validation) is a parameter (--tree). It sets ENROLL_DIR and
is written into the capture paths inside wifi_tx_updated.py / wifi_rx.py via
patch_capture_path(); the two must agree or every capture is reported as 0 MB / 0 frames
while the data is actually fine. The per-device directory is created here too - the
flowgraphs raise FileNotFoundError INSIDE tb.start() if it is missing, which hangs
silently instead of erroring out.
"""
import os, re, sys, time, signal, argparse, subprocess
CAPTURES = os.environ.get('CAPTURES_ROOT', os.path.expanduser('~/captures'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RXLOG = '/tmp/.calib_rx.log'
CAPTURE_ROOT = CAPTURES
ENROLL_DIR = None                      # set from --tree in main()


def patch_capture_path(py, tree):
    """Rewrite the capture tree (enrollment/validation) in a flowgraph's output paths.

    The generated flowgraphs embed the path in several places (the __init__ block and
    again in the set_device_id / set_run_id setters), so this rewrites all of them.
    Matches either tree so a run can be retargeted from enrollment to validation and
    back. Returns how many it changed, so the caller can fail loudly on zero."""
    s = open(py).read()
    new, n = re.subn(r'(?<=/)(?:enrollment|validation)(?=/device)', tree, s)
    if n:
        open(py, 'w').write(new)
    return n


def kill_stale():
    """free the radios: kill any lingering flowgraph python procs (never a shell)."""
    for pid in subprocess.run(['pgrep', '-f', 'wifi_tx_updated.py|wifi_rx.py'],
                              capture_output=True, text=True).stdout.split():
        try:
            if open('/proc/%s/comm' % pid).read().strip() == 'python3':
                os.kill(int(pid), signal.SIGKILL)
        except Exception:
            pass


def patch_var(py, **kv):
    s = open(py).read()
    for k, v in kv.items():
        s = re.sub(r'(self\.%s = %s = )[0-9.]+' % (k, k), r'\g<1>%s' % v, s)
    open(py, 'w').write(s)


def patch_serial(py, ident):
    kind = 'addr' if re.match(r'^\d+\.\d+\.\d+\.\d+$', ident) else 'serial'
    s = open(py).read()
    assert re.search(r'(?:serial|addr)=[A-Za-z0-9.:_]+', s), "no serial=/addr= in %s" % py
    open(py, 'w').write(re.sub(r'(?:serial|addr)=[A-Za-z0-9.:_]+', '%s=%s' % (kind, ident), s, count=1))
    return '%s=%s' % (kind, ident)


def patch_antenna(py, ant):
    """usrp2 wants J1/J2, b200 wants TX/RX — set the TX antenna to match the device."""
    s = open(py).read()
    s2 = re.sub(r'set_antenna\("[^"]*", 0\)', 'set_antenna("%s", 0)' % ant, s, count=1)
    open(py, 'w').write(s2)
    return ant


def last_signal_db(log):
    m = re.findall(r'signal=\s*(-?\d+\.?\d*)\s*dB', open(log).read()) if os.path.exists(log) else []
    return float(m[-1]) if m else None


def rx_run(rx_py, secs, warm=5.0):
    """run wifi_rx.py for ~secs of measurement (b205mini inits ~2s); return signal dB.
    Also (re)writes clean_run — search probes overwrite; the final call is the keeper."""
    p = subprocess.Popen("QT_QPA_PLATFORM=offscreen timeout %d python3 %s" % (int(warm + secs), rx_py),
                         shell=True, stdout=open(RXLOG, 'w'), stderr=subprocess.STDOUT,
                         preexec_fn=os.setsid)
    p.wait()
    return last_signal_db(RXLOG)


def rx_bad_packet():
    return os.path.exists(RXLOG) and any(('bad packet' in l.lower() or 'vrt' in l.lower())
                                         for l in open(RXLOG))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device-id', type=int, required=True)
    ap.add_argument('--run-id', type=int, default=1)
    ap.add_argument('--run-ids', default=None, help='comma list, e.g. 2,3 — capture several runs in one init')
    ap.add_argument('--tx-serial', required=True, help="serial= (or IP for addr=)")
    ap.add_argument('--tx-antenna', default=None, help="TX antenna: TX/RX (b200) or J1 (usrp2)")
    ap.add_argument('--target-db', type=float, default=-44.5)
    ap.add_argument('--tol', type=float, default=0.5)
    ap.add_argument('--g0', type=float, default=0.6)
    ap.add_argument('--max-gain', type=float, default=1.0, help='hard cap on tx_gain (device limit)')
    ap.add_argument('--fix-gain', type=float, default=None, help='skip search: capture at this exact gain')
    ap.add_argument('--tx-py', default=os.path.join(HERE, 'wifi_tx_updated.py'))
    ap.add_argument('--rx-py', default=os.path.join(HERE, 'wifi_rx.py'))
    ap.add_argument('--init-wait', type=float, default=70.0, help='TX radio init seconds (>60)')
    ap.add_argument('--settle', type=float, default=2.0, help='seconds after a live gain change')
    ap.add_argument('--probe-secs', type=float, default=5.0)
    ap.add_argument('--capture-secs', type=float, default=30.0)
    ap.add_argument('--max-iters', type=int, default=8)
    ap.add_argument('--slope', type=float, default=89.75)
    ap.add_argument('--log-csv', default=os.path.join(HERE, 'enroll_gains.csv'))
    ap.add_argument('--tree', default='enrollment', choices=['enrollment', 'validation'],
                    help='which capture tree to write into. Cross-day validation captures '
                         'belong in validation/, NOT alongside the enrollment a model was '
                         'trained on.')
    a = ap.parse_args()

    global ENROLL_DIR
    ENROLL_DIR = os.path.join(CAPTURE_ROOT, a.tree)
    ntx = patch_capture_path(a.tx_py, a.tree)
    nrx = patch_capture_path(a.rx_py, a.tree)
    if not ntx or not nrx:
        sys.exit(f'[calib] ABORT: no capture path found in '
                 f'{"tx " if not ntx else ""}{"rx " if not nrx else ""}flowgraph - '
                 f'refusing to capture into an unknown directory')
    os.makedirs(f'{ENROLL_DIR}/device_{a.device_id}', exist_ok=True)
    print(f'[calib] {a.tree}  ->  {ENROLL_DIR}/device_{a.device_id}  '
          f'(patched {ntx} tx + {nrx} rx paths)', flush=True)

    kill_stale(); time.sleep(2)
    # stamp identity + initial gain BEFORE importing the TX flowgraph
    patch_var(a.tx_py, device_id=a.device_id, run_id=a.run_id, tx_gain='%.4f' % a.g0)
    patch_var(a.rx_py, device_id=a.device_id, run_id=a.run_id)
    used = patch_serial(a.tx_py, a.tx_serial)
    ant = patch_antenna(a.tx_py, a.tx_antenna) if a.tx_antenna else 'TX/RX'
    print(f"[calib] device_{a.device_id} run_{a.run_id}  TX {used} ant {ant}  target {a.target_db} dB (+-{a.tol})")

    # bring the TX up RESIDENT (DSP in GR threads); QApplication offscreen for the Qt widgets
    from PyQt5 import Qt
    import importlib
    TXM = importlib.import_module('wifi_tx_updated')
    qapp = Qt.QApplication(sys.argv)
    tb = TXM.wifi_tx_updated()
    tb.start()
    print(f"[calib] TX resident; waiting {a.init_wait:.0f}s for radio init (>60s)...", flush=True)
    time.sleep(a.init_wait)

    g, hist, locked = a.g0, [], None
    try:
        if a.fix_gain is not None:                       # no search: capture at the device's cap
            g = a.fix_gain
            tb.set_tx_gain(g)
            time.sleep(a.settle)
            db = rx_run(a.rx_py, a.probe_secs)
            locked = (g, db)
            print(f"[calib] FIXED gain {g:.4f} -> signal={db if db is None else round(db,2)} dB", flush=True)
        else:
          for it in range(a.max_iters):
            tb.set_tx_gain(g)                       # LIVE gain change, no re-init
            time.sleep(a.settle)
            db = rx_run(a.rx_py, a.probe_secs)
            print(f"  iter {it}: tx_gain={g:.4f} -> signal={db if db is None else round(db,2)} dB", flush=True)
            if db is None:
                g = min(1.0, g + 0.1); continue
            hist.append((g, db))
            err = a.target_db - db
            if abs(err) <= a.tol:
                locked = (g, db); break
            if len(hist) >= 2 and hist[-1][0] != hist[-2][0] and hist[-1][1] != hist[-2][1]:
                slope = (hist[-1][1] - hist[-2][1]) / (hist[-1][0] - hist[-2][0])
                slope = slope if abs(slope) > 5 else a.slope
            else:
                slope = a.slope
            g = min(1.0, max(0.0, g + err / slope))

        if locked is None:
            print(f"[calib] did NOT converge (last g={g:.4f}); not capturing."); return
        g, db = locked
        run_ids = [int(x) for x in a.run_ids.split(',')] if a.run_ids else [a.run_id]
        print(f"[calib] LOCKED device_{a.device_id}: tx_gain={g:.4f}  signal={db if db is None else round(db,2)} dB — "
              f"capturing runs {run_ids} @ {a.capture_secs:.0f}s each (one init)", flush=True)
        tb.set_tx_gain(g)
        header = not os.path.exists(a.log_csv)
        for rid in run_ids:                          # TX stays resident; only the RX re-runs per run
            patch_var(a.rx_py, run_id=rid)
            cap_db = rx_run(a.rx_py, a.capture_secs)
            cr = f"{ENROLL_DIR}/device_{a.device_id}/clean_run_{rid}.bin"
            sz = os.path.getsize(cr) / 1e6 if os.path.exists(cr) else 0
            ber = f"{ENROLL_DIR}/device_{a.device_id}/ber_frames_run_{rid}.jsonl"
            nfr = sum(1 for _ in open(ber)) if os.path.exists(ber) else 0
            print(f"RESULT device_{a.device_id} run_{rid}: tx_gain={g:.4f}  "
                  f"signal={cap_db if cap_db is None else round(cap_db, 2)} dB  clean_run={sz:.0f} MB  "
                  f"frames={nfr}  bad_packet={'YES' if rx_bad_packet() else 'no'}", flush=True)
            with open(a.log_csv, 'a') as f:
                if header:
                    f.write("device_id,run_id,tx_gain,signal_db,clean_run_MB,frames\n"); header = False
                f.write(f"{a.device_id},{rid},{g:.4f},{cap_db if cap_db is not None else db:.2f},{sz:.0f},{nfr}\n")
    finally:
        tb.stop(); tb.wait()


if __name__ == '__main__':
    main()
