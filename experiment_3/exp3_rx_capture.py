#!/usr/bin/env python3
"""exp3_rx_capture.py — headless RX capture, front-end IDENTICAL to wifi_rx.py.

Replicates the enrollment receiver exactly (b200 serial 325936A, 5.29 GHz,
lo_offset 0, POLICY_MANUAL tune, antenna TX/RX, normalized gain 0.6, 5 MHz)
so recaptured attack signals sit in the same domain as the training clean_run.
Streams raw complex64 to --out for --secs, then exits. Score offline.

  python3 exp3_rx_capture.py --out cap.bin --secs 20 [--gain 0.6]
"""
import sys, time, signal, argparse
from gnuradio import gr, blocks, uhd

SERIAL = "325936A"; SAMP_RATE = 5e6; FREQ = 5.29e9; LO_OFFSET = 0.0


class rx_cap(gr.top_block):
    def __init__(self, out, gain):
        gr.top_block.__init__(self, "Exp3 RX capture", catch_exceptions=True)
        self.src = uhd.usrp_source(",".join((f"serial={SERIAL}", "")),
                                   uhd.stream_args(cpu_format="fc32", args='', channels=[0]))
        self.src.set_samp_rate(SAMP_RATE)
        self.src.set_time_now(uhd.time_spec(time.time()), uhd.ALL_MBOARDS)
        self.src.set_center_freq(uhd.tune_request(FREQ, rf_freq=FREQ - LO_OFFSET,
                                 rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 0)
        self.src.set_antenna("TX/RX", 0)
        self.src.set_normalized_gain(gain, 0)
        self.sink = blocks.file_sink(gr.sizeof_gr_complex, out, False)
        self.sink.set_unbuffered(False)
        self.connect((self.src, 0), (self.sink, 0))
        print(f"  RX serial={SERIAL} freq={FREQ/1e9:.3f}GHz gain={gain} -> {out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--secs', type=float, default=20.0)
    p.add_argument('--gain', type=float, default=0.6)
    a = p.parse_args()
    tb = rx_cap(a.out, a.gain)
    tb.start()
    time.sleep(a.secs)
    tb.stop(); tb.wait()
    print("  RX done", flush=True)


if __name__ == '__main__':
    main()
