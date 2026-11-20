#!/usr/bin/env python3
"""Experiment 3, targeted attack (target = Device 1).
Target success rate and BER vs perturbation budget delta (0.01 - 0.1).
Six methods: FGSM, BIM, MIM, PGD (alpha = 0.01, 0.05, 0.1).

Secondary top axis: PSR (dB) = 20*log10(delta) + 3.01, matching the sheet.

BER note: cells with ZERO observed bit errors are 95% confidence upper bounds,
-ln(0.05)/N (7.28e-06 for 103 decoded frames). Drawn hollow; filled markers are
measured BER.
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
OUT = os.environ.get("FIG_OUT", os.path.join(HERE, "output"))
if "__file__" in globals():
    os.makedirs(OUT, exist_ok=True)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DELTA = np.array([0.01, 0.012, 0.014, 0.016, 0.018, 0.02, 0.022, 0.025,
                  0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.08, 0.1])
F = 7.28e-06

DATA = {
    "FGSM": {
        "fr": [0.00, 0.00, 0.00, 11.65, 71.84, 99.03, 100.00, 100.00,
               100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00],
        "ber": [F, F, F, F, F, 1.70e-05, 3.22e-05, F, F,
                1.60e-04, 2.16e-04, 1.81e-03, 6.40e-03, 5.23e-02, 2.93e-01, 4.20e-01],
        "err": [0, 0, 0, 0, 0, 7, 13, 0, 0, 66, 89, 747, 2612, 19683, 80743, 109109],
        "rej": [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 9, 34, 38],
    },
    "BIM": {
        "fr": [0.00, 0.00, 0.00, 0.97, 27.18, 81.55, 100.00, 100.00,
               100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00],
        "ber": [F, F, F, F, F, 3.64e-05, F, 1.10e-03, F,
                5.58e-05, 1.48e-04, 1.03e-03, 4.31e-03, 2.89e-02, 2.14e-01, 3.83e-01],
        "err": [0, 0, 0, 0, 0, 15, 0, 443, 0, 23, 61, 425, 1777, 11107, 65860, 101072],
        "rej": [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 7, 26, 37],
    },
    "MIM": {
        "fr": [0.00, 0.00, 0.00, 2.91, 44.66, 98.06, 100.00, 100.00,
               100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00],
        "ber": [F, F, F, F, F, F, 2.48e-06, 1.24e-05, F,
                2.91e-05, 2.21e-04, 1.24e-03, 4.16e-03, 3.60e-02, 2.51e-01, 4.03e-01],
        "err": [0, 0, 0, 0, 0, 0, 1, 5, 0, 12, 91, 511, 1680, 12947, 73408, 104904],
        "rej": [0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 13, 30, 38],
    },
    r"PGD ($\alpha$=0.01)": {
        "fr": [0.00, 0.00, 0.00, 0.00, 11.65, 56.31, 95.15, 100.00,
               100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00],
        "ber": [F, F, F, F, F, F, 2.43e-05, F, F,
                1.02e-04, 2.77e-04, 1.42e-03, 5.29e-03, 2.85e-02, 1.48e-01, 2.98e-01],
        "err": [0, 0, 0, 0, 0, 0, 10, 0, 0, 42, 114, 583, 2136, 10948, 50847, 78669],
        "rej": [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 7, 17, 37],
    },
    r"PGD ($\alpha$=0.05)": {
        "fr": [0.00, 0.00, 0.00, 0.00, 11.65, 57.28, 94.17, 100.00,
               100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00],
        "ber": [F, F, F, F, F, F, F, F,
                2.18e-05, 2.18e-05, 3.50e-04, 1.53e-03, 8.48e-03, 5.40e-02,
                2.93e-01, 4.16e-01],
        "err": [0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 144, 629, 3427, 20738, 83337, 109828],
        "rej": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 7, 32, 37],
    },
    r"PGD ($\alpha$=0.1)": {
        "fr": [0.00, 0.00, 0.00, 0.00, 10.68, 50.49, 90.29, 100.00,
               100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00],
        "ber": [F, F, F, F, F, F, F, F, F,
                1.26e-04, 5.95e-04, 2.41e-03, 8.98e-03, 4.23e-02, 2.23e-01, 3.86e-01],
        "err": [0, 0, 0, 0, 0, 0, 0, 0, 0, 52, 245, 993, 3699, 15736, 64100, 98862],
        "rej": [0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 10, 31, 39],
    },
}

STYLE = {
    "FGSM":                 dict(marker="o", color="#1f77b4", ls="-"),
    "BIM":                  dict(marker="s", color="#d62728", ls="-"),
    "MIM":                  dict(marker="^", color="#2ca02c", ls="-"),
    r"PGD ($\alpha$=0.01)": dict(marker="D", color="#9467bd", ls="--"),
    r"PGD ($\alpha$=0.05)": dict(marker="v", color="#ff7f0e", ls="--"),
    r"PGD ($\alpha$=0.1)":  dict(marker="P", color="#8c564b", ls="--"),
}

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

XTICKS = [0.01, 0.02, 0.03, 0.05, 0.1]
XLIM = (0.0092, 0.109)
TITLE = "Experiment 3: Targeted Attack (target = Device 1)"

for k, v in DATA.items():
    for f in ("fr", "ber", "err", "rej"):
        assert len(v[f]) == len(DELTA), (k, f, len(v[f]))


def psr_db(d):
    return 20 * np.log10(d) + 3.0103


def add_psr_axis(ax):
    ax2 = ax.twiny()
    ax2.set_xscale("log")
    ax2.set_xlim(*XLIM)
    ax2.set_xticks(XTICKS)
    ax2.set_xticklabels([f"{psr_db(t):.1f}" for t in XTICKS])
    ax2.set_xlabel("PSR (dB)", labelpad=8)
    ax2.tick_params(labelsize=10)
    return ax2


def save(fig, stem):
    fig.savefig(f"{OUT}/{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT}/{stem}.pdf", bbox_inches="tight")


# ---------- Target success rate ----------
fig, ax = plt.subplots(figsize=(7.5, 5.2))
for name, d in DATA.items():
    ax.plot(DELTA, d["fr"], label=name, markersize=6, linewidth=1.6, **STYLE[name])
ax.set_xscale("log")
ax.set_xlabel(r"Perturbation budget $\epsilon$")
ax.set_ylabel("Target success rate (%)")
ax.set_xticks(XTICKS)
ax.set_xticklabels([str(t) for t in XTICKS])
ax.set_xlim(*XLIM)
ax.set_ylim(-3, 105)
ax.grid(True, which="both", ls="--", alpha=0.4)
ax.legend(loc="lower right", framealpha=0.95)
add_psr_axis(ax)
ax.set_title(TITLE, pad=38)
save(fig, "exp3_targeted_target_rate")
plt.close(fig)

# ---------- BER ----------
fig, ax = plt.subplots(figsize=(7.5, 5.2))
for name, d in DATA.items():
    st = STYLE[name]
    ber = np.array(d["ber"])
    err = np.array(d["err"])
    ax.plot(DELTA, ber, color=st["color"], ls=st["ls"], linewidth=1.6, zorder=2)
    m = err > 0
    ax.plot(DELTA[m], ber[m], linestyle="none", marker=st["marker"],
            color=st["color"], markersize=6, zorder=3)
    ax.plot(DELTA[~m], ber[~m], linestyle="none", marker=st["marker"],
            markerfacecolor="none", markeredgecolor=st["color"], markersize=5,
            markeredgewidth=1.1, zorder=3)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Perturbation budget $\epsilon$")
ax.set_ylabel("Bit error rate (BER)")
ax.set_xticks(XTICKS)
ax.set_xticklabels([str(t) for t in XTICKS])
ax.set_xlim(*XLIM)
ax.set_ylim(1.2e-6, 1.6)
ax.grid(True, which="both", ls="--", alpha=0.4)

handles = [Line2D([], [], color=STYLE[n]["color"], ls=STYLE[n]["ls"],
                  marker=STYLE[n]["marker"], markersize=6, label=n) for n in DATA]
handles += [
    Line2D([], [], color="0.35", ls="none", marker="o", markersize=6,
           label="measured (bit errors > 0)"),
    Line2D([], [], color="0.35", ls="none", marker="o", markerfacecolor="none",
           markersize=6, label="95% CI upper bound (0 errors)"),
]
ax.legend(handles=handles, loc="upper left", framealpha=0.95, ncol=2, fontsize=9)
add_psr_axis(ax)
ax.set_title(TITLE, pad=38)
save(fig, "exp3_targeted_ber")
plt.close(fig)

print("done")
