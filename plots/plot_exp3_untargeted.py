#!/usr/bin/env python3
"""Experiment 3, untargeted attack.
Fooling rate and BER vs perturbation budget delta (0.01 - 0.1).
Six methods: FGSM, BIM, MIM, PGD (alpha = 0.01, 0.05, 0.1).

A secondary top axis shows PSR (dB), which is a deterministic function of delta:
PSR = 20*log10(delta) + 3.01 dB  (matches the sheet's PSR column exactly).

BER note: cells with ZERO observed bit errors are 95% confidence upper bounds,
-ln(0.05)/N (7.28e-06 for 103 decoded frames). Those points are drawn hollow;
filled markers are measured BER.
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
        "fr": [0.00, 0.00, 0.00, 0.00, 0.97, 1.94, 7.77, 33.01,
               84.47, 97.09, 99.03, 100.00, 100.00, 100.00, 100.00, 100.00],
        "ber": [F, F, F, F, F, F, F, F, F,
                8.01e-05, 5.85e-04, 2.29e-03, 8.00e-03, 5.52e-02, 3.00e-01, 4.24e-01],
        "err": [0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 241, 945, 3233, 20082, 91146, 115297],
        "rej": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 12, 27, 35],
    },
    "BIM": {
        "fr": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.97, 5.83,
               27.18, 51.46, 70.87, 85.44, 88.35, 95.15, 100.00, 100.00],
        "ber": [F, F, F, F, F, F, F, F,
                1.46e-05, 4.37e-05, 2.43e-04, 7.69e-04, 3.25e-03, 2.68e-02,
                1.93e-01, 3.70e-01],
        "err": [0, 0, 0, 0, 0, 0, 0, 0, 6, 18, 100, 317, 1325, 10400, 55581, 99219],
        "rej": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 6, 31, 36],
    },
    "MIM": {
        "fr": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 2.91, 15.53,
               48.54, 73.79, 91.26, 97.09, 97.09, 100.00, 100.00, 100.00],
        "ber": [F, F, F, F, F, F, F, F, F,
                1.21e-05, 3.03e-04, 1.69e-03, 4.83e-03, 3.39e-02, 2.44e-01, 3.98e-01],
        "err": [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 125, 697, 1952, 12887, 74236, 98821],
        "rej": [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 8, 27, 41],
    },
    r"PGD ($\alpha$=0.01)": {
        "fr": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.97, 7.77,
               35.92, 61.17, 73.79, 83.50, 92.23, 98.06, 100.00, 100.00],
        "ber": [F, F, F, F, F, F, F, F,
                1.46e-05, 3.64e-05, 4.81e-04, 1.40e-03, 5.23e-03, 2.75e-02,
                1.41e-01, 2.85e-01],
        "err": [0, 0, 0, 0, 0, 0, 0, 0, 6, 15, 198, 570, 2135, 10111, 47791, 82164],
        "rej": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 11, 18, 31],
    },
    r"PGD ($\alpha$=0.05)": {
        "fr": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.94, 9.71,
               45.63, 77.67, 92.23, 98.06, 100.00, 100.00, 100.00, 100.00],
        "ber": [F, F, F, F, F, F, F, F, F,
                6.31e-05, 3.67e-04, 2.25e-03, 8.35e-03, 5.59e-02, 3.03e-01, 4.28e-01],
        "err": [0, 0, 0, 0, 0, 0, 0, 0, 0, 26, 151, 917, 3305, 20330, 81227, 116299],
        "rej": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 12, 36, 35],
    },
    r"PGD ($\alpha$=0.1)": {
        "fr": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.97, 10.68,
               49.51, 80.58, 95.15, 100.00, 100.00, 100.00, 100.00, 100.00],
        "ber": [F, F, F, F, F, F, 1.47e-05, F, F,
                9.80e-05, 5.07e-04, 3.40e-03, 9.29e-03, 5.64e-02, 3.04e-01, 4.25e-01],
        "err": [0, 0, 0, 0, 0, 0, 6, 0, 0, 40, 207, 1399, 3717, 21666, 86436, 108862],
        "rej": [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 3, 7, 32, 39],
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
TITLE = "Experiment 3: Untargeted Attack"

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


# ---------- Fooling rate ----------
fig, ax = plt.subplots(figsize=(7.5, 5.2))
for name, d in DATA.items():
    ax.plot(DELTA, d["fr"], label=name, markersize=6, linewidth=1.6, **STYLE[name])
ax.set_xscale("log")
ax.set_xlabel(r"Perturbation budget $\epsilon$")
ax.set_ylabel("Fooling rate (%)")
ax.set_xticks(XTICKS)
ax.set_xticklabels([str(t) for t in XTICKS])
ax.set_xlim(*XLIM)
ax.set_ylim(-3, 105)
ax.grid(True, which="both", ls="--", alpha=0.4)
ax.legend(loc="upper left", framealpha=0.95)
add_psr_axis(ax)
ax.set_title(TITLE, pad=38)
save(fig, "exp3_untargeted_fooling_rate")
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
ax.set_ylim(4e-6, 1.6)
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
save(fig, "exp3_untargeted_ber")
plt.close(fig)

print("done")
