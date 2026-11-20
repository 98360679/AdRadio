#!/usr/bin/env python3
"""Experiment 1 (digital reachability), single-column IEEE figures.

Sized for one column of a two-column IEEE paper (3.5 in wide), base font 10 pt.
Secondary top axis shows PSR = 10*log10(2*eps^2) dB, which reproduces the sheet
PSR column exactly.
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

EPS = np.array([0.01, 0.012, 0.014, 0.016, 0.018, 0.02, 0.022, 0.025,
                0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.08, 0.1])

UNTARG_FR = {
 "FGSM": [2.00,5.33,10.67,19.00,33.33,50.33,61.00,81.67,95.67,99.67,100,100,100,100,100,100],
 "BIM":  [2.33,6.00,14.67,27.00,45.33,62.00,77.33,91.67,99.67,99.67,100,100,100,100,100,100],
 "MIM":  [2.33,5.67,12.00,24.00,40.67,56.00,69.67,88.67,98.33,100,100,100,100,100,100,100],
 r"PGD ($\alpha$=0.01)": [2.67,7.33,21.00,36.00,59.33,78.00,90.00,98.00,100,100,100,100,100,100,100,100],
 r"PGD ($\alpha$=0.05)": [2.67,7.67,21.00,37.00,58.67,78.33,90.67,98.00,100,100,100,100,100,100,100,100],
 r"PGD ($\alpha$=0.1)":  [2.67,7.33,21.00,37.33,58.67,79.00,90.00,98.00,100,100,100,100,100,100,100,100],
}
UNTARG_BER = {
 "FGSM": [2.36e-3,2.93e-3,3.40e-3,3.91e-3,4.42e-3,4.93e-3,5.42e-3,6.15e-3,
          7.41e-3,8.62e-3,9.92e-3,1.12e-2,1.25e-2,1.51e-2,2.03e-2,2.55e-2],
 "BIM":  [2.18e-3,2.64e-3,3.02e-3,3.40e-3,3.72e-3,4.21e-3,4.52e-3,5.19e-3,
          6.15e-3,7.22e-3,8.02e-3,8.98e-3,9.70e-3,1.14e-2,1.51e-2,1.88e-2],
 "MIM":  [2.25e-3,2.72e-3,3.17e-3,3.59e-3,4.07e-3,4.53e-3,4.98e-3,5.73e-3,
          6.80e-3,7.84e-3,9.05e-3,1.00e-2,1.11e-2,1.31e-2,1.73e-2,2.14e-2],
 r"PGD ($\alpha$=0.01)": [2.20e-3,2.68e-3,3.11e-3,3.65e-3,4.17e-3,4.56e-3,5.10e-3,5.79e-3,
          6.95e-3,7.63e-3,8.55e-3,9.37e-3,1.05e-2,1.22e-2,1.55e-2,1.97e-2],
 r"PGD ($\alpha$=0.05)": [2.23e-3,2.70e-3,3.13e-3,3.78e-3,4.22e-3,4.70e-3,5.16e-3,5.84e-3,
          6.90e-3,7.90e-3,8.76e-3,9.90e-3,1.12e-2,1.38e-2,1.78e-2,2.18e-2],
 r"PGD ($\alpha$=0.1)":  [2.22e-3,2.71e-3,3.19e-3,3.71e-3,4.19e-3,4.69e-3,5.22e-3,5.79e-3,
          6.92e-3,8.11e-3,9.14e-3,1.05e-2,1.19e-2,1.35e-2,1.77e-2,2.22e-2],
}
TARG_TR = {
 "FGSM": [0.67,2.33,8.33,18.67,34.33,52.00,67.33,88.67,99.00,100,100,100,100,100,100,100],
 "BIM":  [0.67,2.33,9.00,20.67,39.00,56.00,72.67,94.67,99.33,100,100,100,100,100,100,100],
 "MIM":  [0.67,2.33,8.33,19.67,36.67,55.00,71.00,94.00,99.33,100,100,100,100,100,100,100],
 r"PGD ($\alpha$=0.01)": [0.67,2.33,10.00,23.00,41.67,60.67,77.33,95.33,99.67,100,100,100,100,100,100,100],
 r"PGD ($\alpha$=0.05)": [0.67,2.33,10.00,23.00,41.33,59.67,77.33,95.33,99.67,100,100,100,100,100,100,100],
 r"PGD ($\alpha$=0.1)":  [0.67,2.33,10.00,23.00,41.33,59.67,77.33,95.33,99.33,98.67,87.33,72.33,63.33,97.67,100,100],
}
TARG_BER = {
 "FGSM": [2.43e-3,2.92e-3,3.48e-3,4.05e-3,4.56e-3,5.00e-3,5.52e-3,6.28e-3,
          7.50e-3,8.70e-3,9.92e-3,1.12e-2,1.24e-2,1.50e-2,2.05e-2,2.59e-2],
 "BIM":  [2.26e-3,2.68e-3,3.08e-3,3.57e-3,4.07e-3,4.40e-3,4.83e-3,5.57e-3,
          6.59e-3,7.60e-3,8.46e-3,9.43e-3,1.02e-2,1.17e-2,1.50e-2,1.86e-2],
 "MIM":  [2.30e-3,2.74e-3,3.27e-3,3.73e-3,4.20e-3,4.73e-3,5.17e-3,5.88e-3,
          6.95e-3,8.25e-3,9.31e-3,1.03e-2,1.13e-2,1.36e-2,1.76e-2,2.16e-2],
 r"PGD ($\alpha$=0.01)": [2.32e-3,2.78e-3,3.41e-3,3.95e-3,4.44e-3,4.93e-3,5.27e-3,6.03e-3,
          7.00e-3,8.11e-3,8.97e-3,9.74e-3,1.08e-2,1.23e-2,1.55e-2,1.93e-2],
 r"PGD ($\alpha$=0.05)": [2.37e-3,2.94e-3,3.38e-3,3.95e-3,4.54e-3,4.94e-3,5.38e-3,6.03e-3,
          7.12e-3,7.85e-3,8.68e-3,9.25e-3,1.01e-2,1.19e-2,1.59e-2,2.07e-2],
 r"PGD ($\alpha$=0.1)":  [2.45e-3,2.93e-3,3.45e-3,3.98e-3,4.48e-3,4.87e-3,5.36e-3,6.25e-3,
          7.28e-3,8.30e-3,9.33e-3,1.05e-2,1.20e-2,1.28e-2,1.59e-2,2.00e-2],
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
    "font.size": 10, "axes.labelsize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 7.2, "legend.handlelength": 1.8,
    "legend.columnspacing": 0.8, "legend.labelspacing": 0.25,
    "legend.handletextpad": 0.45, "legend.borderpad": 0.35,
})

XTICKS = [0.01, 0.02, 0.05, 0.1]
XLIM = (0.0093, 0.108)
FIGSIZE = (3.5, 3.15)          # one IEEE column


def psr(e):
    return 10 * np.log10(2 * np.asarray(e) ** 2)


def add_psr(ax):
    ax2 = ax.twiny()
    ax2.set_xscale("log")
    ax2.set_xlim(*XLIM)
    ax2.set_xticks(XTICKS)
    ax2.set_xticklabels([f"{psr(v):.0f}" for v in XTICKS])
    ax2.set_xlabel("PSR (dB)", labelpad=4)
    return ax2


def save(fig, stem):
    fig.savefig(f"{OUT}/{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(f"{OUT}/{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def rate_fig(data, ylabel, stem, legend_loc="lower right"):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for name, y in data.items():
        st = STYLE[name]
        ax.plot(EPS, y, label=name, markersize=3.4, linewidth=1.15,
                markeredgewidth=0.6, **st)
    ax.set_xscale("log")
    ax.set_xlim(*XLIM); ax.set_ylim(-4, 106)
    ax.set_xticks(XTICKS); ax.set_xticklabels([str(v) for v in XTICKS])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel(r"Perturbation budget $\epsilon$")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", ls="--", alpha=0.35, linewidth=0.5)
    ax.legend(loc=legend_loc, ncol=2, framealpha=0.95)
    add_psr(ax)
    save(fig, stem)


def ber_fig(data, stem):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for name, y in data.items():
        st = STYLE[name]
        ax.plot(EPS, y, label=name, markersize=3.4, linewidth=1.15,
                markeredgewidth=0.6, **st)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*XLIM); ax.set_ylim(1.9e-3, 3.4e-2)
    ax.set_xticks(XTICKS); ax.set_xticklabels([str(v) for v in XTICKS])
    ax.set_yticks([2e-3, 5e-3, 1e-2, 2e-2])
    ax.set_yticklabels([r"$2{\times}10^{-3}$", r"$5{\times}10^{-3}$",
                        r"$10^{-2}$", r"$2{\times}10^{-2}$"])
    ax.set_xlabel(r"Perturbation budget $\epsilon$")
    ax.set_ylabel("Bit error rate (BER)")
    ax.grid(True, which="both", ls="--", alpha=0.35, linewidth=0.5)
    ax.legend(loc="upper left", ncol=2, framealpha=0.95)
    add_psr(ax)
    save(fig, stem)


rate_fig(UNTARG_FR, "Fooling rate (%)", "exp1_untargeted_fooling_rate")
ber_fig(UNTARG_BER, "exp1_untargeted_ber")
rate_fig(TARG_TR, "Target rate (%)", "exp1_targeted_target_rate")
ber_fig(TARG_BER, "exp1_targeted_ber")

print("PSR check:", np.round(psr(EPS), 2)[:4], "... expect -36.99 -35.41 -34.07 -32.91")
print("done")
