#!/usr/bin/env python3
"""Experiment 2 (single-channel OTA), single-column side-by-side figures.

(a) untargeted, (b) targeted, for both the attack rate and the BER.
PSR is taken from the measured column in the sheet, which sits a constant
0.26 dB below the nominal 10*log10(2*eps^2) used in Experiments 1 and 3.

BER note: cells with ZERO observed bit errors are rule-of-three 95% upper
bounds 3/N (7.28e-06 at N = 103*4000 = 412000 bits). Those points are drawn
hollow; filled markers are measured.
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

EPS = np.round(np.arange(0.10, 0.505, 0.01), 2)
F, F2, F3 = 7.28e-06, 7.35e-06, 7.43e-06

# measured PSR column (dB), one per eps
PSR = [-17.25,-16.42,-15.67,-14.97,-14.33,-13.73,-13.17,-12.64,-12.14,-11.68,
       -11.23,-10.81,-10.40,-10.02,-9.65,-9.29,-8.95,-8.62,-8.31,-8.00,
       -7.71,-7.42,-7.15,-6.88,-6.62,-6.37,-6.12,-5.89,-5.65,-5.43,
       -5.21,-4.99,-4.79,-4.58,-4.38,-4.19,-4.00,-3.81,-3.63,-3.45,-3.27]

# ----------------------------- untargeted (current sheet) -------------------
U_FR = {
 "FGSM": [0]*21 + [1.00,4.80,5.80,5.80,8.70,9.70,6.80,4.80,4.80,8.70,11.70,
                   11.70,17.60,11.80,11.80,19.40,19.40,15.50,18.40,24.30],
 "BIM":  [0]*21 + [2.94,23.30,37.86,63.11,76.70,69.90,77.67,84.47,77.67,88.35,
                   88.35,87.38,99.02,100,100,100,100,100,100,100],
 "MIM":  [0]*16 + [4.85,0,6.80,19.42,39.81,51.46,85.29,93.20,94.17,100,99.03,
                   99.03,100,100,100,100,99.03,100,100,100,100,100,100,100,100],
 r"PGD ($\alpha$=0.01)": [0]*20 + [0.97,0.97,0,0,0.97,0,0,0,1.94,1.94,5.83,12.75,
                   13.59,22.55,28.43,29.41,36.89,48.54,56.31,61.17,74.76],
 r"PGD ($\alpha$=0.05)": [0]*10 + [0.97,14.71,21.57,41.18,14.56,52.43,61.17,72.82,
                   76.70,91.26,93.20,91.26,100,100,99.03,100,99.03,100,100,100,
                   100,100,100,100,100,100,100,100,100,100,100],
 r"PGD ($\alpha$=0.1)":  [0]*12 + [6.80,13.59,18.45,21.36,40.78,66.02,77.67,89.32,
                   95.15,93.20,100,100,100,100,100,100,100,100,100,100,100,100,
                   100,100,100,100,100,100,100],
}
U_ERR = {
 "FGSM": [0]*26 + [11,0,0,11,21,16,25,25,48,70,75,78,75,150,287],
 "BIM":  [0]*41,
 "MIM":  [0]*30 + [8,4,0,0,3,8,7,0,26,0,11],
 r"PGD ($\alpha$=0.01)": [0]*41,
 r"PGD ($\alpha$=0.05)": [0]*41,
 r"PGD ($\alpha$=0.1)":  [0]*41,
}
U_BER = {
 "FGSM": [F]*19 + [F2,F3] + [F]*5 + [2.67e-05,F3,F,2.723e-05,5.097e-05,3.96e-05,
          6.13e-05,6.07e-05,1.17e-04,1.72e-04,1.82e-04,1.89e-04,1.82e-04,
          3.64e-04,6.97e-04],
 "BIM":  [F,F3,F,F,F,F,F,F,F3,F2,F2,F,F3,F,F,F,F,F,F,F,
          F2,F3,F,F3,F,F,F,F,F,F,F,F2,F2,F,F,F3,F,F,F,F3,F],
 "MIM":  [F,F,F3,F,F,F,F,F,F2,F2,F,F,F,F,F,F,F,F3,F,F,
          F,F,F,F,F,F,F,F,F,F,1.98e-05,9.71e-06,F,F,F3,1.94e-05,1.73e-05,F,
          6.44e-05,F,2.70e-05],
 r"PGD ($\alpha$=0.01)": [F]*7 + [F3,F,F3] + [F]*10 + [F3,F,F,F,F,F,F,F,F,F3,
          F2,F3,F,F,F,F3,F,F,F,F,F],
 r"PGD ($\alpha$=0.05)": [F,F,F,F3,F,F,F,F,F,F,F3] + [F]*30,
 r"PGD ($\alpha$=0.1)":  [F]*8 + [F3,F,F2] + [F]*30,
}

# ----------------------------- targeted (previous sheet) --------------------
T_TR = {
 "FGSM": [0]*14 + [1.94,4.85,15.53,26.21,28.16,36.89,46.60,49.51,84.47,90.29,
                   92.23,96.12,98.06,99.03,97.09,99.03,99.03,99.03,100,100,100,
                   100,100,100,100,100,100],
 "BIM":  [0]*11 + [2.91,1.94,10.68,25.24,52.43,77.67,93.20,95.15,100,99.03,
                   92.23] + [100]*19,
 "MIM":  [0]*12 + [2.88,2.91,10.68,22.33,33.98,52.43,68.93,86.41,93.20,91.26,
                   97.09,99.03,99.03] + [100]*16,
 r"PGD ($\alpha$=0.01)": [0,0,0,3.88,0,0,0,0,0,2.91,0,0,0,0,0,0,0,0,0,0,0,0,3.92,
                   0.97,0.97,8.74,4.85,14.56,24.27,18.45,28.16,24.27,21.36,30.10,
                   37.86,35.92,52.43,64.08,66.99,72.82,77.67],
 r"PGD ($\alpha$=0.05)": [0]*8 + [1.94,10.68,32.04,80.58,89.32,93.20,100,98.06] + [100]*25,
 r"PGD ($\alpha$=0.1)":  [0,0,0,18.45,0,0.97,4.85,16.50,41.75,68.93,82.52,89.32,
                   88.35,95.15,99.03,99.03,100,100,100,100,100,98.06] + [100]*19,
}
T_ERR = {
 "FGSM": [0]*26 + [6,0,7,4,10,6,0,65,63,64,50,117,161,154,202],
 "BIM":  [0]*35 + [3,0,2,16,10,12],
 "MIM":  [0]*35 + [3,13,4,5,16,40],
 r"PGD ($\alpha$=0.01)": [0]*6 + [275] + [0]*34,
 r"PGD ($\alpha$=0.05)": [0]*38 + [6,0,15],
 r"PGD ($\alpha$=0.1)":  [0]*32 + [2,7,11,8,9,21,21,16,57],
}
T_BER = {
 "FGSM": [F]*8 + [F3,F,F3] + [F]*10 + [F3,F,F,F3,F] + [1.46e-05,F,1.73e-05,
          9.71e-06,2.45e-05,1.47e-05,F2,1.58e-04,1.53e-04,1.55e-04,1.21e-04,
          2.84e-04,3.91e-04,3.81e-04,4.90e-04],
 "BIM":  [F]*15 + [F3,F,F,F,F,F,F2,F,F,F,F,F,F3,F,F3,F,F,F,F,F3,F,F3,
          4.85e-06,3.88e-05,2.43e-05,2.91e-05],
 "MIM":  [F,F,F,F,F,F3,F,F,F,F,F,F,F3,F,F3,F,F,F3,F,F,F,F,F,F,F,F,F,F,F,F,F,F,
          F3,F,F,F,3.16e-05,9.71e-06,1.21e-05,3.88e-05,9.71e-05],
 r"PGD ($\alpha$=0.01)": [F,F,F,F,F,F,6.81e-04,F,F3,F,F3,F,F,F,F,F3,F,F3,F,F3,
          F2,F2,F3,F,F3,F,F3,F,F2,F2,F,F,F,F,F,F,F,F,F,F,F],
 r"PGD ($\alpha$=0.05)": [F]*20 + [F3] + [F]*13 + [F3,F,F,F,1.46e-05,F,3.64e-05],
 r"PGD ($\alpha$=0.1)":  [F]*12 + [F3,F,F3,F,F,F,F2,F2,F2,F2,F,F,F,F,F,F,F3,F,F,
          F3,4.90e-06,1.70e-05,2.67e-05,1.94e-05,2.18e-05,5.10e-05,5.10e-05,
          3.88e-05,1.38e-04],
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
    "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
    "legend.fontsize": 6.4, "legend.handlelength": 1.5,
    "legend.columnspacing": 0.7, "legend.labelspacing": 0.2,
    "legend.handletextpad": 0.35, "legend.borderpad": 0.3,
    "axes.linewidth": 0.7,
})

XTICKS = [0.1, 0.2, 0.3, 0.4, 0.5]
XLIM = (0.093, 0.507)
psr_at = dict(zip(EPS, PSR))

for k in list(U_FR) + list(T_TR):
    pass
for d in (U_FR, U_BER, U_ERR, T_TR, T_BER, T_ERR):
    for k, v in d.items():
        assert len(v) == 41, (k, len(v))


def add_psr(ax):
    ax2 = ax.twiny()
    ax2.set_xlim(*XLIM)
    ax2.set_xticks(XTICKS)
    ax2.set_xticklabels([f"{psr_at[t]:.0f}" for t in XTICKS])
    ax2.tick_params(labelsize=7.2, length=2.5, pad=1.5)
    ax2.set_xlabel("PSR (dB)", fontsize=8, labelpad=2)


def finish(fig, axes, subs, stem,
           gap_label=0.045, gap_legend=0.045, label_fs=8.5):
    """Place panel labels and the shared legend at offsets fixed in INCHES.

    Relative (axes- or figure-fraction) offsets shrink with the figure while
    the text they position does not, so they collide at small heights. Working
    in inches keeps the spacing identical at any figure height.
    """
    fig.subplots_adjust(wspace=0.12)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    H = fig.get_figheight()
    inv = fig.transFigure.inverted()

    boxes = [ax.get_tightbbox(r).transformed(inv) for ax in axes]
    y0 = min(b.y0 for b in boxes)                 # bottom of axes + labels
    y_label = y0 - gap_label / H
    for b, sub in zip(boxes, subs):
        fig.text(0.5 * (b.x0 + b.x1), y_label, sub,
                 ha="center", va="top", fontsize=label_fs)

    label_h = (label_fs / 72.0) * 1.25 / H        # one text line, fig fraction
    anchor = y_label - label_h - gap_legend / H

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, anchor), framealpha=0.95)

    fig.savefig(f"{OUT}/{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{stem}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


# ---------------- rates ----------------
fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.2705), sharey=True)
for ax, data, sub in [(axes[0], U_FR, "(a) Untargeted"),
                      (axes[1], T_TR, "(b) Targeted")]:
    for name, y in data.items():
        ax.plot(EPS, y, label=name, markersize=1.9, linewidth=0.9,
                markeredgewidth=0.4, **STYLE[name])
    ax.set_xlim(*XLIM); ax.set_ylim(-4, 106)
    ax.set_xticks(XTICKS); ax.set_xticklabels(["0.1","0.2","0.3","0.4","0.5"])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, which="major", ls="--", alpha=0.3, linewidth=0.45)
    ax.tick_params(length=2.5, pad=1.5)
    ax.set_xlabel(r"$\epsilon$", fontsize=9, labelpad=1)
    add_psr(ax)
axes[0].set_ylabel("Rate (%)", fontsize=9, labelpad=2)
finish(fig, axes, ["(a) Untargeted", "(b) Targeted"],
       "exp2_rates_sidebyside")

# ---------------- BER ----------------
fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.2704), sharey=True)
for ax, data, err, sub in [(axes[0], U_BER, U_ERR, "(a) Untargeted"),
                           (axes[1], T_BER, T_ERR, "(b) Targeted")]:
    for name in data:
        st = STYLE[name]
        y = np.array(data[name]); e = np.array(err[name])
        ax.plot(EPS, y, color=st["color"], ls=st["ls"], linewidth=0.9,
                label=name, zorder=2)
        m = e > 0
        ax.plot(EPS[m], y[m], ls="none", marker=st["marker"], color=st["color"],
                markersize=2.2, zorder=3)
        ax.plot(EPS[~m], y[~m], ls="none", marker=st["marker"],
                markerfacecolor="none", markeredgecolor=st["color"],
                markersize=1.9, markeredgewidth=0.5, zorder=3)
    ax.set_yscale("log")
    ax.set_xlim(*XLIM); ax.set_ylim(3e-6, 2e-3)
    ax.set_xticks(XTICKS); ax.set_xticklabels(["0.1","0.2","0.3","0.4","0.5"])
    ax.set_yticks([1e-5, 1e-4, 1e-3])
    ax.grid(True, which="major", ls="--", alpha=0.3, linewidth=0.45)
    ax.tick_params(length=2.5, pad=1.5)
    ax.set_xlabel(r"$\epsilon$", fontsize=9, labelpad=1)
    add_psr(ax)
axes[0].set_ylabel("BER", fontsize=9, labelpad=2)
finish(fig, axes, ["(a) Untargeted", "(b) Targeted"],
       "exp2_ber_sidebyside")

print("saved")
