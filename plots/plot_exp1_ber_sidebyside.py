#!/usr/bin/env python3
"""Experiment 1 bit error rate, two panels side by side inside ONE IEEE column.

(a) untargeted, (b) targeted.
Shared y-axis, one legend spanning both panels, PSR on the top axis.
Matches the layout of exp1_rates_sidebyside.
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

exec(open(os.path.join(HERE, "plot_exp1_singlecol.py")).read().split("STYLE =")[0])

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

XTICKS = [0.01, 0.02, 0.05, 0.1]
XLIM = (0.0093, 0.108)


def psr(e):
    return 10 * np.log10(2 * np.asarray(e) ** 2)


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


fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.2704), sharey=True)

panels = [
    (axes[0], UNTARG_BER, "(a) Untargeted"),
    (axes[1], TARG_BER,  "(b) Targeted"),
]

for ax, data, sub in panels:
    for name, y in data.items():
        ax.plot(EPS, y, label=name, markersize=2.4, linewidth=0.95,
                markeredgewidth=0.5, **STYLE[name])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*XLIM)
    ax.set_ylim(1.9e-3, 3.4e-2)
    ax.set_xticks(XTICKS)
    ax.set_xticklabels(["0.01", "0.02", "0.05", "0.1"])
    ax.set_yticks([2e-3, 5e-3, 1e-2, 2e-2])
    ax.set_yticklabels([r"$2\!\times\!10^{-3}$", r"$5\!\times\!10^{-3}$",
                        r"$10^{-2}$", r"$2\!\times\!10^{-2}$"])
    ax.grid(True, which="major", ls="--", alpha=0.3, linewidth=0.45)
    ax.tick_params(length=2.5, pad=1.5)
    ax2 = ax.twiny()
    ax2.set_xscale("log")
    ax2.set_xlim(*XLIM)
    ax2.set_xticks([0.01, 0.03, 0.1])
    ax2.set_xticklabels([f"{psr(v):.0f}" for v in [0.01, 0.03, 0.1]])
    ax2.tick_params(labelsize=7.2, length=2.5, pad=1.5)
    ax2.set_xlabel("PSR (dB)", fontsize=8, labelpad=2)

axes[0].set_ylabel("BER", fontsize=9, labelpad=2)
for ax in axes:
    ax.set_xlabel(r"$\epsilon$", fontsize=9, labelpad=1)

finish(fig, axes, ["(a) Untargeted", "(b) Targeted"],
       "exp1_ber_sidebyside")
print("saved")
