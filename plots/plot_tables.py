#!/usr/bin/env python3
"""Figures drawn from the LaTeX tables, not from the full sweeps.

One figure per table, each (a) victim-class accuracy, (b) bit error rate,
over exactly the epsilon rows and the four methods that the tables report.

  Experiment 1 / Table 5 (plot_exp1_singlecol.py)  -> exp1_table_sidebyside.{pdf,png}
  Experiment 2 / Table 6 (plot_exp2_sidebyside.py) -> exp2_table_sidebyside.{pdf,png}
  Experiment 3 / Table 7 (plot_exp3_untargeted.py) -> exp3_table_sidebyside.{pdf,png}

Accuracy is 100 - fooling rate, computed in Decimal so the plotted values are
the table's values exactly. Experiment 2 BER points at which zero bit errors
were observed are rule-of-three upper bounds and are drawn hollow.
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
OUT = os.environ.get("FIG_OUT", os.path.join(HERE, "output"))
if "__file__" in globals():
    os.makedirs(OUT, exist_ok=True)
from decimal import Decimal as D
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


n1 = {}
exec(open(os.path.join(HERE, "plot_exp1_singlecol.py")).read().split("STYLE =")[0], n1)
n2 = {}
exec(open(os.path.join(HERE, "plot_exp2_sidebyside.py")).read().split("STYLE =")[0], n2)

M = ["FGSM", "BIM", "MIM", r"PGD ($\alpha$=0.01)"]
STYLE = {
    "FGSM":                 dict(marker="o", color="#1f77b4", ls="-"),
    "BIM":                  dict(marker="s", color="#d62728", ls="-"),
    "MIM":                  dict(marker="^", color="#2ca02c", ls="-"),
    r"PGD ($\alpha$=0.01)": dict(marker="D", color="#9467bd", ls="--"),
}

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 10,
    "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
    "legend.fontsize": 6.4, "legend.handlelength": 1.5,
    "legend.columnspacing": 0.7, "legend.labelspacing": 0.2,
    "legend.handletextpad": 0.35, "legend.borderpad": 0.3,
    "axes.linewidth": 0.7,
})


def acc(fr):
    """100 - fooling rate, exact, as the table prints it."""
    return float(D(100) - D(repr(float(fr))))


def finish(fig, axes, subs, stem, gap_label=0.045, gap_legend=0.045,
           label_fs=8.5, handles=None, wspace=0.42):
    """Panel labels and shared legend at offsets fixed in INCHES."""
    fig.subplots_adjust(wspace=wspace)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    H = fig.get_figheight()
    inv = fig.transFigure.inverted()

    boxes = [ax.get_tightbbox(r).transformed(inv) for ax in axes]
    y0 = min(b.y0 for b in boxes)
    y_label = y0 - gap_label / H
    for b, sub in zip(boxes, subs):
        fig.text(0.5 * (b.x0 + b.x1), y_label, sub,
                 ha="center", va="top", fontsize=label_fs)

    label_h = (label_fs / 72.0) * 1.25 / H
    anchor = y_label - label_h - gap_legend / H

    if handles is None:
        handles, labels = axes[0].get_legend_handles_labels()
    else:
        labels = [h.get_label() for h in handles]
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, anchor), framealpha=0.95)

    fig.savefig(f"{OUT}/{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{stem}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def psr_axis(ax, xticks, psr_at, xlim):
    ax2 = ax.twiny()
    ax2.set_xlim(*xlim)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([f"{psr_at[t]:.0f}" for t in xticks])
    ax2.tick_params(labelsize=7.2, length=2.5, pad=1.5)
    ax2.set_xlabel("PSR (dB)", fontsize=8, labelpad=2)


# ===================== Experiment 1 (Table 5) =====================
E1, FR1, BE1 = n1["EPS"], n1["UNTARG_FR"], n1["UNTARG_BER"]
rows1 = [0.010, 0.012, 0.014, 0.016, 0.018, 0.020, 0.030, 0.040]
i1 = [int(np.argmin(np.abs(E1 - e))) for e in rows1]
x1 = np.array(rows1)
XL1 = (0.0085, 0.0415)
XT1 = [0.01, 0.02, 0.03, 0.04]
psr1 = {t: 10 * np.log10(2 * t ** 2) for t in XT1}

fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.4327))
for name in M:
    axes[0].plot(x1, [acc(FR1[name][i]) for i in i1], label=name,
                 markersize=2.4, linewidth=0.9, markeredgewidth=0.4,
                 **STYLE[name])
    axes[1].plot(x1, [BE1[name][i] * 1e3 for i in i1], label=name,
                 markersize=2.4, linewidth=0.9, markeredgewidth=0.4,
                 **STYLE[name])
axes[0].set_ylim(-4, 106)
axes[0].set_yticks([0, 20, 40, 60, 80, 100])
axes[0].set_ylabel("Accuracy (%)", fontsize=9, labelpad=2)
axes[1].set_ylabel(r"BER ($\times10^{-3}$)", fontsize=9, labelpad=2)
axes[1].set_ylim(0, 10.6)
for ax in axes:
    ax.set_xlim(*XL1)
    ax.set_xticks(XT1)
    ax.set_xticklabels(["0.01", "0.02", "0.03", "0.04"])
    ax.grid(True, which="major", ls="--", alpha=0.3, linewidth=0.45)
    ax.tick_params(length=2.5, pad=1.5)
    ax.set_xlabel(r"$\epsilon$", fontsize=9, labelpad=1)
    psr_axis(ax, XT1, psr1, XL1)
finish(fig, axes, ["(a) Accuracy", "(b) Proxy BER"], "exp1_table_sidebyside")
print("exp1_table_sidebyside saved")


# ===================== Experiment 2 (Table 6) =====================
E2, PSR2, FR2, BE2 = n2["EPS"], n2["PSR"], n2["U_FR"], n2["U_BER"]
F, F2, F3 = n2["F"], n2["F2"], n2["F3"]
CENS = {F, F2, F3}
rows2 = [0.10, 0.26, 0.30, 0.34, 0.38, 0.42, 0.46, 0.50]
i2 = [int(np.argmin(np.abs(E2 - e))) for e in rows2]
x2 = np.array(rows2)
XL2 = (0.085, 0.515)
XT2 = [0.1, 0.2, 0.3, 0.4, 0.5]
psr_full = dict(zip([round(float(v), 2) for v in E2], PSR2))
psr2 = {t: psr_full[round(t, 2)] for t in XT2}

fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.4327))
for name in M:
    axes[0].plot(x2, [acc(FR2[name][i]) for i in i2], label=name,
                 markersize=2.4, linewidth=0.9, markeredgewidth=0.4,
                 **STYLE[name])

    y = np.array([BE2[name][i] for i in i2])
    cen = np.array([BE2[name][i] in CENS for i in i2])
    st = STYLE[name]
    axes[1].plot(x2, y, color=st["color"], ls=st["ls"], linewidth=0.9,
                 label=name, zorder=2)
    # measured points filled, censored upper bounds hollow
    axes[1].plot(x2[~cen], y[~cen], ls="none", marker=st["marker"],
                 color=st["color"], markersize=2.4, markeredgewidth=0.4,
                 zorder=3)
    axes[1].plot(x2[cen], y[cen], ls="none", marker=st["marker"],
                 markerfacecolor="white", markeredgecolor=st["color"],
                 markersize=2.4, markeredgewidth=0.5, zorder=3)

axes[0].set_ylim(-4, 106)
axes[0].set_yticks([0, 20, 40, 60, 80, 100])
axes[0].set_ylabel("Accuracy (%)", fontsize=9, labelpad=2)
axes[1].set_yscale("log")
axes[1].set_ylim(4e-6, 2e-3)
axes[1].set_ylabel("BER", fontsize=9, labelpad=2)
for ax in axes:
    ax.set_xlim(*XL2)
    ax.set_xticks(XT2)
    ax.set_xticklabels(["0.1", "0.2", "0.3", "0.4", "0.5"])
    ax.grid(True, which="major", ls="--", alpha=0.3, linewidth=0.45)
    ax.tick_params(length=2.5, pad=1.5)
    ax.set_xlabel(r"$\epsilon$", fontsize=9, labelpad=1)
    psr_axis(ax, XT2, psr2, XL2)

handles = [Line2D([], [], label=n, markersize=2.4, linewidth=0.9,
                  markeredgewidth=0.4, **STYLE[n]) for n in M]
finish(fig, axes, ["(a) Accuracy", "(b) BER"], "exp2_table_sidebyside",
       handles=handles)
print("exp2_table_sidebyside saved")


# ===================== Experiment 3 (Table 7) =====================
n3 = {}
exec(open(os.path.join(HERE, "plot_exp3_untargeted.py")).read().split("STYLE =")[0], n3)
E3, D3, FC3 = np.asarray(n3["DELTA"], float), n3["DATA"], n3["F"]
FR3 = {m: D3[m]["fr"] for m in D3}
BE3 = {m: D3[m]["ber"] for m in D3}
rows3 = [0.010, 0.020, 0.025, 0.030, 0.040, 0.050, 0.080, 0.100]
i3 = [int(np.argmin(np.abs(E3 - e))) for e in rows3]
x3 = np.array(rows3)
XL3 = (0.008, 0.104)
XT3 = [0.01, 0.03, 0.05, 0.08, 0.10]
psr3 = {t: 20 * np.log10(t) + 3.01 for t in XT3}

fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.4327))
for name in M:
    axes[0].plot(x3, [acc(FR3[name][i]) for i in i3], label=name,
                 markersize=2.4, linewidth=0.9, markeredgewidth=0.4,
                 **STYLE[name])

    y = np.array([BE3[name][i] for i in i3])
    cen = np.array([BE3[name][i] == FC3 for i in i3])
    st = STYLE[name]
    axes[1].plot(x3, y, color=st["color"], ls=st["ls"], linewidth=0.9,
                 label=name, zorder=2)
    # measured points filled, censored (rule-of-three) upper bounds hollow
    axes[1].plot(x3[~cen], y[~cen], ls="none", marker=st["marker"],
                 color=st["color"], markersize=2.4, markeredgewidth=0.4, zorder=3)
    axes[1].plot(x3[cen], y[cen], ls="none", marker=st["marker"],
                 markerfacecolor="white", markeredgecolor=st["color"],
                 markersize=2.4, markeredgewidth=0.5, zorder=3)

axes[0].set_ylim(-4, 106)
axes[0].set_yticks([0, 20, 40, 60, 80, 100])
axes[0].set_ylabel("Accuracy (%)", fontsize=9, labelpad=2)
axes[1].set_yscale("log")
axes[1].set_ylim(4e-6, 1)
axes[1].set_ylabel("BER", fontsize=9, labelpad=2)
for ax in axes:
    ax.set_xlim(*XL3)
    ax.set_xticks(XT3)
    ax.set_xticklabels(["0.01", "0.03", "0.05", "0.08", "0.10"])
    ax.grid(True, which="major", ls="--", alpha=0.3, linewidth=0.45)
    ax.tick_params(length=2.5, pad=1.5)
    ax.set_xlabel(r"$\epsilon$", fontsize=9, labelpad=1)
    psr_axis(ax, XT3, psr3, XL3)

handles = [Line2D([], [], label=n, markersize=2.4, linewidth=0.9,
                  markeredgewidth=0.4, **STYLE[n]) for n in M]
finish(fig, axes, ["(a) Accuracy", "(b) BER"], "exp3_table_sidebyside",
       handles=handles)
print("exp3_table_sidebyside saved")
