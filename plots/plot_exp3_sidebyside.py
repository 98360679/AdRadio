#!/usr/bin/env python3
"""Experiment 3 (two-channel OTA), single-column side-by-side figures.

(a) untargeted, (b) targeted, for both the attack rate and the BER.
PSR follows the nominal 10*log10(2*eps^2), which matches the sheet exactly.
Censored BER points (zero observed errors, reported as the 3/N bound) are drawn
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

# pull the Experiment 3 dictionaries straight from the existing scripts
ns_u, ns_t = {}, {}
exec(open(os.path.join(HERE, "plot_exp3_untargeted.py")).read().split("STYLE =")[0], ns_u)
exec(open(os.path.join(HERE, "plot_exp3_targeted.py")).read().split("STYLE =")[0], ns_t)
EPS = ns_u["DELTA"]
U, T = ns_u["DATA"], ns_t["DATA"]

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


def add_psr(ax):
    ax2 = ax.twiny()
    ax2.set_xscale("log")
    ax2.set_xlim(*XLIM)
    ax2.set_xticks([0.01, 0.03, 0.1])
    ax2.set_xticklabels([f"{psr(v):.0f}" for v in [0.01, 0.03, 0.1]])
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
fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.2704), sharey=True)
for ax, data, sub in [(axes[0], U, "(a) Untargeted"), (axes[1], T, "(b) Targeted")]:
    for name, d in data.items():
        ax.plot(EPS, d["fr"], label=name, markersize=2.4, linewidth=0.95,
                markeredgewidth=0.5, **STYLE[name])
    ax.set_xscale("log")
    ax.set_xlim(*XLIM); ax.set_ylim(-4, 106)
    ax.set_xticks(XTICKS); ax.set_xticklabels(["0.01", "0.02", "0.05", "0.1"])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, which="major", ls="--", alpha=0.3, linewidth=0.45)
    ax.tick_params(length=2.5, pad=1.5)
    ax.set_xlabel(r"$\epsilon$", fontsize=9, labelpad=1)
    add_psr(ax)
axes[0].set_ylabel("Rate (%)", fontsize=9, labelpad=2)
finish(fig, axes, ["(a) Untargeted", "(b) Targeted"],
       "exp3_rates_sidebyside")

# ---------------- BER ----------------
fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.2704), sharey=True)
for ax, data, sub in [(axes[0], U, "(a) Untargeted"), (axes[1], T, "(b) Targeted")]:
    for name, d in data.items():
        st = STYLE[name]
        y = np.array(d["ber"]); e = np.array(d["err"])
        ax.plot(EPS, y, color=st["color"], ls=st["ls"], linewidth=0.95,
                label=name, zorder=2)
        m = e > 0
        ax.plot(EPS[m], y[m], ls="none", marker=st["marker"], color=st["color"],
                markersize=2.4, zorder=3)
        ax.plot(EPS[~m], y[~m], ls="none", marker=st["marker"],
                markerfacecolor="none", markeredgecolor=st["color"],
                markersize=2.1, markeredgewidth=0.55, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*XLIM); ax.set_ylim(3e-6, 2.0)
    ax.set_xticks(XTICKS); ax.set_xticklabels(["0.01", "0.02", "0.05", "0.1"])
    ax.set_yticks([1e-5, 1e-3, 1e-1])
    ax.grid(True, which="major", ls="--", alpha=0.3, linewidth=0.45)
    ax.tick_params(length=2.5, pad=1.5)
    ax.set_xlabel(r"$\epsilon$", fontsize=9, labelpad=1)
    add_psr(ax)
axes[0].set_ylabel("BER", fontsize=9, labelpad=2)
finish(fig, axes, ["(a) Untargeted", "(b) Targeted"],
       "exp3_ber_sidebyside")

# ---------------- numbers for the text ----------------
def onset(v):
    return EPS[next(i for i, x in enumerate(v) if x > 0)]
def sat(v):
    h = [EPS[i] for i, x in enumerate(v) if x >= 100]
    return h[0] if h else None

print("EXP 3 untargeted: onset / 100% / BER at 0.1 / frames rejected at 0.1")
for k, d in U.items():
    print(f"  {k:<22} {onset(d['fr']):.3f}  {sat(d['fr'])}  "
          f"{d['ber'][-1]:.2e}  {d['rej'][-1]}/103")
print("EXP 3 targeted:")
for k, d in T.items():
    print(f"  {k:<22} {onset(d['fr']):.3f}  {sat(d['fr'])}  "
          f"{d['ber'][-1]:.2e}  {d['rej'][-1]}/103")
print("\nBER<=1e-3 boundary, untargeted FGSM: between",
      [f"{EPS[i]:.3f}" for i in range(len(EPS)-1)
       if U['FGSM']['ber'][i] <= 1e-3 < U['FGSM']['ber'][i+1]])
print("saved")
