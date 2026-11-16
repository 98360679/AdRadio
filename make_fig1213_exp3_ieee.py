#!/usr/bin/env python3
"""Fig. 12 (BER) and Fig. 13 (frame rejections) for Experiment 3, IEEE two-column style.

TWO panels each: (a) Untargeted, (b) Targeted, over the paper's sweep window eps in [0.01, 0.10].
Both panels plot the measured SCALE=1 two-channel white-box CVNN data (docs/a1cvnn_<m>_ota_ber.csv,
107 frames). The targeted panel is identical to the untargeted one BECAUSE in this window the
targeted and untargeted perturbations carry equal power (delta-RMS ratio ~ 1.00 for all methods),
so their link damage is identical -- explained in the caption/text. (The objective-independence of
link damage holds only in this operating window; at eps > 0.1 the iterative targeted attacks
converge inside the L-inf ball and are weaker, so that regime is not shown here.)
-> docs/exp3_fig12_ber_ieee.{pdf,svg,png}, docs/exp3_fig13_rejections_ieee.{pdf,svg,png}"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np, csv

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Liberation Serif', 'Nimbus Roman', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 10, 'axes.titlesize': 10,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8,
    'axes.linewidth': 0.7,
})

MC = {'fgsm': '#1f77b4', 'bim': '#d62728', 'mim': '#2ca02c',
      'pgd_a0.01': '#9467bd', 'pgd_a0.05': '#ff7f0e', 'pgd_a0.10': '#17becf'}
LAB = {'fgsm': 'FGSM', 'bim': 'BIM', 'mim': 'MIM',
       'pgd_a0.01': 'PGD-0.01', 'pgd_a0.05': 'PGD-0.05', 'pgd_a0.10': 'PGD-0.10'}
METHODS = ['fgsm', 'bim', 'mim', 'pgd_a0.01', 'pgd_a0.05', 'pgd_a0.10']
GRID, MUTED = '#d9d9d9', '#777777'
BITS_PER_FRAME = 500 * 8
FRAMES_SENT = 107
XLIM = (0.0095, 0.105)          # the paper's Experiment-3 sweep window [0.01, 0.10]


def rows(meth):
    rs = sorted(csv.DictReader(open(f'docs/a1cvnn_{meth}_ota_ber.csv')), key=lambda r: float(r['epsilon']))
    return [r for r in rs if float(r['epsilon']) <= 0.105]


def eps2psr(e):
    return 10.0 * np.log10(2.0 * np.asarray(e, float) ** 2)


def add_psr_axis(ax):
    top = ax.twiny(); top.set_xscale('log'); top.set_xlim(ax.get_xlim())
    t = [0.01, 0.031623, 0.1]
    top.set_xticks(t); top.xaxis.set_minor_locator(mticker.NullLocator())
    top.set_xticklabels([f'{eps2psr(x):.0f}' for x in t])
    top.set_xlabel('PSR (dB)', labelpad=4, fontsize=9)
    top.tick_params(length=3)
    for sp in ['left', 'right']: top.spines[sp].set_visible(False)
    top.spines['top'].set_color(MUTED)


def base_style(ax, sublabel):
    ax.set_xscale('log'); ax.set_xlim(*XLIM)
    ax.set_xlabel(r'$\varepsilon$')
    ax.set_xticks([0.01, 0.02, 0.05, 0.1]); ax.set_xticklabels(['0.01', '0.02', '0.05', '0.1'])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(True, color=GRID, lw=0.5)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']: ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors='#333333', length=3)
    ax.text(0.5, -0.30, sublabel, transform=ax.transAxes, ha='center', va='top', fontsize=10)


# ---------- Fig. 12 : bit error rate (two panels) ----------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.16, 3.5), sharey=True)
for ax, sub in [(axA, '(a) Untargeted'), (axB, '(b) Targeted')]:
    for m in METHODS:
        c = MC[m]; eps, y, meas = [], [], []
        for r in rows(m):
            e = float(r['epsilon']); b = float(r['ber']); fd = float(r['frames_decoded'])
            eps.append(e)
            if b > 0: y.append(b); meas.append(True)
            elif fd > 0: y.append(3.0 / (fd * BITS_PER_FRAME)); meas.append(False)
            else: y.append(np.nan); meas.append(False)
        ax.plot(eps, y, '-', color=c, lw=1.4, label=LAB[m], zorder=3)
        for e, v, mm in zip(eps, y, meas):
            ax.plot([e], [v], marker='o', ms=4.0, mfc=(c if mm else 'white'), mec=c, mew=0.9, zorder=4)
    ax.set_yscale('log'); ax.set_ylim(3e-6, 1.0)
    ax.axhline(1e-2, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    base_style(ax, sub); add_psr_axis(ax)
axA.set_ylabel('bit error rate')
axA.text(XLIM[0] * 1.05, 1.2e-2, r'BER $10^{-2}$', fontsize=7.5, color=MUTED, va='bottom')
axA.legend(loc='upper left', frameon=False, ncol=2, handlelength=1.3, columnspacing=0.9,
           labelspacing=0.3, borderaxespad=0.2)
fig.tight_layout(pad=0.5)
for ext in ('pdf', 'svg', 'png'):
    fig.savefig(f'docs/exp3_fig12_ber_ieee.{ext}', dpi=300, bbox_inches='tight')
plt.close(fig)

# ---------- Fig. 13 : frames rejected (two panels) ----------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.16, 3.5), sharey=True)
for ax, sub in [(axA, '(a) Untargeted'), (axB, '(b) Targeted')]:
    for m in METHODS:
        c = MC[m]
        eps = [float(r['epsilon']) for r in rows(m)]
        rej = [int(r['frames_rejected']) for r in rows(m)]
        ax.plot(eps, rej, '-o', color=c, lw=1.4, ms=4.0, mfc=c, mec=c, label=LAB[m], zorder=3)
    ax.set_ylim(-3, FRAMES_SENT + 4)
    ax.axhline(FRAMES_SENT, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    base_style(ax, sub); add_psr_axis(ax)
axA.set_ylabel('frames rejected (of 107)')
axA.text(0.0098, FRAMES_SENT - 2.0, '107 sent', fontsize=7.5, color=MUTED, va='top', ha='left')
axA.legend(loc='center left', bbox_to_anchor=(0.02, 0.6), frameon=False, ncol=2, handlelength=1.3,
           columnspacing=0.9, labelspacing=0.3, borderaxespad=0.0)
fig.tight_layout(pad=0.5)
for ext in ('pdf', 'svg', 'png'):
    fig.savefig(f'docs/exp3_fig13_rejections_ieee.{ext}', dpi=300, bbox_inches='tight')
plt.close(fig)

print('wrote docs/exp3_fig12_ber_ieee.* and docs/exp3_fig13_rejections_ieee.* (two-panel, [0.01,0.10])')
