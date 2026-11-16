#!/usr/bin/env python3
"""Fig. 14 (reduced) — A1 white-box OTA fooling rate, CNN1D vs CVNN, IEEE two-column style.

Fooling-only model comparison (BER panels dropped; CVNN BER is covered by Fig. 12). Shows the
real-valued CNN1D is evaded earlier (eps ~ 0.02-0.03) than the complex-valued CVNN (eps ~ 0.08-0.10)
-- the ordering is a property of the attribution task, not the representation.
CNN1D = docs/a1_<m>_ota_ber.csv, CVNN = docs/a1cvnn_<m>_ota_ber.csv (same session, 107 frames).
-> docs/exp3_fig14_fooling_ieee.{pdf,svg,png}"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np, csv

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Liberation Serif', 'Nimbus Roman', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'axes.labelsize': 10, 'axes.titlesize': 10,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8, 'axes.linewidth': 0.7})

MC = {'fgsm': '#1f77b4', 'bim': '#d62728', 'mim': '#2ca02c',
      'pgd_a0.01': '#9467bd', 'pgd_a0.05': '#ff7f0e', 'pgd_a0.10': '#17becf'}
LAB = {'fgsm': 'FGSM', 'bim': 'BIM', 'mim': 'MIM',
       'pgd_a0.01': 'PGD-0.01', 'pgd_a0.05': 'PGD-0.05', 'pgd_a0.10': 'PGD-0.10'}
METHODS = ['fgsm', 'bim', 'mim', 'pgd_a0.01', 'pgd_a0.05', 'pgd_a0.10']
GRID, MUTED = '#d9d9d9', '#777777'
XLIM = (0.0095, 0.105)


def load(prefix, m):
    r = [x for x in csv.DictReader(open(f'docs/{prefix}{m}_ota_ber.csv')) if float(x['epsilon']) <= 0.105]
    r.sort(key=lambda x: float(x['epsilon']))
    return [float(x['epsilon']) for x in r], [float(x['fooling_pct']) for x in r]


def eps2psr(e):
    return 10.0 * np.log10(2.0 * np.asarray(e, float) ** 2)


def style(ax, sub):
    ax.set_xscale('log'); ax.set_xlim(*XLIM); ax.set_ylim(-4, 105)
    ax.axhline(50, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.set_xticks([0.01, 0.02, 0.05, 0.1]); ax.set_xticklabels(['0.01', '0.02', '0.05', '0.1'])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel(r'$\varepsilon$')
    ax.grid(True, color=GRID, lw=0.5)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']: ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors='#333333', length=3)
    ax.text(0.5, -0.30, sub, transform=ax.transAxes, ha='center', va='top', fontsize=10)
    top = ax.twiny(); top.set_xscale('log'); top.set_xlim(ax.get_xlim())
    t = [0.01, 0.031623, 0.1]; top.set_xticks(t); top.xaxis.set_minor_locator(mticker.NullLocator())
    top.set_xticklabels([f'{eps2psr(x):.0f}' for x in t])
    top.set_xlabel('PSR (dB)', labelpad=4, fontsize=9); top.tick_params(length=3)
    for sp in ['left', 'right']: top.spines[sp].set_visible(False)
    top.spines['top'].set_color(MUTED)


fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.16, 3.5), sharey=True)
for ax, prefix, sub in [(axA, 'a1_', '(a) CNN1D'), (axB, 'a1cvnn_', '(b) CVNN')]:
    for m in METHODS:
        e, f = load(prefix, m)
        ax.plot(e, f, '-o', color=MC[m], lw=1.4, ms=4.0, mfc=MC[m], mec=MC[m], label=LAB[m], zorder=3)
    style(ax, sub)
axA.set_ylabel('fooling rate (%)')
axA.legend(loc='upper left', frameon=False, ncol=2, handlelength=1.3, columnspacing=0.9,
           labelspacing=0.3, borderaxespad=0.2)
fig.tight_layout(pad=0.5)
for ext in ('pdf', 'svg', 'png'):
    fig.savefig(f'docs/exp3_fig14_fooling_ieee.{ext}', dpi=300, bbox_inches='tight')
print('wrote docs/exp3_fig14_fooling_ieee.{pdf,svg,png}')
