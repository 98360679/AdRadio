#!/usr/bin/env python3
"""Full-range untargeted BER (Experiment 3, two-channel OTA, white-box CVNN), IEEE style.

Shows where the link degrades: BER sits at the rule-of-three floor through the operating
window, rises from eps ~ 0.3, crosses 1e-2 near eps ~ 0.5, and plateaus by eps ~ 0.8-1.0.
Measured SCALE=1 data (docs/a1cvnn_<m>_ota_ber.csv, 107 frames). Untargeted; at these high
budgets the targeted iterative attacks are weaker (see paper), so only untargeted is shown.
-> docs/exp3_ber_fullrange_ieee.{pdf,svg,png}"""
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
GRID, MUTED, MARK = '#d9d9d9', '#777777', '#b5544e'
BITS = 500 * 8
XLIM = (0.0095, 1.05)


def rows(m):
    return sorted(csv.DictReader(open(f'docs/a1cvnn_{m}_ota_ber.csv')), key=lambda r: float(r['epsilon']))


def eps2psr(e):
    return 10.0 * np.log10(2.0 * np.asarray(e, float) ** 2)


fig, ax = plt.subplots(figsize=(7.16, 3.6))

# plateau band (where BER has saturated near its max)
ax.axvspan(0.7, XLIM[1], color='#f0efe9', zorder=0)

for m in METHODS:
    c = MC[m]; eps, y, meas = [], [], []
    for r in rows(m):
        e = float(r['epsilon']); b = float(r['ber']); fd = float(r['frames_decoded'])
        eps.append(e)
        if b > 0: y.append(b); meas.append(True)
        elif fd > 0: y.append(3.0 / (fd * BITS)); meas.append(False)
        else: y.append(np.nan); meas.append(False)
    ax.plot(eps, y, '-', color=c, lw=1.4, label=LAB[m], zorder=3)
    for e, v, mm in zip(eps, y, meas):
        ax.plot([e], [v], marker='o', ms=4.0, mfc=(c if mm else 'white'), mec=c, mew=0.9, zorder=4)

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(*XLIM); ax.set_ylim(3e-6, 1.0)

# 1e-2 link-damage line
ax.axhline(1e-2, color=MUTED, lw=0.8, ls=(0, (3, 3)))
ax.text(XLIM[0] * 1.05, 1.2e-2, r'BER $10^{-2}$ (link damage)', fontsize=7.5, color=MUTED, va='bottom')
# rise onset
ax.axvline(0.3, color=MARK, lw=0.9, ls=(0, (2, 2)))
ax.text(0.3, 3.4e-6, 'rise onset\n$\\varepsilon\\approx0.3$', fontsize=8, color=MARK, ha='center', va='bottom')
# 1e-2 crossing
ax.axvline(0.5, color=MARK, lw=0.9, ls=(0, (2, 2)))
ax.text(0.5, 1.7e0, r'crosses $10^{-2}$' + '\n$\\varepsilon\\approx0.5$', fontsize=8, color=MARK, ha='center', va='top')
# plateau label
ax.text(0.86, 1.7e0, 'plateau', fontsize=8.5, color='#6b6a62', ha='center', va='top', style='italic')

ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
ax.set_xticklabels(['0.01', '0.02', '0.05', '0.1', '0.2', '0.5', '1.0'])
ax.xaxis.set_minor_locator(mticker.NullLocator())
ax.set_xlabel(r'$\varepsilon$'); ax.set_ylabel('bit error rate')
ax.grid(True, color=GRID, lw=0.5)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
for sp in ['left', 'bottom']: ax.spines[sp].set_color(MUTED)
ax.tick_params(colors='#333333', length=3)

top = ax.twiny(); top.set_xscale('log'); top.set_xlim(ax.get_xlim())
t = [0.01, 0.031623, 0.1, 0.31623, 1.0]
top.set_xticks(t); top.xaxis.set_minor_locator(mticker.NullLocator())
top.set_xticklabels([f'{eps2psr(x):.0f}' for x in t])
top.set_xlabel('PSR (dB)', labelpad=4, fontsize=9); top.tick_params(length=3)
for sp in ['left', 'right']: top.spines[sp].set_visible(False)
top.spines['top'].set_color(MUTED)

ax.legend(loc='center left', bbox_to_anchor=(0.012, 0.52), frameon=False, ncol=2, handlelength=1.3,
          columnspacing=0.9, labelspacing=0.3, borderaxespad=0.0)
fig.tight_layout(pad=0.5)
for ext in ('pdf', 'svg', 'png'):
    fig.savefig(f'docs/exp3_ber_fullrange_ieee.{ext}', dpi=300, bbox_inches='tight')
print('wrote docs/exp3_ber_fullrange_ieee.{pdf,svg,png}')
