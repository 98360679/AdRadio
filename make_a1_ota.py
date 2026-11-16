#!/usr/bin/env python3
"""A1 white-box CNN1D over the air, all 6 methods, 16-point eps: fooling-rate curve and BER curve.
docs/a1_<method>_ota_ber.csv -> docs/a1_cnn1d_ota.png/.svg"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, csv

METHODS = [('fgsm', '#1f77b4', 'o', 'FGSM'), ('bim', '#d62728', 's', 'BIM'), ('mim', '#2ca02c', '^', 'MIM'),
           ('pgd_a0.01', '#9467bd', 'D', 'PGD α0.01'), ('pgd_a0.05', '#ff7f0e', 'v', 'PGD α0.05'),
           ('pgd_a0.10', '#17becf', 'P', 'PGD α0.10')]
INK, INK2, MUTED, SURF, GRID = '#0b0b0b', '#52514e', '#8a8880', '#fcfcfb', '#e4e3de'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': SURF, 'axes.facecolor': SURF})
BER_FLOOR = 5e-6

def load(m):
    r = list(csv.DictReader(open(f'docs/a1_{m}_ota_ber.csv')))
    eps = [float(x['epsilon']) for x in r]
    fool = [float(x['fooling_pct']) for x in r]
    ber = [max(float(x['ber']) if x['ber'] not in ('', None) else 0.0, BER_FLOOR) for x in r]
    return eps, fool, ber

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.6))
for m, c, mk, lab in METHODS:
    eps, fool, ber = load(m)
    axL.plot(eps, fool, marker=mk, color=c, lw=2.0, ms=5, label=lab, solid_capstyle='round')
    axR.plot(eps, ber, marker=mk, color=c, lw=2.0, ms=5, label=lab, solid_capstyle='round')

axL.axhline(50, color=MUTED, lw=1.0, ls=(0, (2, 3)))
axL.set_xscale('log'); axL.set_ylim(-4, 105); axL.set_xlim(0.009, 1.1)
axL.set_xlabel('perturbation budget ε (log)', fontsize=11, color=INK)
axL.set_ylabel('fooling rate — victim device_5 overwritten (%)', fontsize=11, color=INK)
axL.set_title('A1 fooling rate — white-box CNN1D, over the air', fontsize=12, color=INK, fontweight='bold', loc='left', pad=8)
axL.legend(loc='lower right', frameon=False, fontsize=9, labelcolor=INK2, ncol=2)

axR.set_xscale('log'); axR.set_yscale('log'); axR.set_xlim(0.009, 1.1); axR.set_ylim(3e-6, 1.0)
axR.axhline(1e-2, color=MUTED, lw=1.0, ls=(0, (2, 3)))
axR.text(0.0095, BER_FLOOR * 1.25, 'clean (BER 0)', fontsize=8.5, color=MUTED, va='bottom')
axR.set_xlabel('perturbation budget ε (log)', fontsize=11, color=INK)
axR.set_ylabel('bit error rate (log)', fontsize=11, color=INK)
axR.set_title('A1 link denial — BER vs power', fontsize=12, color=INK, fontweight='bold', loc='left', pad=8)
axR.legend(loc='lower right', frameon=False, fontsize=9, labelcolor=INK2, ncol=2)

for ax in (axL, axR):
    ax.grid(True, color=GRID, lw=0.8)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']: ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9.5)

fig.suptitle('A1 — white-box CNN1D attack over the air (all 6 methods, 16-point ε): overwrite by ε≈0.02–0.03, '
             'denial (BER, frame loss) past ε≈0.5',
             fontsize=11.5, fontweight='bold', color=INK, x=0.01, ha='left')
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig('docs/a1_cnn1d_ota.png', dpi=220, bbox_inches='tight', facecolor=SURF)
fig.savefig('docs/a1_cnn1d_ota.svg', bbox_inches='tight', facecolor=SURF)
print('wrote docs/a1_cnn1d_ota.png and .svg')
