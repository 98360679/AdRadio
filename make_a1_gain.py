#!/usr/bin/env python3
"""A1 white-box CNN1D over the air: EXPECTED fooling (mean over 16 blind phases) vs
adversary TX gain (dB) at fixed eps=0.01, all 6 methods. The gain axis resolves the
graded power response that the fixed-gain eps sweep hides (max-over-phase there steps
straight to 100%). docs/../attack_a1_gain/g16_<method>.csv -> docs/a1_gain_sweep.png/.svg"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv, os
import os
CAPTURES = os.environ.get('CAPTURES_ROOT', os.path.expanduser('~/captures'))

CAP = f'{CAPTURES}/attack_a1_gain'
METHODS = [('fgsm', '#1f77b4', 'o', 'FGSM'), ('bim', '#d62728', 's', 'BIM'), ('mim', '#2ca02c', '^', 'MIM'),
           ('pgd_a0.01', '#9467bd', 'D', 'PGD α0.01'), ('pgd_a0.05', '#ff7f0e', 'v', 'PGD α0.05'),
           ('pgd_a0.10', '#17becf', 'P', 'PGD α0.10')]
INK, INK2, MUTED, SURF, GRID = '#0b0b0b', '#52514e', '#8a8880', '#fcfcfb', '#e4e3de'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': SURF, 'axes.facecolor': SURF})


def load(m):
    r = list(csv.DictReader(open(f'{CAP}/g16_{m}.csv')))
    g = [float(x['adv_gain_db']) for x in r]
    fm = [float(x['fool_mean_pct']) for x in r]
    return g, fm


fig, ax = plt.subplots(figsize=(9.5, 6))
for m, c, mk, lab in METHODS:
    p = f'{CAP}/g16_{m}.csv'
    if not os.path.exists(p):
        continue
    g, fm = load(m)
    ax.plot(g, fm, marker=mk, color=c, lw=2.0, ms=5.5, label=lab, solid_capstyle='round')

ax.axvline(13, color=MUTED, lw=1.0, ls=(0, (2, 3)))
ax.text(13.4, 2, 'victim replay power (13 dB)', fontsize=8.6, color=MUTED, rotation=90, va='bottom')
ax.set_ylim(-3, 100)
ax.set_xlabel('adversary TX gain (dB)', fontsize=11.5, color=INK)
ax.set_ylabel('expected fooling — victim device_5 overwritten (%)\nmean over 16 blind phases', fontsize=11, color=INK)
ax.set_title('A1 fooling vs adversary power — white-box CNN1D, over the air (ε = 0.01)',
             fontsize=12.5, color=INK, fontweight='bold', loc='left', pad=8)
ax.legend(loc='upper left', frameon=False, fontsize=9.5, labelcolor=INK2, ncol=2)
ax.grid(True, color=GRID, lw=0.8)
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)
for sp in ['left', 'bottom']:
    ax.spines[sp].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=10)

fig.suptitle('Expected over-the-air fooling rises gradually with adversary power — the graded curve the '
             'fixed-gain ε sweep hides',
             fontsize=11, fontweight='bold', color=INK, x=0.01, ha='left')
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig('docs/a1_gain_sweep.png', dpi=220, bbox_inches='tight', facecolor=SURF)
fig.savefig('docs/a1_gain_sweep.svg', bbox_inches='tight', facecolor=SURF)
print('wrote docs/a1_gain_sweep.png and .svg')
