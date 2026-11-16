#!/usr/bin/env python3
"""OTA confirmation of A1-A4: one transmitted delta (crafted on CVNN), scored on all 5 models."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, csv

# reference-palette slots; learned models cool, physics/robust warm, freq green
CVNN_C, CNN1D_C, FREQ_C, SVM_C, RF_C = '#2a78d6', '#6b5b9a', '#1baf7a', '#eb6834', '#eda100'
INK, INK2, MUTED, SURF, GRID = '#0b0b0b', '#52514e', '#8a8880', '#fcfcfb', '#e4e3de'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': SURF, 'axes.facecolor': SURF})

rows = list(csv.DictReader(open('docs/exp3_ota_multimodel.csv')))
eps = [float(r['epsilon']) for r in rows]
def col(k): return [float(r[k]) for r in rows]

series = [
    ('fool_cvnn', CVNN_C, 'o-', 'CVNN (learned, IQ)'),
    ('fool_cnn1d', CNN1D_C, 's-', 'CNN1D (learned, real)'),
    ('fool_freqcvnn', FREQ_C, '^-', 'FreqCVNN (spectral)'),
    ('fool_svm', SVM_C, 'D--', 'SVM (physics feats)'),
    ('fool_rf', RF_C, 'v--', 'Random Forest (physics feats)'),
]
fig, ax = plt.subplots(figsize=(9.5, 5.8))
for k, c, mk, lab in series:
    ax.plot(eps, col(k), mk, color=c, lw=2.2, ms=5.5, label=lab, solid_capstyle='round')

ax.axhline(50, color=MUTED, lw=1.0, ls=(0, (2, 3)))
ax.axvspan(0.0, 0.3, color='#f2f4f6', zorder=0)
ax.text(0.15, 70, 'stealthy δ (ε≤0.2):\nfools nothing OTA', color=INK2, fontsize=9.5, ha='center')
ax.axvline(0.3, color=MUTED, lw=0.8, ls=(0, (1, 3)))
ax.set_xlim(0, 1.02)
ax.set_xlabel('perturbation budget ε — one OTA-transmitted δ, crafted on the CVNN', fontsize=10.5, color=INK)
ax.set_ylabel('OTA fooling rate — victim device_5 lost (%)', fontsize=11, color=INK)
ax.set_title('OTA confirmation of A1–A4: the transmitted δ fools every model class,\n'
             'in the same robustness order as digital (FreqCVNN hardest, CNN1D/CVNN easiest)',
             fontsize=11.5, color=INK, fontweight='bold', loc='left', pad=10)
ax.set_ylim(-4, 105)
ax.legend(loc='lower right', frameon=False, fontsize=9.4, labelcolor=INK2)
ax.grid(True, color=GRID, lw=0.8)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
for sp in ['left', 'bottom']: ax.spines[sp].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9.5)

plt.tight_layout()
fig.savefig('docs/ota_multimodel_A1A4.png', dpi=220, bbox_inches='tight', facecolor=SURF)
fig.savefig('docs/ota_multimodel_A1A4.svg', bbox_inches='tight', facecolor=SURF)
print('wrote docs/ota_multimodel_A1A4.png and .svg')
