#!/usr/bin/env python3
"""B3 OTA: L-inf per-window vs frame-level energy-eps50 per method. Bars nearly equal = the digital
null holds over the air (frame-level gives no benefit under L-inf). docs/b3_ota_linf.csv -> docs/b3_ota_linf.png/.svg"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, csv

METHODS = ['fgsm', 'bim', 'mim', 'pgd_a0.01', 'pgd_a0.05', 'pgd_a0.10']
LAB = {'fgsm': 'FGSM', 'bim': 'BIM', 'mim': 'MIM', 'pgd_a0.01': 'PGD α0.01', 'pgd_a0.05': 'PGD α0.05', 'pgd_a0.10': 'PGD α0.10'}
WIN_C, FRM_C = '#9aacba', '#4c6b82'
INK, INK2, MUTED, SURF, GRID = '#0b0b0b', '#52514e', '#8a8880', '#fcfcfb', '#e4e3de'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': SURF, 'axes.facecolor': SURF})

rows = {r['method']: r for r in csv.DictReader(open('docs/b3_ota_linf.csv'))}
win = [float(rows[m]['linf_window_eps50']) for m in METHODS]
frm = [float(rows[m]['linf_frame_eps50']) for m in METHODS]

fig, ax = plt.subplots(figsize=(11.5, 5.8))
x = np.arange(len(METHODS)); w = 0.36
ax.bar(x - w/2, win, w, color=WIN_C, label='L∞ · per-window', edgecolor=SURF, linewidth=1.2, zorder=3)
ax.bar(x + w/2, frm, w, color=FRM_C, label='L∞ · frame-level', edgecolor=SURF, linewidth=1.2, zorder=3)
for xi, vw, vf in zip(x, win, frm):
    ax.text(xi - w/2, vw + 0.0003, f'{vw:.4f}', ha='center', va='bottom', fontsize=7.5, color=INK2, rotation=90)
    ax.text(xi + w/2, vf + 0.0003, f'{vf:.4f}', ha='center', va='bottom', fontsize=7.5, color=INK2, rotation=90)
    ax.text(xi, max(vw, vf) + 0.0016, f'{vf/vw:.2f}×', ha='center', va='bottom', fontsize=8.5, color=INK, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels([LAB[m] for m in METHODS], fontsize=10.5, color=INK)
ax.set_ylabel('OTA energy-eps₅₀ — tx-RMS at 50% fool  (lower = more efficient)', fontsize=10.5, color=INK)
ax.set_ylim(0, 0.027)
ax.legend(loc='upper left', frameon=False, fontsize=10, labelcolor=INK2)
ax.grid(True, axis='y', color=GRID, lw=0.8)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
for sp in ['left', 'bottom']: ax.spines[sp].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9.5)
ax.set_title('B3 OTA: under L∞, frame-level gives NO benefit over per-window (mean 1.13×)\n'
             'the digital null holds over the air — the frame advantage needs an L2 budget (see B5)',
             fontsize=11.5, color=INK, fontweight='bold', loc='left', pad=10)
plt.tight_layout()
fig.savefig('docs/b3_ota_linf.png', dpi=220, bbox_inches='tight', facecolor=SURF)
fig.savefig('docs/b3_ota_linf.svg', bbox_inches='tight', facecolor=SURF)
print('wrote docs/b3_ota_linf.png and .svg')
