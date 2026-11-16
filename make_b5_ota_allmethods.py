#!/usr/bin/env python3
"""B5 OTA all-methods: energy-eps50 (tx-RMS at 50% frame-fool) per method x {Linf-win, L2-win, L2-frame}.
L2 beats L-inf for every method; L2-frame is the most consistent winner. docs/b5_ota_allmethods.csv -> docs/b5_ota_allmethods.png/.svg"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, csv

METHODS = ['fgsm', 'bim', 'mim', 'pgd_a0.01', 'pgd_a0.05', 'pgd_a0.10']
LAB = {'fgsm': 'FGSM', 'bim': 'BIM', 'mim': 'MIM', 'pgd_a0.01': 'PGD α0.01', 'pgd_a0.05': 'PGD α0.05', 'pgd_a0.10': 'PGD α0.10'}
CFG = [('linf_window', '#9aacba', 'L∞ · per-window'),
       ('l2_window',   '#f2b56b', 'L2 · per-window'),
       ('l2_frame',    '#c0392b', 'L2 · frame-level')]
INK, INK2, MUTED, SURF, GRID = '#0b0b0b', '#52514e', '#8a8880', '#fcfcfb', '#e4e3de'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': SURF, 'axes.facecolor': SURF})

rows = list(csv.DictReader(open('docs/b5_ota_allmethods.csv')))
def e50(meth, cfg):
    rs = sorted([r for r in rows if r['method']==meth and r['config']==cfg], key=lambda z: float(z['ota_tx_rms']))
    xs = [float(r['ota_tx_rms']) for r in rs]; ys = [float(r['fool_pct']) for r in rs]
    for i in range(1, len(xs)):
        if ys[i-1] < 50 <= ys[i]:
            return xs[i-1]+(50-ys[i-1])*(xs[i]-xs[i-1])/(ys[i]-ys[i-1])
    return np.nan

fig, ax = plt.subplots(figsize=(12.5, 6))
x = np.arange(len(METHODS)); w = 0.26
for j, (cfg, c, lab) in enumerate(CFG):
    vals = [e50(m, cfg) for m in METHODS]
    ax.bar(x + (j-1)*w, vals, w, color=c, label=lab, edgecolor=SURF, linewidth=1.2, zorder=3)
    for xi, v in zip(x + (j-1)*w, vals):
        if not np.isnan(v): ax.text(xi, v+0.0002, f'{v:.4f}', ha='center', va='bottom', fontsize=7.3, color=INK2, rotation=90)
ax.set_xticks(x); ax.set_xticklabels([LAB[m] for m in METHODS], fontsize=10.5, color=INK)
ax.set_ylabel('OTA energy-eps₅₀ — tx-RMS at 50% fool  (lower = more efficient)', fontsize=10.5, color=INK)
ax.set_ylim(0, 0.023)
ax.legend(loc='upper left', frameon=False, fontsize=9.6, labelcolor=INK2)
ax.grid(True, axis='y', color=GRID, lw=0.8)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
for sp in ['left', 'bottom']: ax.spines[sp].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9.5)
ax.set_title('B5 OTA, all 6 methods: L2 beats L∞ for every method; L2·frame-level the most consistent winner\n'
             '(untargeted, 16-point 2ch rig — OTA gaps compressed vs digital by adversary presence)',
             fontsize=11.5, color=INK, fontweight='bold', loc='left', pad=10)
plt.tight_layout()
fig.savefig('docs/b5_ota_allmethods.png', dpi=220, bbox_inches='tight', facecolor=SURF)
fig.savefig('docs/b5_ota_allmethods.svg', bbox_inches='tight', facecolor=SURF)
print('wrote docs/b5_ota_allmethods.png and .svg')
