#!/usr/bin/env python3
"""B5 — L2 vs L-inf x {per-window, frame-level}: energy-eps50 (delta RMS at 50% frame-fool).
Under L2 the frame-level objective concentrates energy on the cheapest windows and wins big
(untargeted) -- unlocking the gain B3 could not get under L-inf. docs/b5_l2_vs_linf.csv -> docs/b5_l2_vs_linf.png/.svg"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, csv

METHODS = ['fgsm', 'bim', 'pgd_a0.05']
LAB = {'fgsm': 'FGSM', 'bim': 'BIM', 'pgd_a0.05': 'PGD α0.05'}
# 4 configs: norm x objective. high-contrast, colorblind-safe
CFG = [('linf', 'window', '#9aacba', 'L∞ · per-window'),
       ('linf', 'frame',  '#4c6b82', 'L∞ · frame-level'),
       ('l2',   'window', '#f2b56b', 'L2 · per-window'),
       ('l2',   'frame',  '#c0392b', 'L2 · frame-level')]
INK, INK2, MUTED, SURF, GRID = '#0b0b0b', '#52514e', '#8a8880', '#fcfcfb', '#e4e3de'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': SURF, 'axes.facecolor': SURF})

rows = list(csv.DictReader(open('docs/b5_l2_vs_linf.csv')))
def curve(meth, mode, norm, obj):
    rs = [r for r in rows if r['method']==meth and r['mode']==mode and r['norm']==norm and r['objective']==obj]
    xs = np.array([float(r['delta_rms']) for r in rs]); ys = np.array([float(r['frame_fool_pct']) for r in rs])
    o = np.argsort(xs); return xs[o], ys[o]
def e50(meth, mode, norm, obj):
    xs, ys = curve(meth, mode, norm, obj)
    for i in range(1, len(xs)):
        if ys[i-1] < 50 <= ys[i]:
            return float(xs[i-1]+(50-ys[i-1])*(xs[i]-xs[i-1])/(ys[i]-ys[i-1]))
    return float(xs[0]) if len(ys) and ys[0] >= 50 else np.nan

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for ax, mode in zip(axes, ['untargeted', 'targeted']):
    x = np.arange(len(METHODS)); w = 0.2
    for j, (norm, obj, c, lab) in enumerate(CFG):
        vals = [e50(m, mode, norm, obj) for m in METHODS]
        bars = ax.bar(x + (j-1.5)*w, vals, w, color=c, label=lab, edgecolor=SURF, linewidth=1.2, zorder=3)
        for xi, v in zip(x + (j-1.5)*w, vals):
            if not np.isnan(v): ax.text(xi, v+0.001, f'{v:.3f}', ha='center', va='bottom', fontsize=7.5, color=INK2, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([LAB[m] for m in METHODS], fontsize=10.5, color=INK)
    ax.set_title(f'{mode.capitalize()}', fontsize=12, color=INK, fontweight='bold', loc='left', pad=8)
    ax.set_ylabel('energy-eps₅₀  —  δ RMS at 50% frame-fool  (lower = more efficient)', fontsize=10.5, color=INK) if mode=='untargeted' else None
    ax.grid(True, axis='y', color=GRID, lw=0.8)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']: ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9.5)
    ax.set_ylim(0, 0.135)
axes[0].legend(loc='upper right', frameon=False, fontsize=9.2, labelcolor=INK2)
axes[0].annotate('L2 · frame-level wins\n(concentrates energy on\nthe cheapest ~7/13 windows)',
                 xy=(2+0.3, 0.032), xytext=(0.55, 0.098), fontsize=8.8, color='#c0392b', ha='left',
                 arrowprops=dict(arrowstyle='-|>', color='#c0392b', lw=1.2))
fig.suptitle('Experiment 3 B5 — L2 vs L∞ × objective: the frame-level gain B3 could not get under L∞ '
             'is unlocked by L2 (untargeted)',
             fontsize=12, fontweight='bold', color=INK, x=0.01, ha='left')
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig('docs/b5_l2_vs_linf.png', dpi=220, bbox_inches='tight', facecolor=SURF)
fig.savefig('docs/b5_l2_vs_linf.svg', bbox_inches='tight', facecolor=SURF)
print('wrote docs/b5_l2_vs_linf.png and .svg')
