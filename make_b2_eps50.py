#!/usr/bin/env python3
"""B2 all-6-methods: eps_50 matched vs universal (unmatched) as a dumbbell plot.
The gap = the frame-matching realism penalty. docs/b2_matched_vs_unmatched.csv -> docs/b2_eps50.png/.svg"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, csv

METHODS = ['fgsm', 'bim', 'mim', 'pgd_a0.01', 'pgd_a0.05', 'pgd_a0.10']
LAB = {'fgsm': 'FGSM', 'bim': 'BIM', 'mim': 'MIM', 'pgd_a0.01': 'PGD α0.01',
       'pgd_a0.05': 'PGD α0.05', 'pgd_a0.10': 'PGD α0.10'}
MAT_C, UNI_C, LINE_C = '#2a78d6', '#eb6834', '#8a8880'
INK, INK2, MUTED, SURF, GRID = '#0b0b0b', '#52514e', '#8a8880', '#fcfcfb', '#e4e3de'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': SURF, 'axes.facecolor': SURF})

rows = list(csv.DictReader(open('docs/b2_matched_vs_unmatched.csv')))
def eps50(meth, mode, arm):
    rs = sorted([r for r in rows if r['method'] == meth and r['mode'] == mode and r['arm'] == arm],
                key=lambda z: float(z['eps']))
    eps = np.array([float(r['eps']) for r in rs]); met = np.array([float(r['metric_pct']) for r in rs])
    for i in range(1, len(eps)):
        if met[i-1] < 50 <= met[i]:
            return float(eps[i-1] + (50-met[i-1])*(eps[i]-eps[i-1])/(met[i]-met[i-1]))
    return float(eps[0]) if len(met) and met[0] >= 50 else None

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

def panel(ax, mode, title):
    ys = np.arange(len(METHODS))[::-1]
    for y, meth in zip(ys, METHODS):
        m, u = eps50(meth, mode, 'mat'), eps50(meth, mode, 'uni')
        if m is None or u is None: continue
        ax.plot([m, u], [y, y], '-', color=LINE_C, lw=2.4, zorder=1, solid_capstyle='round')
        ax.plot([m], [y], 'o', color=MAT_C, ms=11, zorder=3)
        ax.plot([u], [y], 'o', color=UNI_C, ms=11, zorder=3)
        ax.annotate(f'{u/m:.1f}×', xy=((m*u)**0.5, y), xytext=(0, 8), textcoords='offset points',
                    fontsize=9, color=INK2, ha='center', fontweight='bold')
    ax.set_yticks(ys); ax.set_yticklabels([LAB[m] for m in METHODS], fontsize=10.5, color=INK)
    ax.set_xscale('log'); ax.set_xlim(0.03, 1.2)
    ax.set_xlabel('eps₅₀  (budget to reach 50%, log)', fontsize=11, color=INK)
    ax.set_title(title, fontsize=12, color=INK, fontweight='bold', loc='left', pad=8)
    ax.grid(True, axis='x', color=GRID, lw=0.8)
    for sp in ['top', 'right', 'left']: ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9.5); ax.tick_params(axis='y', length=0)
    ax.set_ylim(-0.6, len(METHODS)-0.4)

panel(axL, 'unt', 'Untargeted — overwrite')
panel(axR, 'tgt', 'Targeted — impersonate device_3')
# legend
axL.plot([], [], 'o', color=MAT_C, ms=10, label='matched (per-frame δ)')
axL.plot([], [], 'o', color=UNI_C, ms=10, label='universal (one δ, all frames)')
axL.legend(loc='lower right', frameon=False, fontsize=9.6, labelcolor=INK2)

fig.suptitle('Experiment 3 B2 — matched vs unmatched eps₅₀: universalizing costs 1.6–10× budget; '
             'small-step PGD-α0.01 overfits per-frame the most',
             fontsize=12.3, fontweight='bold', color=INK, x=0.01, ha='left')
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig('docs/b2_eps50.png', dpi=220, bbox_inches='tight', facecolor=SURF)
fig.savefig('docs/b2_eps50.svg', bbox_inches='tight', facecolor=SURF)
print('wrote docs/b2_eps50.png and .svg')
