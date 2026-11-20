#!/usr/bin/env python3
"""B2 matched vs unmatched (universal) frames, 16-point 2-channel eps sweep.
Universal delta (one waveform for all frames = a realizable adversary) needs a larger eps to reach
the same fooling/target as the frame-matched delta. docs/b2_matched_vs_unmatched.csv -> docs/b2_matched_vs_unmatched.png/.svg"""
import os as _os, sys as _sys
_R = _os.path.abspath(_os.path.dirname(__file__))
while _R != _os.path.dirname(_R) and not _os.path.isdir(_os.path.join(_R, ".git")):
    _R = _os.path.dirname(_R)
_sys.path.insert(0, _os.path.join(_R, "lib"))
_os.chdir(_R)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, csv

# high-contrast, well-separated categorical palette (one per method)
MC = {'fgsm': '#1f77b4', 'bim': '#d62728', 'mim': '#2ca02c',
      'pgd_a0.01': '#9467bd', 'pgd_a0.05': '#ff7f0e', 'pgd_a0.10': '#17becf'}
LAB = {'fgsm': 'FGSM', 'bim': 'BIM', 'mim': 'MIM',
       'pgd_a0.01': 'PGD α0.01', 'pgd_a0.05': 'PGD α0.05', 'pgd_a0.10': 'PGD α0.10'}
METHODS = ['fgsm', 'bim', 'mim', 'pgd_a0.01', 'pgd_a0.05', 'pgd_a0.10']
INK, INK2, MUTED, SURF, GRID = '#0b0b0b', '#52514e', '#8a8880', '#fcfcfb', '#e4e3de'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': SURF, 'axes.facecolor': SURF})

rows = list(csv.DictReader(open('docs/b2_matched_vs_unmatched.csv')))
def curve(meth, mode, arm):
    rs = sorted([r for r in rows if r['method'] == meth and r['mode'] == mode and r['arm'] == arm],
                key=lambda z: float(z['eps']))
    return [float(r['eps']) for r in rs], [float(r['metric_pct']) for r in rs]

def eps50(eps, met):
    eps, met = np.array(eps), np.array(met)
    for i in range(1, len(eps)):
        if met[i-1] < 50 <= met[i]:
            return eps[i-1] + (50-met[i-1])*(eps[i]-eps[i-1])/(met[i]-met[i-1])
    return None

fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)

def panel(ax, mode, ylab, title):
    for meth in METHODS:
        c = MC[meth]
        # matched = solid + filled circle ; universal = dashed + open circle
        for arm, ls, fill in [('mat', '-', c), ('uni', (0, (5, 2)), 'none')]:
            eps, met = curve(meth, mode, arm)
            ax.plot(eps, met, linestyle=ls, color=c, lw=2.2, marker='o', ms=5.5, mfc=fill, mec=c, mew=1.5,
                    solid_capstyle='round', zorder=3)
            e50 = eps50(eps, met)
            if e50: ax.plot([e50], [50], marker='|', color=c, ms=11, mew=2, zorder=4)
    ax.set_xscale('log'); ax.axhline(50, color=MUTED, lw=1.0, ls=(0, (2, 3)))
    ax.set_xlim(0.009, 1.1); ax.set_ylim(-4, 106)
    ax.set_xlabel('perturbation budget ε (log)', fontsize=11, color=INK)
    if ylab: ax.set_ylabel(ylab, fontsize=11, color=INK)
    ax.set_title(title, fontsize=12, color=INK, fontweight='bold', loc='left', pad=8)
    ax.grid(True, color=GRID, lw=0.8)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']: ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9.5)

panel(axL, 'unt', 'fool %  (victim identity lost)', 'Untargeted — overwrite')
panel(axR, 'tgt', 'target %  (lands on device_3)', 'Targeted — impersonate device_3')

# two legends: method (color) and arm (line style)
from matplotlib.lines import Line2D
meth_handles = [Line2D([], [], color=MC[m], lw=2.6, label=LAB[m]) for m in METHODS]
arm_handles = [Line2D([], [], color=INK2, lw=2.2, ls='-', marker='o', mfc=INK2, mec=INK2, ms=6, label='matched (per-frame δ)'),
               Line2D([], [], color=INK2, lw=2.2, ls=(0, (5, 2)), marker='o', mfc='none', mec=INK2, ms=6, label='universal (one δ, all frames)')]
leg1 = axL.legend(handles=meth_handles, loc='lower right', frameon=False, fontsize=9.4, labelcolor=INK2, ncol=2, title='attack method')
leg1.get_title().set_color(INK2)
axR.legend(handles=arm_handles, loc='lower right', frameon=False, fontsize=9.4, labelcolor=INK2)

fig.suptitle('Experiment 3 B2 — matched vs unmatched frames: every method still attacks universalized, '
             'at 1.6–10× the budget',
             fontsize=12.5, fontweight='bold', color=INK, x=0.01, ha='left')
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig('docs/b2_matched_vs_unmatched.png', dpi=220, bbox_inches='tight', facecolor=SURF)
fig.savefig('docs/b2_matched_vs_unmatched.svg', bbox_inches='tight', facecolor=SURF)
print('wrote docs/b2_matched_vs_unmatched.png and .svg')
