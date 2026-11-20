#!/usr/bin/env python3
"""Per-source ensemble untargeted fooling-rate vs epsilon, CNN1D (A1)."""
import os as _os, sys as _sys
_R = _os.path.abspath(_os.path.dirname(__file__))
while _R != _os.path.dirname(_R) and not _os.path.isdir(_os.path.join(_R, ".git")):
    _R = _os.path.dirname(_R)
_sys.path.insert(0, _os.path.join(_R, "lib"))
_os.chdir(_R)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# reference-palette categorical slots 1-5, fixed order
CATS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']   # blue orange aqua yellow magenta
INK, INK2, MUTED, SURF, GRID = '#0b0b0b', '#52514e', '#8a8880', '#fcfcfb', '#e4e3de'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': SURF, 'axes.facecolor': SURF})

z = np.load('docs/reachability_shortfall_curves_cnn1d.npz')
g = z['grid']
SRC = [(1, 'Tx1 (device_1)'), (2, 'Tx2 (device_2)'), (3, 'Tx3 (device_3)'),
       (4, 'Tx4 (device_4)'), (5, 'Tx5 (device_5)')]


def eps50(frac):
    for k in range(len(g)):
        if frac[k] >= 0.5:
            if k == 0:
                return g[0]
            f0, f1 = frac[k - 1], frac[k]
            return g[k - 1] + (0.5 - f0) / (f1 - f0) * (g[k] - g[k - 1]) if f1 != f0 else g[k]
    return np.nan


fig, ax = plt.subplots(figsize=(9, 5.6))
for (s, lab), c in zip(SRC, CATS):
    ens = z[f'ens_{s}']
    ax.plot(g, ens, color=c, lw=2.4, label=lab, solid_capstyle='round', zorder=3)
    e = eps50(ens)
    if not np.isnan(e):
        ax.plot([e], [0.5], 'o', color=c, ms=6.5, mec=SURF, mew=1.4, zorder=4)

ax.axhline(0.5, color=MUTED, lw=1.0, ls=(0, (2, 3)), zorder=1)
ax.text(0.037, 0.53, '50% fooled', color=MUTED, fontsize=8.5, ha='left', va='bottom')

ax.set_xlim(g[0], g[-1]); ax.set_ylim(0, 1.02)
ax.set_xlabel('perturbation budget  ε  (L∞)', fontsize=11, color=INK)
ax.set_ylabel('untargeted fooling rate  (ensemble of 6 attacks)', fontsize=11, color=INK)
ax.set_title('CNN1D fragility by source: fooling rate vs budget',
             fontsize=12.5, color=INK, fontweight='bold', loc='left', pad=12)
ax.grid(True, color=GRID, lw=0.8, zorder=0)
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)
for sp in ['left', 'bottom']:
    ax.spines[sp].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9.5)
ax.legend(loc='center right', bbox_to_anchor=(1.0, 0.62), frameon=False, fontsize=9.4,
          labelcolor=INK2, handlelength=2.0, title='source device', title_fontsize=9.4)

# note: device_3 is the hard target
ax.annotate('Tx3 (device_3) is the hardest to fool —\nfurthest from its neighbours',
            xy=(0.0234, 0.5), xytext=(0.034, 0.20), fontsize=9.0, color=INK2, va='center',
            bbox=dict(boxstyle='round,pad=0.5', fc='#f4f6f8', ec=GRID, lw=1.0),
            arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.3))

plt.tight_layout()
fig.savefig('docs/cnn1d_fooling_vs_eps.png', dpi=220, bbox_inches='tight', facecolor=SURF)
fig.savefig('docs/cnn1d_fooling_vs_eps.svg', bbox_inches='tight', facecolor=SURF)
print('wrote docs/cnn1d_fooling_vs_eps.png and .svg')
print('eps_50 per source:', {lab.split()[0]: round(float(eps50(z[f"ens_{s}"])), 4) for s, lab in SRC})
