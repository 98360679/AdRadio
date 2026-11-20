#!/usr/bin/env python3
"""A3: attack-success vs epsilon — learned CVNN vs physics-explicit SVM / Random Forest."""
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

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'   # reference-palette slots 1,2,3
INK, INK2, MUTED, SURF, GRID = '#0b0b0b', '#52514e', '#8a8880', '#fcfcfb', '#e4e3de'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': SURF, 'axes.facecolor': SURF})

d = np.load('docs/exp3_a3_feature_model.npz')
rows = d['rows']            # eps, cvnn, svm, rf
g = rows[:, 0]

fig, ax = plt.subplots(figsize=(9, 5.6))
series = [(1, BLUE, 'CVNN (learned, IQ)'), (2, ORANGE, 'SVM (physics features)'),
          (3, AQUA, 'Random Forest (physics features)')]
for col, c, lab in series:
    ax.plot(g, rows[:, col], color=c, lw=2.4, label=lab, solid_capstyle='round', zorder=3)

# mark the budget where the CVNN saturates
ax.axvline(0.030, color=MUTED, lw=1.0, ls=(0, (2, 3)), zorder=1)
ax.text(0.031, 0.05, 'ε≈0.03: CVNN fully broken', color=MUTED, fontsize=8.5, ha='left', rotation=90, va='bottom')

ax.set_xlim(g[0], g[-1]); ax.set_ylim(0, 1.02)
ax.set_xlabel('perturbation budget  ε  (L∞, crafted on CVNN)', fontsize=11, color=INK)
ax.set_ylabel('attack success  (flip from correct clean prediction)', fontsize=11, color=INK)
ax.set_title('A3 — physics-explicit models are only partly more robust to the transferred attack',
             fontsize=12, color=INK, fontweight='bold', loc='left', pad=12)
ax.grid(True, color=GRID, lw=0.8, zorder=0)
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)
for sp in ['left', 'bottom']:
    ax.spines[sp].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9.5)
ax.legend(loc='lower right', frameon=False, fontsize=9.6, labelcolor=INK2, handlelength=2.0)

# gap annotation at eps=0.03
cvnn30 = rows[np.argmin(np.abs(g - 0.03)), 1]
svm30 = rows[np.argmin(np.abs(g - 0.03)), 2]
ax.annotate('', xy=(0.030, cvnn30), xytext=(0.030, svm30),
            arrowprops=dict(arrowstyle='<->', color=INK2, lw=1.2))
ax.text(0.0335, (cvnn30 + svm30) / 2, 'physics models resist\n~40% here (60% vs 100%)',
        fontsize=8.8, color=INK2, va='center', ha='left')

plt.tight_layout()
fig.savefig('docs/a3_feature_model_robustness.png', dpi=220, bbox_inches='tight', facecolor=SURF)
fig.savefig('docs/a3_feature_model_robustness.svg', bbox_inches='tight', facecolor=SURF)
print('wrote docs/a3_feature_model_robustness.png and .svg')
