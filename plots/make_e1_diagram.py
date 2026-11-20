#!/usr/bin/env python3
"""Experiment 1 core loop: perturbation generation -> scoring -> BER approximation (PNG + SVG)."""
import os as _os, sys as _sys
_R = _os.path.abspath(_os.path.dirname(__file__))
while _R != _os.path.dirname(_R) and not _os.path.isdir(_os.path.join(_R, ".git")):
    _R = _os.path.dirname(_R)
_sys.path.insert(0, _os.path.join(_R, "lib"))
_os.chdir(_R)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.family'] = 'DejaVu Sans'

fig, ax = plt.subplots(figsize=(15, 6.2))
ax.set_xlim(0, 15); ax.set_ylim(0, 6.4); ax.axis('off')
ax.text(0.2, 6.05, 'Experiment 1 - perturbation, scoring, and BER approximation',
        fontsize=16, fontweight='bold', color='#1f2d3d', ha='left', va='center')


def box(x, y, w, h, title, lines, fill, edge, tfs=12, lfs=9.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle='round,pad=0.02,rounding_size=0.13',
                 linewidth=1.8, edgecolor=edge, facecolor=fill))
    ax.text(x + w / 2, y + h - 0.34, title, fontsize=tfs, fontweight='bold',
            color=edge, ha='center', va='center')
    ax.plot([x + 0.22, x + w - 0.22], [y + h - 0.64] * 2, color=edge, lw=1.0, alpha=0.5)
    for j, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 1.02 - j * 0.36, ln, fontsize=lfs, color='#243447',
                ha='center', va='center')


def arrow(p0, p1, color='#55606e', lw=2.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=20, lw=lw, color=color))


# 1. perturbation generation (left, centered)
box(0.3, 2.05, 4.0, 2.5, '1  Perturbation generation',
    ['clean signal u  (enrolled window)', 'attack: FGSM . BIM . MIM . PGD',
     'budget: L-inf epsilon  (equiv. PSR)', 'targeted (i->j) or untargeted',
     'output: adversarial  u + delta'],
    '#f5dede', '#b5544e')

# fork point
fx = 4.55
arrow((4.3, 3.3), (fx, 3.3))
ax.plot([fx, fx], [1.95, 4.65], color='#55606e', lw=2.0)

# 2. scoring (top-right)
box(6.1, 3.55, 4.1, 2.25, '2  Scoring',
    ['u + delta  ->  CVNN fingerprint model', 'prediction = argmax f(u+delta)',
     'fooling rate / landing class', 'reachability radius R_ij'],
    '#e3ddef', '#6b5b9a')
arrow((fx, 4.65), (6.1, 4.65))

# 3. BER approximation (bottom-right)
box(6.1, 0.5, 4.1, 2.25, '3  BER approximation',
    ['u + delta  ->  software 802.11 decode', '(subprocess PHY, no radio)',
     'bit-compare vs known payload', 'approx BER = link stealthiness'],
    '#d7ece7', '#3f8f7d')
arrow((fx, 1.95), (6.1, 1.95))

# joint outcome (right, centered)
box(11.2, 2.05, 3.5, 2.5, 'Attack figure of merit',
    ['effective  AND  stealthy:', 'fools the model', '(high fooling / low R_ij)',
     'while BER stays low', 'sweep epsilon / PSR'],
    '#dcebd8', '#5a8f4e', tfs=11.5, lfs=9.2)
arrow((10.2, 4.65), (11.2, 3.55), color='#6b5b9a')
arrow((10.2, 1.6), (11.2, 3.0), color='#3f8f7d')

plt.tight_layout()
fig.savefig('docs/e1_perturbation_scoring_ber.png', dpi=220, bbox_inches='tight', facecolor='white')
fig.savefig('docs/e1_perturbation_scoring_ber.svg', bbox_inches='tight', facecolor='white')
print('wrote docs/e1_perturbation_scoring_ber.png and .svg')
