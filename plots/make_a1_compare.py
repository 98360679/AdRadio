#!/usr/bin/env python3
"""A1 CNN1D vs CVNN white-box OTA, side by side: fooling (top) and BER (bottom) vs eps, all 6 methods.
Both SAME SESSION (9/14): CNN1D = docs/a1_<m>_ota_ber.csv, CVNN = docs/a1cvnn_<m>_ota_ber.csv.
-> docs/a1_cnn1d_vs_cvnn.png/.svg"""
import os as _os, sys as _sys
_R = _os.path.abspath(_os.path.dirname(__file__))
while _R != _os.path.dirname(_R) and not _os.path.isdir(_os.path.join(_R, ".git")):
    _R = _os.path.dirname(_R)
_sys.path.insert(0, _os.path.join(_R, "lib"))
_os.chdir(_R)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, csv, os
import os
CAPTURES = os.environ.get('CAPTURES_ROOT', os.path.expanduser('~/captures'))

METHODS = [('fgsm', '#1f77b4', 'o', 'FGSM'), ('bim', '#d62728', 's', 'BIM'), ('mim', '#2ca02c', '^', 'MIM'),
           ('pgd_a0.01', '#9467bd', 'D', 'PGD α0.01'), ('pgd_a0.05', '#ff7f0e', 'v', 'PGD α0.05'),
           ('pgd_a0.10', '#17becf', 'P', 'PGD α0.10')]
INK, INK2, MUTED, SURF, GRID = '#0b0b0b', '#52514e', '#8a8880', '#fcfcfb', '#e4e3de'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': SURF, 'axes.facecolor': SURF})
BER_FLOOR = 5e-6
CVNN_DIR = f'{CAPTURES}/attack_a1_transfer'

def cnn1d(m):
    r = list(csv.DictReader(open(f'docs/a1_{m}_ota_ber.csv')))
    return ([float(x['epsilon']) for x in r], [float(x['fooling_pct']) for x in r],
            [max(float(x['ber']) if x['ber'] not in ('', None) else 0, BER_FLOOR) for x in r])

def cvnn(m):
    r = list(csv.DictReader(open(f'docs/a1cvnn_{m}_ota_ber.csv')))
    return ([float(x['epsilon']) for x in r], [float(x['fooling_pct']) for x in r],
            [max(float(x['ber']) if x['ber'] not in ('', None) else 0, BER_FLOOR) for x in r])

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
(fA, fB), (bA, bB) = axes

def foolax(ax, loader, title):
    for m, c, mk, lab in METHODS:
        e, f, _ = loader(m); ax.plot(e, f, marker=mk, color=c, lw=1.9, ms=4.5, label=lab)
    ax.axhline(50, color=MUTED, lw=1.0, ls=(0, (2, 3))); ax.set_xscale('log')
    ax.set_ylim(-4, 105); ax.set_xlim(0.009, 1.1); ax.set_ylabel('fooling rate (%)', fontsize=10.5, color=INK)
    ax.set_title(title, fontsize=12, color=INK, fontweight='bold', loc='left', pad=6)

def berax(ax, loader, title):
    for m, c, mk, lab in METHODS:
        e, _, b = loader(m); ax.plot(e, b, marker=mk, color=c, lw=1.9, ms=4.5, label=lab)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.009, 1.1); ax.set_ylim(3e-6, 1.0)
    ax.axhline(1e-2, color=MUTED, lw=1.0, ls=(0, (2, 3)))
    ax.set_ylabel('bit error rate (log)', fontsize=10.5, color=INK); ax.set_xlabel('perturbation budget ε (log)', fontsize=10.5, color=INK)
    ax.set_title(title, fontsize=12, color=INK, fontweight='bold', loc='left', pad=6)

foolax(fA, cnn1d, 'CNN1D — fooling (white-box, OTA)')
foolax(fB, cvnn, 'CVNN — fooling (white-box, OTA)')
berax(bA, cnn1d, 'CNN1D — BER')
berax(bB, cvnn, 'CVNN — BER')
fA.legend(loc='lower right', frameon=False, fontsize=8.4, labelcolor=INK2, ncol=2)
for ax in [fA, fB, bA, bB]:
    ax.grid(True, color=GRID, lw=0.8)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']: ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
fB.set_xlabel(''); # only bottom row gets x-labels visually; top handled by berax bottoms

fig.suptitle('A1 white-box over the air — CNN1D (left) vs CVNN (right): same overwrite-then-denial (same session) '
             'behaviour, confirming fragility is a task property',
             fontsize=12, fontweight='bold', color=INK, x=0.01, ha='left')
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig('docs/a1_cnn1d_vs_cvnn.png', dpi=220, bbox_inches='tight', facecolor=SURF)
fig.savefig('docs/a1_cnn1d_vs_cvnn.svg', bbox_inches='tight', facecolor=SURF)
print('wrote docs/a1_cnn1d_vs_cvnn.png and .svg')
