#!/usr/bin/env python3
"""CORRECTED single-channel vs 2-channel OTA targeting: BOTH reach ~100% device_3; 2-channel is
run/phase-dependent (6-run spread). Supersedes the withdrawn 'only single-channel' figure."""
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

AQUA, ORANGE, GREY = '#1baf7a', '#eb6834', '#8a8880'
INK, INK2, MUTED, SURF, GRID = '#0b0b0b', '#52514e', '#8a8880', '#fcfcfb', '#e4e3de'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': SURF, 'axes.facecolor': SURF})

def load_rep(fn):
    rep = list(csv.DictReader(open(fn)))
    eps = [float(r['epsilon']) for r in rep]
    cols = [c for c in rep[0].keys() if c.startswith('target_d3_')]
    mat = np.array([[float(r[c]) if r[c] not in ('', '-') else np.nan for r in rep] for c in cols])
    return eps, np.nanmedian(mat, axis=0), np.nanmin(mat, axis=0), np.nanmax(mat, axis=0), mat

sc_eps, sc_med, sc_lo, sc_hi, sc_mat = load_rep('docs/b1_single_channel_targeted_reproducibility.csv')
tc_eps, tc_med, tc_lo, tc_hi, tc_mat = load_rep('docs/b1_2ch_targeted_reproducibility.csv')

fig, ax = plt.subplots(figsize=(10, 5.8))
# 2-channel band + median
ax.fill_between(tc_eps, tc_lo, tc_hi, color=ORANGE, alpha=0.12, zorder=1)
ax.plot(tc_eps, tc_med, 's--', color=ORANGE, lw=2.4, ms=6, zorder=4,
        label='2-channel — median of 6 runs (band = min–max)')
# single-channel band + median
ax.fill_between(sc_eps, sc_lo, sc_hi, color=AQUA, alpha=0.14, zorder=2)
ax.plot(sc_eps, sc_med, 'o-', color=AQUA, lw=2.6, ms=6, zorder=5,
        label='single-channel — median of 6 runs (band = min–max)')

ax.axhline(50, color=MUTED, lw=1.0, ls=(0, (2, 3)))
ax.annotate('one phase-unlucky 2-ch run → 14%\n(single-ch never collapses: min 85%)',
            xy=(0.8, 14), xytext=(0.42, 26), fontsize=9, color=INK2, ha='left',
            arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.2))
ax.set_xlabel('perturbation budget ε', fontsize=11, color=INK)
ax.set_ylabel('targeted success — victim lands on the CHOSEN device_3 (%)', fontsize=10.5, color=INK)
ax.set_title('BOTH channels can impersonate a chosen device OTA (≈100%);\n'
             'single-channel tighter (min 85%), 2-channel one phase-collapse (median 96%) — 6 runs each',
             fontsize=11.5, color=INK, fontweight='bold', loc='left', pad=10)
ax.set_ylim(-4, 105); ax.set_xlim(0.15, 1.02)
ax.legend(loc='lower right', frameon=False, fontsize=9.8, labelcolor=INK2)
ax.grid(True, color=GRID, lw=0.8)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
for sp in ['left', 'bottom']: ax.spines[sp].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9.5)

plt.tight_layout()
fig.savefig('docs/b1_sc_vs_2ch.png', dpi=220, bbox_inches='tight', facecolor=SURF)
fig.savefig('docs/b1_sc_vs_2ch.svg', bbox_inches='tight', facecolor=SURF)
print('wrote corrected docs/b1_sc_vs_2ch.png and .svg')
