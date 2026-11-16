#!/usr/bin/env python3
"""A4: per-source eps_50 (attack budget to 50% fool), time-CVNN vs frequency-CVNN."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BLUE, AQUA = '#2a78d6', '#1baf7a'                 # time vs frequency
INK, INK2, MUTED, SURF, GRID = '#0b0b0b', '#52514e', '#8a8880', '#fcfcfb', '#e4e3de'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': SURF, 'axes.facecolor': SURF})

src = ['Tx1\n(d2)', 'Tx2\n(d3)', 'Tx3\n(d4)', 'Tx4\n(d5)', 'Tx5\n(d6)\nvictim']
e_time = [0.00610, 0.01234, 0.02316, 0.00471, 0.01700]
e_freq = [0.00615, 0.00609, 0.01986, 0.00668, 0.03268]

x = np.arange(5); w = 0.38
fig, ax = plt.subplots(figsize=(9, 5.4))
b1 = ax.bar(x - w/2, e_time, w, color=BLUE, label='time-domain CVNN', zorder=3)
b2 = ax.bar(x + w/2, e_freq, w, color=AQUA, label='frequency-domain CVNN', zorder=3)
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.0006, f'{b.get_height():.3f}',
            ha='center', va='bottom', fontsize=8.2, color=INK2)

ax.set_xticks(x); ax.set_xticklabels(src, fontsize=9.2, color=INK2)
ax.set_ylabel('ε₅₀  (L∞ budget to fool 50% of the source)', fontsize=11, color=INK)
ax.set_title('A4 — the spectral representation redistributes robustness (victim hardens ~2×)',
             fontsize=12, color=INK, fontweight='bold', loc='left', pad=12)
ax.set_ylim(0, max(e_freq) * 1.18)
ax.grid(True, axis='y', color=GRID, lw=0.8, zorder=0)
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)
for sp in ['left', 'bottom']:
    ax.spines[sp].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9.5)
ax.legend(loc='upper left', frameon=False, fontsize=9.6, labelcolor=INK2)

# victim callout
ax.annotate('victim Tx5 (device_5):\n0.017 → 0.033 (≈2× harder)',
            xy=(4 + w/2, 0.03268), xytext=(2.55, 0.028), fontsize=9.0, color=INK2, va='center',
            bbox=dict(boxstyle='round,pad=0.5', fc='#f4f6f8', ec=GRID, lw=1.0),
            arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.3))

plt.tight_layout()
fig.savefig('docs/a4_time_vs_freq_eps50.png', dpi=220, bbox_inches='tight', facecolor=SURF)
fig.savefig('docs/a4_time_vs_freq_eps50.svg', bbox_inches='tight', facecolor=SURF)
print('wrote docs/a4_time_vs_freq_eps50.png and .svg')
