#!/usr/bin/env python3
"""B1 OTA CORRECTED (2-channel, SCALE=1): a stealthy delta cannot overwrite the victim at any
reachable adversary power. Supersedes the withdrawn SCALE=13.06 figure."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BLUE, MUTED, SURF, GRID, INK, INK2 = '#2a78d6', '#8a8880', '#fcfcfb', '#e4e3de', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': SURF, 'axes.facecolor': SURF})

psr = [-6, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
fool = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.2]
dec = [107]*14

fig, ax = plt.subplots(figsize=(9.5, 5.4))
ax.plot(psr, fool, 'o-', color=BLUE, lw=2.4, ms=6, label='untargeted PGD-α0.10 ε=0.03 (correctly scaled)')
ax.axvline(22, color=MUTED, lw=1.0, ls=(0, (2, 3)))
ax.text(21.4, 55, 'adversary max\ngain (35 dB)', color=INK2, fontsize=8.5, ha='right')
ax.axvline(0, color=MUTED, lw=1.0, ls=(0, (2, 3)))
ax.text(0.5, 55, 'matched\npower', color=INK2, fontsize=8.5)
ax.set_xlabel('adversary power − victim power  (dB)', fontsize=11, color=INK)
ax.set_ylabel('fooling rate — victim lost  (%)', fontsize=11, color=INK)
ax.set_title('CORRECTED 2-channel B1: a stealthy δ never overwrites the victim (SCALE=1)',
             fontsize=12, color=INK, fontweight='bold', loc='left', pad=12)
ax.set_ylim(-3, 105); ax.set_xticks(psr)
ax.tick_params(axis='x', labelsize=8)
ax.text(6, 40, 'victim retained (device_5) at every power,\nlink clean (all 107 frames decode) —\neven at the radio\'s +22 dB ceiling, only 3% fool',
        color=INK2, fontsize=9.5, ha='left',
        bbox=dict(boxstyle='round,pad=0.5', fc='#f4f6f8', ec=GRID, lw=1.0))
ax.grid(True, color=GRID, lw=0.8)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
for sp in ['left', 'bottom']: ax.spines[sp].set_color(GRID)
ax.tick_params(colors=INK2)
ax.legend(loc='upper left', frameon=False, fontsize=9, labelcolor=INK2)

plt.tight_layout()
fig.savefig('docs/b1_ota_power_sweep.png', dpi=220, bbox_inches='tight', facecolor=SURF)
fig.savefig('docs/b1_ota_power_sweep.svg', bbox_inches='tight', facecolor=SURF)
print('wrote corrected docs/b1_ota_power_sweep.png and .svg')
