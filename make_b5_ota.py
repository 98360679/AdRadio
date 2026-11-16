#!/usr/bin/env python3
"""B5 OTA confirmation: fooling vs delta energy for L-inf/L2 x window/frame (pgd_a0.05, untargeted,
16-point 2ch rig). L2+frame-level reaches 50% fool at the lowest OTA energy -- the digital B5 ordering
(L2-frame < L2-window < L-inf) survives over the air. docs/b5_ota_energy.csv -> docs/b5_ota.png/.svg"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, csv

CFG = [('L∞·window', '#9aacba', 'o', '-'),
       ('L2·window', '#f2b56b', 's', '-'),
       ('L2·frame',  '#c0392b', 'D', '-')]
INK, INK2, MUTED, SURF, GRID = '#0b0b0b', '#52514e', '#8a8880', '#fcfcfb', '#e4e3de'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': SURF, 'axes.facecolor': SURF})

rows = list(csv.DictReader(open('docs/b5_ota_energy.csv')))
def curve(name):
    rs = sorted([r for r in rows if r['config'] == name], key=lambda z: float(z['ota_tx_rms']))
    return [float(r['ota_tx_rms']) for r in rs], [float(r['fool_pct']) for r in rs]
def e50(xs, ys):
    for i in range(1, len(xs)):
        if ys[i-1] < 50 <= ys[i]:
            return xs[i-1]+(50-ys[i-1])*(xs[i]-xs[i-1])/(ys[i]-ys[i-1])
    return None

fig, ax = plt.subplots(figsize=(10, 6))
for name, c, mk, ls in CFG:
    xs, ys = curve(name)
    ax.plot(xs, ys, ls, color=c, lw=2.4, marker=mk, ms=7, label=name, solid_capstyle='round', zorder=3)
    e = e50(xs, ys)
    if e:
        ax.plot([e], [50], marker='|', color=c, ms=16, mew=2.5, zorder=4)
        ax.annotate(f'{e:.4f}', xy=(e, 50), xytext=(0, -16 if name!='L2·frame' else 12), textcoords='offset points',
                    fontsize=8.5, color=c, ha='center', fontweight='bold')
ax.axhline(50, color=MUTED, lw=1.0, ls=(0, (2, 3)))
ax.set_xlabel('transmitted δ energy — tx-domain RMS  (∝ received δ energy)', fontsize=11, color=INK)
ax.set_ylabel('OTA fooling — victim device_5 overwritten (%)', fontsize=11, color=INK)
ax.set_title('B5 OTA: L2 + frame-level is the most energy-efficient attack over the air\n'
             '(digital ordering L2·frame < L2·window < L∞ survives; gaps compressed by adversary presence)',
             fontsize=11.5, color=INK, fontweight='bold', loc='left', pad=10)
ax.set_ylim(-4, 105); ax.set_xlim(0.004, 0.055)
ax.legend(loc='center right', frameon=False, fontsize=10.5, labelcolor=INK2, title='pgd_a0.05, untargeted')
ax.grid(True, color=GRID, lw=0.8)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
for sp in ['left', 'bottom']: ax.spines[sp].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9.5)
plt.tight_layout()
fig.savefig('docs/b5_ota.png', dpi=220, bbox_inches='tight', facecolor=SURF)
fig.savefig('docs/b5_ota.svg', bbox_inches='tight', facecolor=SURF)
print('wrote docs/b5_ota.png and .svg')
