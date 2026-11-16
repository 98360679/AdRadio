#!/usr/bin/env python3
"""B3 objective effect over the air: frame-level vs per-window energy-eps50 RATIO per method, under
L-inf (from b3_ota_linf) vs under L2 (from b5_ota_allmethods). Ratio < 1 = frame-level wins; > 1 = no
benefit. The frame objective helps ONLY under L2. -> docs/b3_objective_effect.png/.svg"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, csv

METHODS = ['fgsm', 'bim', 'mim', 'pgd_a0.01', 'pgd_a0.05', 'pgd_a0.10']
LAB = {'fgsm': 'FGSM', 'bim': 'BIM', 'mim': 'MIM', 'pgd_a0.01': 'PGD α0.01', 'pgd_a0.05': 'PGD α0.05', 'pgd_a0.10': 'PGD α0.10'}
LINF_C, L2_C = '#9aacba', '#c0392b'
INK, INK2, MUTED, SURF, GRID = '#0b0b0b', '#52514e', '#8a8880', '#fcfcfb', '#e4e3de'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': SURF, 'axes.facecolor': SURF})

# L-inf ratios from b3_ota_linf.csv (frame/window)
lr = {r['method']: float(r['linf_frame_eps50']) / float(r['linf_window_eps50'])
      for r in csv.DictReader(open('docs/b3_ota_linf.csv'))}
# L2 ratios from b5_ota_allmethods.csv (compute eps50 for l2_window / l2_frame)
b5 = list(csv.DictReader(open('docs/b5_ota_allmethods.csv')))
def e50(meth, cfg):
    rs = sorted([r for r in b5 if r['method'] == meth and r['config'] == cfg], key=lambda z: float(z['ota_tx_rms']))
    xs = [float(r['ota_tx_rms']) for r in rs]; ys = [float(r['fool_pct']) for r in rs]
    for i in range(1, len(xs)):
        if ys[i-1] < 50 <= ys[i]:
            return xs[i-1] + (50-ys[i-1])*(xs[i]-xs[i-1])/(ys[i]-ys[i-1])
    return np.nan
l2r = {m: e50(m, 'l2_frame') / e50(m, 'l2_window') for m in METHODS}

fig, ax = plt.subplots(figsize=(11.5, 6))
x = np.arange(len(METHODS)); w = 0.36
linf = [lr[m] for m in METHODS]; l2 = [l2r[m] for m in METHODS]
ax.bar(x - w/2, linf, w, color=LINF_C, label='under L∞ (per-sample budget)', edgecolor=SURF, linewidth=1.2, zorder=3)
ax.bar(x + w/2, l2, w, color=L2_C, label='under L2 (shared-energy budget)', edgecolor=SURF, linewidth=1.2, zorder=3)
for xi, a, b in zip(x, linf, l2):
    ax.text(xi - w/2, a + 0.01, f'{a:.2f}', ha='center', va='bottom', fontsize=8, color=INK2)
    ax.text(xi + w/2, b + 0.01, f'{b:.2f}', ha='center', va='bottom', fontsize=8, color=INK2)
ax.axhline(1.0, color=INK, lw=1.4, ls=(0, (4, 2)))
ax.text(len(METHODS)-0.5, 1.008, 'break-even (frame = window)', fontsize=8.5, color=INK2, va='bottom', ha='right')
ax.set_xticks(x); ax.set_xticklabels([LAB[m] for m in METHODS], fontsize=10.5, color=INK)
ax.set_ylabel('frame-level ÷ per-window  (OTA energy-eps₅₀ ratio)', fontsize=11, color=INK)
ax.set_ylim(0.6, 1.45)
ax.legend(loc='upper left', frameon=False, fontsize=10, labelcolor=INK2)
ax.grid(True, axis='y', color=GRID, lw=0.8)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
for sp in ['left', 'bottom']: ax.spines[sp].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9.5)
ax.set_title('B3 objective effect over the air: the frame-level objective helps ONLY under L2\n'
             'L∞ ratios sit at/above 1.0 (mean 1.13×, no benefit); L2 ratios sit at/below 1.0 (mean 0.90×, frame wins)',
             fontsize=11.5, color=INK, fontweight='bold', loc='left', pad=10)
plt.tight_layout()
fig.savefig('docs/b3_objective_effect.png', dpi=220, bbox_inches='tight', facecolor=SURF)
fig.savefig('docs/b3_objective_effect.svg', bbox_inches='tight', facecolor=SURF)
print('wrote docs/b3_objective_effect.png and .svg')
