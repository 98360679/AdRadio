#!/usr/bin/env python3
"""Corrected Fig. 13 — frames rejected by the 802.11 receiver in Experiment 3 (two-channel OTA).

Regenerated from the SCALE=1 two-channel white-box CVNN data (docs/a1cvnn_<method>_ota_ber.csv,
107 frames, same source as A1/Fig. 14 and the corrected Fig. 12). Zero frames are rejected across
the entire eps 0.01-0.10 sweep window; rejections only begin near eps ~ 0.4-0.5, in the denial
regime. Replaces the figure whose '31-41 of 103 rejected' conflated the 103-frame digital BER run
with the 107-frame OTA rejection counts (which reach 31-41 only at eps 0.6-1.0).
docs/a1cvnn_*_ota_ber.csv -> docs/exp3_fig13_rejections_scale1.png/.pdf/.svg"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

MC = {'fgsm': '#1f77b4', 'bim': '#d62728', 'mim': '#2ca02c',
      'pgd_a0.01': '#9467bd', 'pgd_a0.05': '#ff7f0e', 'pgd_a0.10': '#17becf'}
LAB = {'fgsm': 'FGSM', 'bim': 'BIM', 'mim': 'MIM',
       'pgd_a0.01': 'PGD α0.01', 'pgd_a0.05': 'PGD α0.05', 'pgd_a0.10': 'PGD α0.10'}
METHODS = ['fgsm', 'bim', 'mim', 'pgd_a0.01', 'pgd_a0.05', 'pgd_a0.10']
INK, INK2, MUTED, SURF, GRID = '#0b0b0b', '#52514e', '#8a8880', '#fcfcfb', '#e4e3de'
FRAMES_SENT = 107
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': SURF, 'axes.facecolor': SURF})


def curve(meth):
    rows = sorted(csv.DictReader(open(f'docs/a1cvnn_{meth}_ota_ber.csv')),
                  key=lambda r: float(r['epsilon']))
    return [float(r['epsilon']) for r in rows], [int(r['frames_rejected']) for r in rows]


fig, ax = plt.subplots(figsize=(8.5, 5.8))

ax.axvspan(0.01, 0.10, color='#d9e6f5', alpha=0.55, zorder=0)
ax.text(0.031, 50, 'Experiment 3 sweep window\n(ε 0.01–0.10) — 0 frames rejected',
        fontsize=9.0, color='#3a5a86', ha='center', va='center')

for meth in METHODS:
    c = MC[meth]
    eps, rej = curve(meth)
    ax.plot(eps, rej, '-o', color=c, lw=2.0, ms=5.5, mfc=c, mec=c, zorder=3,
            label=LAB[meth], solid_capstyle='round')

ax.set_xscale('log')
ax.set_xlim(0.009, 1.1); ax.set_ylim(-3, FRAMES_SENT + 3)
ax.axhline(FRAMES_SENT, color=MUTED, lw=1.0, ls=(0, (2, 3)))
ax.text(0.0095, FRAMES_SENT - 1, '107 frames sent', fontsize=8.4, color=MUTED, va='top')
ax.axvline(0.4, color='#b5544e', lw=1.0, ls=(0, (2, 3)))
ax.text(0.4, 92, 'rejections begin\n~ ε 0.4', fontsize=8.8, color='#b5544e', ha='center', va='center')
ax.set_xlabel('perturbation budget ε (log)', fontsize=11.5, color=INK)
ax.set_ylabel('frames rejected (of 107)', fontsize=11.5, color=INK)
ax.set_title('Experiment 3 — frames rejected by the 802.11 receiver vs ε (SCALE = 1, white-box CVNN):\n'
             'no rejections through the ε 0.01–0.10 sweep; frames shed only in the denial regime (ε ≳ 0.4)',
             fontsize=11.5, color=INK, fontweight='bold', loc='left', pad=10)
ax.grid(True, color=GRID, lw=0.8)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
for sp in ['left', 'bottom']: ax.spines[sp].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9.5)
leg = ax.legend(loc='upper left', bbox_to_anchor=(0.012, 0.86), frameon=False, fontsize=9.4,
                labelcolor=INK2, ncol=2, title='attack method')
leg.get_title().set_color(INK2)

plt.tight_layout()
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(f'docs/exp3_fig13_rejections_scale1.{ext}', dpi=220, bbox_inches='tight', facecolor=SURF)
print('wrote docs/exp3_fig13_rejections_scale1.{png,pdf,svg}')
