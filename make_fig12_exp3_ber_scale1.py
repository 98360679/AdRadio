#!/usr/bin/env python3
"""Corrected Fig. 12 — Experiment 3 (two-channel OTA) bit error rate vs perturbation radius ε.

Regenerated from the SCALE=1 two-channel white-box CVNN data (docs/a1cvnn_<method>_ota_ber.csv,
107 frames, the same source A1/Fig. 14 uses) instead of the mis-applied x8.76-rescaled digital
e1 curve. Shows the link stays clean through the fooling-saturation window (eps 0.08-0.10) and
denial (BER > 1e-2, frames rejected) only sets in near eps ~ 0.5.
docs/a1cvnn_*_ota_ber.csv -> docs/exp3_fig12_ber_scale1.png/.pdf/.svg"""
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
BITS_PER_FRAME = 500 * 8   # PDU 500 B -> 4000 known bits/frame (matches the paper's N)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': SURF, 'axes.facecolor': SURF})


def curve(meth):
    """eps, y, measured-flag. Zero-error points -> rule-of-three 3/N (N = decoded*4000);
    0 decoded -> nan (link gone)."""
    rows = sorted(csv.DictReader(open(f'docs/a1cvnn_{meth}_ota_ber.csv')),
                  key=lambda r: float(r['epsilon']))
    eps, y, meas = [], [], []
    for r in rows:
        e = float(r['epsilon']); b = float(r['ber']); fd = float(r['frames_decoded'])
        eps.append(e)
        if b > 0:
            y.append(b); meas.append(True)
        elif fd > 0:
            y.append(3.0 / (fd * BITS_PER_FRAME)); meas.append(False)
        else:
            y.append(float('nan')); meas.append(False)
    return eps, y, meas


fig, ax = plt.subplots(figsize=(8.5, 5.8))

# original Experiment-3 sweep window (eps 0.01-0.10) — entirely in the clean region
ax.axvspan(0.01, 0.10, color='#d9e6f5', alpha=0.55, zorder=0)
ax.text(0.031, 4.4e-6, 'Experiment 3 sweep window\n(ε 0.01–0.10) — link clean',
        fontsize=8.6, color='#3a5a86', ha='center', va='bottom')

for meth in METHODS:
    c = MC[meth]
    eps, y, meas = curve(meth)
    ax.plot(eps, y, '-', color=c, lw=2.0, zorder=3, label=LAB[meth], solid_capstyle='round')
    for e, v, m in zip(eps, y, meas):
        ax.plot([e], [v], marker='o', ms=5.5, mfc=(c if m else 'none'), mec=c, mew=1.5, zorder=4)

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(0.009, 1.1); ax.set_ylim(3e-6, 1.0)
ax.axhline(1e-2, color=MUTED, lw=1.0, ls=(0, (2, 3)))
ax.text(0.0095, 1.15e-2, 'BER 1e-2 (link damage)', fontsize=8.4, color=MUTED, va='bottom')
ax.axvline(0.5, color='#b5544e', lw=1.0, ls=(0, (2, 3)))
ax.text(0.5, 3.6e-6, 'BER crosses 1e-2\n~ ε 0.5', fontsize=8.6, color='#b5544e', ha='center', va='bottom')
ax.set_xlabel('perturbation budget ε (log)', fontsize=11.5, color=INK)
ax.set_ylabel('bit error rate (log)', fontsize=11.5, color=INK)
ax.set_title('Experiment 3 — two-channel OTA bit error rate vs ε (SCALE = 1, white-box CVNN):\n'
             'link stays clean through fooling saturation (ε ≈ 0.08–0.10); denial only past ε ≈ 0.5',
             fontsize=11.5, color=INK, fontweight='bold', loc='left', pad=10)
ax.grid(True, color=GRID, lw=0.8)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
for sp in ['left', 'bottom']: ax.spines[sp].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9.5)
leg = ax.legend(loc='upper left', bbox_to_anchor=(0.012, 0.985), frameon=False, fontsize=9.4, labelcolor=INK2, ncol=2, title='attack method')
leg.get_title().set_color(INK2)

plt.tight_layout()
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(f'docs/exp3_fig12_ber_scale1.{ext}', dpi=220, bbox_inches='tight', facecolor=SURF)
print('wrote docs/exp3_fig12_ber_scale1.{png,pdf,svg}')
