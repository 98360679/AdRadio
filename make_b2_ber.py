#!/usr/bin/env python3
"""B2 matched vs universal (unmatched) — BER vs eps, 2-channel OTA sweep.
Matched (per-frame delta) rides the victim signal and drives BER up in the denial regime;
universal (one delta for all frames) keeps BER low but sheds frames instead. Same panels /
encoding as b2_matched_vs_unmatched (fooling). docs/b2_matched_vs_unmatched.csv -> docs/b2_ber.png/.svg"""
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
BITS_PER_FRAME = 500 * 8   # PDU_LENGTH(500 B) * 8 -> known bits compared per decoded frame
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': SURF, 'axes.facecolor': SURF})

rows = list(csv.DictReader(open('docs/b2_matched_vs_unmatched.csv')))


def curve(meth, mode, arm):
    """BER vs eps. A measured-zero point is drawn at its rule-of-three 95% upper bound
    3/N (N = frames_decoded * BITS_PER_FRAME), not an arbitrary floor — so a 'clean' point
    on few surviving frames reads as the weak bound it actually is."""
    rs = sorted([r for r in rows if r['method'] == meth and r['mode'] == mode and r['arm'] == arm],
                key=lambda z: float(z['eps']))
    eps, y = [], []
    for r in rs:
        b = float(r['ber']) if r['ber'] not in ('', 'None', None) else 0.0
        fd = float(r['frames_decoded']) if r['frames_decoded'] not in ('', 'None', None) else 0.0
        if b > 0:
            val = b
        elif fd > 0:
            val = 3.0 / (fd * BITS_PER_FRAME)   # rule of three: 0 errors in N bits
        else:
            val = float('nan')
        eps.append(float(r['eps'])); y.append(val)
    return eps, y


fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)


def panel(ax, mode, ylab, title):
    for meth in METHODS:
        c = MC[meth]
        for arm, ls, fill in [('mat', '-', c), ('uni', (0, (5, 2)), 'none')]:
            eps, ber = curve(meth, mode, arm)
            ax.plot(eps, ber, linestyle=ls, color=c, lw=2.2, marker='o', ms=5.5, mfc=fill, mec=c, mew=1.5,
                    solid_capstyle='round', zorder=3)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.axhline(1e-2, color=MUTED, lw=1.0, ls=(0, (2, 3)))
    ax.text(0.0095, 1.15e-2, 'BER 1e-2 (link damage)', fontsize=8.4, color=MUTED, va='bottom')
    ax.set_xlim(0.009, 1.1); ax.set_ylim(3e-6, 1.0)
    ax.set_xlabel('perturbation budget ε (log)', fontsize=11, color=INK)
    if ylab: ax.set_ylabel(ylab, fontsize=11, color=INK)
    ax.set_title(title, fontsize=12, color=INK, fontweight='bold', loc='left', pad=8)
    ax.grid(True, color=GRID, lw=0.8)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']: ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9.5)


panel(axL, 'unt', 'bit error rate (log)', 'Untargeted — overwrite')
panel(axR, 'tgt', '', 'Targeted — impersonate device_3')

from matplotlib.lines import Line2D
meth_handles = [Line2D([], [], color=MC[m], lw=2.6, label=LAB[m]) for m in METHODS]
arm_handles = [Line2D([], [], color=INK2, lw=2.2, ls='-', marker='o', mfc=INK2, mec=INK2, ms=6, label='matched (per-frame δ)'),
               Line2D([], [], color=INK2, lw=2.2, ls=(0, (5, 2)), marker='o', mfc='none', mec=INK2, ms=6, label='universal (one δ, all frames)')]
leg1 = axL.legend(handles=meth_handles, loc='upper left', frameon=False, fontsize=9.4, labelcolor=INK2, ncol=2, title='attack method')
leg1.get_title().set_color(INK2)
axR.legend(handles=arm_handles, loc='upper left', frameon=False, fontsize=9.4, labelcolor=INK2)

fig.suptitle('Experiment 3 B2 — link damage (BER) vs budget: matched δ climbs into denial past ε≈0.5; '
             'universal δ stays near-clean but sheds frames (0-error points shown at 3/N)',
             fontsize=12.5, fontweight='bold', color=INK, x=0.01, ha='left')
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig('docs/b2_ber.png', dpi=220, bbox_inches='tight', facecolor=SURF)
fig.savefig('docs/b2_ber.svg', bbox_inches='tight', facecolor=SURF)
print('wrote docs/b2_ber.png and .svg')
