#!/usr/bin/env python3
"""B1 per-method power sweep: attack success (%) and BER vs realized PSR, untargeted + targeted.
One threshold delta (eps=0.4) per method, swept in power (PSR -6..+8 dB), 2-channel matched 13/13.
Files: docs/b1_pwr_allmethods_{unt,tgt}.csv -> docs/b1_pwr_fool_ber.png/.svg"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, csv

# fixed per-method color + marker (identity never color-alone)
METHODS = [
    ('fgsm',       '#eb6834', 'o', 'FGSM'),
    ('bim',        '#1baf7a', 's', 'BIM'),
    ('mim',        '#2a78d6', '^', 'MIM'),
    ('pgd_a0.01',  '#6b5b9a', 'D', 'PGD α0.01'),
    ('pgd_a0.05',  '#eda100', 'v', 'PGD α0.05'),
    ('pgd_a0.10',  '#b5544e', 'P', 'PGD α0.10'),
]
INK, INK2, MUTED, SURF, GRID = '#0b0b0b', '#52514e', '#8a8880', '#fcfcfb', '#e4e3de'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': SURF, 'axes.facecolor': SURF})
BER_FLOOR = 5e-7   # BER==0 (clean) plotted at the axis floor


def load(fn, success_col):
    rows = list(csv.DictReader(open(fn)))
    by = {}
    for r in rows:
        by.setdefault(r['method'], []).append(r)
    data = {}
    for m, rs in by.items():
        rs = sorted(rs, key=lambda z: float(z['psr_db']))
        psr = [float(r['psr_db']) for r in rs]
        succ = [float(r[success_col]) for r in rs]
        ber = []
        for r in rs:
            b = r['ber']
            ber.append(np.nan if b in ('', None) else (BER_FLOOR if float(b) == 0 else float(b)))
        data[m] = (psr, succ, ber)
    return data

unt = load('docs/b1_pwr_allmethods_unt.csv', 'fool_pct')
tgt = load('docs/b1_pwr_allmethods_tgt.csv', 'target_pct_d3')

fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), sharex=True)
(axA, axB), (axC, axD) = axes

def plot_success(ax, data, title, ylab):
    for m, c, mk, lab in METHODS:
        psr, succ, _ = data[m]
        ax.plot(psr, succ, marker=mk, color=c, lw=2.2, ms=6, label=lab, solid_capstyle='round')
    ax.axhline(50, color=MUTED, lw=1.0, ls=(0, (2, 3)))
    ax.set_ylim(-4, 106); ax.set_ylabel(ylab, fontsize=11, color=INK)
    ax.set_title(title, fontsize=12, color=INK, fontweight='bold', loc='left', pad=8)

def plot_ber(ax, data, title):
    for m, c, mk, lab in METHODS:
        psr, _, ber = data[m]
        ax.plot(psr, ber, marker=mk, color=c, lw=2.2, ms=6, label=lab, solid_capstyle='round')
    ax.set_yscale('log'); ax.set_ylim(3e-7, 1.0)
    ax.axhline(1e-2, color=MUTED, lw=1.0, ls=(0, (2, 3)))
    ax.text(-5.8, 1.3e-2, 'BER 1e-2', fontsize=8.5, color=MUTED, va='bottom')
    ax.text(-5.8, BER_FLOOR*1.15, 'clean (BER 0)', fontsize=8.5, color=MUTED, va='bottom')
    ax.set_ylabel('bit error rate (log)', fontsize=11, color=INK)
    ax.set_xlabel('realized adversary PSR (dB)  —  adversary power relative to victim', fontsize=10.5, color=INK)
    ax.set_title(title, fontsize=12, color=INK, fontweight='bold', loc='left', pad=8)

plot_success(axA, unt, 'Untargeted — fingerprint overwritten (fool %)', 'fool %  (victim identity lost)')
plot_success(axB, tgt, 'Targeted — impersonate device_3 (target %)', 'target %  (lands on device_3)')
plot_ber(axC, unt, 'Untargeted — link denial (BER vs power)')
plot_ber(axD, tgt, 'Targeted — link denial (BER vs power)')

# annotate the one real fooling threshold (pgd_a0.01 targeted) and the overwrite-at-negative-PSR point
axB.annotate('only PGD α0.01 has a real threshold\n(40% @ -6 dB, victim retained)',
             xy=(-6, 40), xytext=(-3.4, 30), fontsize=8.8, color=INK2, ha='left',
             arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.1))
axA.text(-5.9, 90, 'overwrite is TOTAL from PSR -6 dB up:\nno power parity needed',
         fontsize=9, color=INK2, ha='left')

for ax in (axA, axB, axC, axD):
    ax.set_xlim(-6.6, 8.6); ax.set_xticks(range(-6, 9, 2))
    ax.grid(True, color=GRID, lw=0.8)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']: ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9.5)

axA.legend(loc='lower right', frameon=False, fontsize=9.2, labelcolor=INK2, ncol=2)
fig.suptitle('Experiment 3 B1 — per-method power sweep: identity attack saturates at low power; '
             'denial is the power-driven axis',
             fontsize=12.5, fontweight='bold', color=INK, x=0.01, ha='left')
plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig('docs/b1_pwr_fool_ber.png', dpi=220, bbox_inches='tight', facecolor=SURF)
fig.savefig('docs/b1_pwr_fool_ber.svg', bbox_inches='tight', facecolor=SURF)
print('wrote docs/b1_pwr_fool_ber.png and .svg')
