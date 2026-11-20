#!/usr/bin/env python3
"""Digital-vs-OTA contrast: the attack works digitally but fails over the air (hardware barrier)."""
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

BLUE, ORANGE, AQUA, RED = '#2a78d6', '#eb6834', '#1baf7a', '#b5544e'
INK, INK2, MUTED, SURF, GRID = '#0b0b0b', '#52514e', '#8a8880', '#fcfcfb', '#e4e3de'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': SURF, 'axes.facecolor': SURF})

rows = list(csv.DictReader(open('docs/reachability_cvnn.csv')))
eps_all = sorted(set(float(r['epsilon']) for r in rows))
def curve(mode, intent, attack, col):
    d = {}
    for r in rows:
        if r['attack'] == attack and r['mode'] == mode and (intent is None or r['intent'] == intent):
            d[float(r['epsilon'])] = float(r[col]) * 100
    return [d.get(e, np.nan) for e in eps_all]

show = [('fgsm', ORANGE, 'FGSM'), ('bim', AQUA, 'BIM'), ('pgd_a0.05', BLUE, 'PGD-α0.05')]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

# LEFT: targeted device_3
for atk, c, lab in show:
    ax1.plot(eps_all, curve('targeted', 'device_3', atk, 'device_3'), 'o-', color=c, lw=2.2, ms=4, label=f'DIGITAL {lab}')
ax1.axhline(0, color=RED, lw=3, alpha=0.9)
ax1.text(0.5, 4, 'OTA: 0% at every ε and every power (all methods, 2-ch & single-ch)',
         color=RED, fontsize=9.5, ha='center', fontweight='bold')
ax1.set_xscale('log'); ax1.set_xlabel('perturbation budget ε (log)', fontsize=11, color=INK)
ax1.set_ylabel('targeted success — lands on device_3 (%)', fontsize=11, color=INK)
ax1.set_title('Targeted impersonation: 100% digital, 0% OTA', fontsize=12, color=INK, fontweight='bold', loc='left')
ax1.set_ylim(-4, 105); ax1.legend(loc='center right', frameon=False, fontsize=9, labelcolor=INK2)

# RIGHT: untargeted fooling
for atk, c, lab in show:
    ax2.plot(eps_all, curve('untargeted', None, atk, 'fool'), 'o-', color=c, lw=2.2, ms=4, label=f'DIGITAL {lab}')
ax2.axhline(3, color=RED, lw=3, alpha=0.9)
ax2.text(0.5, 8, 'OTA 2-ch (correctly scaled): ≤3% even at the +22 dB power ceiling',
         color=RED, fontsize=9.5, ha='center', fontweight='bold')
ax2.set_xscale('log'); ax2.set_xlabel('perturbation budget ε (log)', fontsize=11, color=INK)
ax2.set_ylabel('untargeted fooling — victim lost (%)', fontsize=11, color=INK)
ax2.set_title('Evasion: 100% digital, ≈0% OTA', fontsize=12, color=INK, fontweight='bold', loc='left')
ax2.set_ylim(-4, 105); ax2.legend(loc='center right', frameon=False, fontsize=9, labelcolor=INK2)

for ax in (ax1, ax2):
    ax.grid(True, color=GRID, lw=0.8)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']: ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)

fig.suptitle('Experiment 3 — the hardware barrier: adversarial attacks that work digitally fail over the air',
             fontsize=12.5, fontweight='bold', color=INK, x=0.01, ha='left')
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig('docs/digital_vs_ota.png', dpi=220, bbox_inches='tight', facecolor=SURF)
fig.savefig('docs/digital_vs_ota.svg', bbox_inches='tight', facecolor=SURF)
print('wrote docs/digital_vs_ota.png and .svg')
