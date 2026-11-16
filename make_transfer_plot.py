#!/usr/bin/env python3
"""Cross-model transferability (A2 + A3): a perturbation crafted on the CVNN transfers to other models.
Left: neural-to-neural (CNN1D <-> CVNN, white-box vs transferred, both directions).
Right: transfer onto the non-differentiable physics-feature SVM / Random Forest (crafted on the CVNN
surrogate; no white-box baseline exists for them). docs/transfer_fooling_vs_eps.png/.svg"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, csv

BLUE, ORANGE, GREEN, GOLD = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, MUTED, SURF, GRID = '#0b0b0b', '#52514e', '#8a8880', '#fcfcfb', '#e4e3de'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': SURF, 'axes.facecolor': SURF})

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.5, 6), sharey=True)

def e50(xs, ys):
    for i in range(1, len(xs)):
        if ys[i-1] < 0.5 <= ys[i]:
            return xs[i-1] + (0.5-ys[i-1])*(xs[i]-xs[i-1])/(ys[i]-ys[i-1])
    return None

def style_ax(ax, title, xr):
    ax.axhline(0.5, color=MUTED, lw=1.0, ls=(0, (2, 3)), zorder=1)
    ax.set_xlim(0, xr); ax.set_ylim(0, 1.02)
    ax.set_xlabel('perturbation budget  ε  (L∞)', fontsize=11, color=INK)
    ax.set_title(title, fontsize=12, color=INK, fontweight='bold', loc='left', pad=10)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']: ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9.5)

# ---- LEFT: neural-to-neural ----
z = np.load('docs/exp3_transfer_cnn1d_cvnn.npz'); g = z['grid']
for key, c, ls, lw, lab in [
        ('cvnn_on_cvnn',   BLUE,   '-',  2.4, 'eval CVNN  · white-box'),
        ('cnn1d_on_cvnn',  BLUE,   '--', 2.0, 'eval CVNN  · transferred (craft CNN1D)'),
        ('cnn1d_on_cnn1d', ORANGE, '-',  2.4, 'eval CNN1D · white-box'),
        ('cvnn_on_cnn1d',  ORANGE, '--', 2.0, 'eval CNN1D · transferred (craft CVNN)')]:
    axL.plot(g, z[key], color=c, linestyle=ls, linewidth=lw, label=lab, solid_capstyle='round', zorder=3)
style_ax(axL, 'Neural-to-neural (CNN1D and CVNN)', g[-1])
axL.set_ylabel('untargeted fooling rate', fontsize=11, color=INK)
axL.legend(loc='lower right', frameon=False, fontsize=9.0, labelcolor=INK2, handlelength=2.4)
axL.text(0.028, 0.28, f'transfer ratio τ = ε₅₀(white-box) / ε₅₀(transferred)\n'
         f'onto CVNN: τ = {z["tau_cvnn"]:.2f}   onto CNN1D: τ = {z["tau_cnn1d"]:.2f}\n'
         f'(τ ≈ 1: transfer nearly free)', fontsize=9.0, color=INK2, va='top', ha='left',
         bbox=dict(boxstyle='round,pad=0.5', fc='#f4f6f8', ec=GRID, lw=1.0))

# ---- RIGHT: transfer onto physics-feature SVM / RF ----
r = list(csv.DictReader(open('docs/exp3_a3_feature_model.csv')))
eps = [float(x['eps']) for x in r]
fc = {k: [float(x['fool_' + k]) for x in r] for k in ['cvnn', 'svm', 'rf']}
for key, c, ls, lw, lab in [
        ('cvnn', BLUE,  '-',  2.4, 'CVNN (surrogate, white-box)'),
        ('svm',  GREEN, '--', 2.2, 'SVM (physics feats) · transferred'),
        ('rf',   GOLD,  '--', 2.2, 'Random Forest (physics feats) · transferred')]:
    axR.plot(eps, fc[key], color=c, linestyle=ls, linewidth=lw, marker='o', ms=4, label=lab, solid_capstyle='round', zorder=3)
style_ax(axR, 'Transfer onto physics-feature models (SVM / RF)', eps[-1])
axR.legend(loc='lower right', frameon=False, fontsize=9.0, labelcolor=INK2, handlelength=2.4)
ec, es, er = e50(eps, fc['cvnn']), e50(eps, fc['svm']), e50(eps, fc['rf'])
axR.text(0.026, 0.28, f'CVNN-crafted δ transfers to the non-differentiable\nfeature models at a modest budget penalty:\n'
         f'SVM ≈ {es/ec:.1f}×   Random Forest ≈ {er/ec:.1f}×  (digital)\n'
         f'~1.3–1.4× over the air — not a black-box defense', fontsize=9.0, color=INK2, va='top', ha='left',
         bbox=dict(boxstyle='round,pad=0.5', fc='#f4f6f8', ec=GRID, lw=1.0))

fig.suptitle('Cross-model transferability: one CVNN-crafted perturbation fools other model families',
             fontsize=12.5, fontweight='bold', color=INK, x=0.01, ha='left')
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig('docs/transfer_fooling_vs_eps.png', dpi=220, bbox_inches='tight', facecolor=SURF)
fig.savefig('docs/transfer_fooling_vs_eps.svg', bbox_inches='tight', facecolor=SURF)
print('wrote docs/transfer_fooling_vs_eps.png ; SVM %.1fx RF %.1fx (digital)' % (es/ec, er/ec))
