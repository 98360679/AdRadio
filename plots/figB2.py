import numpy as np, pandas as pd, matplotlib.pyplot as plt

SRC = '/mnt/user-data/uploads/exp3_all_results_byexp.xlsx'
raw = pd.ExcelFile(SRC).parse('B2-W', header=None)
df = raw.iloc[4:388, :8].copy()
df.columns = ['method','mode','arm','eps','pct','landed','dec','ber']
df = df.dropna(subset=['method'])
for c in ['eps','pct','dec','ber']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

styles = {
 'fgsm':      dict(color='#1f77b4', marker='o', label='FGSM'),
 'bim':       dict(color='#d62728', marker='s', label='BIM'),
 'mim':       dict(color='#2ca02c', marker='^', label='MIM'),
 'pgd_a0.01': dict(color='#ff7f0e', marker='D', label=r'PGD $\alpha$=0.01'),
 'pgd_a0.05': dict(color='#9467bd', marker='v', label=r'PGD $\alpha$=0.05'),
 'pgd_a0.10': dict(color='#8c564b', marker='P', label=r'PGD $\alpha$=0.10'),
}
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
                     'legend.fontsize': 7, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'axes.grid': True, 'grid.alpha': 0.3})

fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.95), sharey=True)
for col, mode in enumerate(['unt', 'tgt']):
    a = ax[col]
    for m, s in styles.items():
        for arm, ls, mf in [('mat', '-', s['color']), ('uni', '--', 'none')]:
            g = df[(df.method == m) & (df['mode'] == mode) &
                   (df.arm == arm)].sort_values('eps')
            a.plot(g.eps, g.pct, color=s['color'], marker=s['marker'], ls=ls,
                   ms=2.5, lw=1.1, mfc=mf, mec=s['color'],
                   label=s['label'] if arm == 'mat' else None)
    a.set_xscale('log'); a.set_ylim(-3, 103)
    a.axhline(50, color='0.6', lw=0.8, ls=(0, (2, 3)))
    a.set_xlabel(r'$\varepsilon$')
    a.set_xticks([0.01, 0.1, 1.0])
    a.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    a.text(0.5, -0.27, '(a)' if col == 0 else '(b)', transform=a.transAxes,
           ha='center', fontsize=8)
ax[0].set_ylabel('attack success rate (%)')

from matplotlib.lines import Line2D
h = [Line2D([], [], **{k: v for k, v in s.items() if k != 'label'}, ls='-',
            ms=2.5, lw=1.1) for s in styles.values()]
l = [s['label'] for s in styles.values()]
h += [Line2D([], [], color='0.25', ls='-', lw=1.1),
      Line2D([], [], color='0.25', ls='--', lw=1.1)]
l += ['matched', 'unmatched']
fig.tight_layout()
fig.subplots_adjust(bottom=0.33, top=0.97, left=0.085, right=0.985)
fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.5, 0.005), ncol=4,
           frameon=True, framealpha=1.0, edgecolor='0.6', fancybox=False,
           handlelength=1.8, columnspacing=1.6, handletextpad=0.5,
           borderpad=0.5)
fig.savefig('/mnt/user-data/outputs/varB2_matched_vs_universal.png', dpi=300)
fig.savefig('/mnt/user-data/outputs/varB2_matched_vs_universal.pdf')

def eps50(g):
    e, y = g.eps.values, g.pct.values
    i = np.argmax(y >= 50)
    return np.nan if i == 0 else e[i-1] + (e[i]-e[i-1])*(50-y[i-1])/(y[i]-y[i-1])

for mode, name in [('unt', 'untargeted'), ('tgt', 'targeted')]:
    print(f'--- {name}')
    for m in styles:
        a = eps50(df[(df.method==m)&(df['mode']==mode)&(df.arm=='mat')].sort_values('eps'))
        b = eps50(df[(df.method==m)&(df['mode']==mode)&(df.arm=='uni')].sort_values('eps'))
        print(f'  {m:10s} matched {a:.4f}   universal {b:.4f}   x{b/a:.1f}')
