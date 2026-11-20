import os as _os
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repo root (parent of plots/)
OUT = _os.environ.get("FIG_OUT", _os.path.join(_R, "plots", "output"))
_os.makedirs(OUT, exist_ok=True)
import matplotlib.cbook as _cb, pandas as _pd
_c1d = _cb._check_1d
_cb._check_1d = lambda x: _c1d(x.to_numpy() if isinstance(x, (_pd.Series, _pd.Index)) else x)
import numpy as np, pandas as pd, matplotlib.pyplot as plt

SRC = _os.path.join(_R, 'docs', 'exp3_all_results_byexp.xlsx')
raw = pd.ExcelFile(SRC).parse('A1-ML', header=None)

def block(start):
    df = raw.iloc[start+1:start+97, :7].copy()
    df.columns = ['method','eps','fool','sent','dec','rej','ber']
    df = df.dropna(subset=['method'])
    for c in ['eps','fool','sent','dec','rej','ber']:
        df[c] = pd.to_numeric(df[c])
    return df

cnn, cvnn = block(595), block(695)
BITS_PER_FRAME = 2000          # from smallest observed BER 1/214000 over 107 frames

styles = {
 'fgsm':      dict(color='#1f77b4', marker='o', ls='-',   label='FGSM'),
 'bim':       dict(color='#d62728', marker='s', ls='--',  label='BIM'),
 'mim':       dict(color='#2ca02c', marker='^', ls='-.',  label='MIM'),
 'pgd_a0.01': dict(color='#ff7f0e', marker='D', ls=':',   label=r'PGD $\alpha$=0.01'),
 'pgd_a0.05': dict(color='#9467bd', marker='v', ls=(0,(3,1,1,1)), label=r'PGD $\alpha$=0.05'),
 'pgd_a0.10': dict(color='#8c564b', marker='P', ls=(0,(5,1)),     label=r'PGD $\alpha$=0.10'),
}
# Sized for \columnwidth: drawn at 3.5in and placed at 1:1, so point sizes
# here are the point sizes on the page.
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
                     'legend.fontsize': 6.5, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7,
                     'axes.grid': True, 'grid.alpha': 0.3})
xt = [0.01, 0.1, 1.0]

fig, ax = plt.subplots(2, 2, figsize=(3.5, 3.6), sharex=True)
for col, (df, name) in enumerate([(cnn, 'CNN1D'), (cvnn, 'CVNN')]):
    a, b = ax[0, col], ax[1, col]
    for m, g in df.groupby('method', sort=False):
        g = g.sort_values('eps'); s = styles[m]
        a.plot(g.eps, g.fool, ms=2.5, lw=1.0, **s)
        # censored BER: zero errors -> rule-of-three 95% upper bound
        bound = 2.9957 / (BITS_PER_FRAME * g.dec.clip(lower=1))
        obs = g.ber.values.astype(float)
        y = np.where(obs > 0, obs, bound)
        cen = obs == 0
        b.plot(g.eps, y, ms=2.5, lw=1.0, **{k: v for k, v in s.items() if k != 'label'})
        b.plot(g.eps[cen], y[cen], ls='none', marker=s['marker'], ms=2.5,
               mfc='none', mec=s['color'])
    a.set_title(f'{name} — fooling rate')
    a.set_ylim(-3, 105); a.set_xscale('log')
    a.axhline(50, color='0.6', lw=0.8, ls=(0, (2, 3)))
    b.set_xscale('log'); b.set_yscale('log'); b.set_ylim(5e-7, 1); b.set_yticks([1e-6,1e-4,1e-2,1e0])
    b.axhline(1e-2, color='0.6', lw=0.8, ls=(0, (2, 3)))
    b.set_title(f'{name} — BER')
    b.set_xlabel(r'$\varepsilon$')
    b.set_xticks(xt); b.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax[0, 0].set_ylabel('fooling (%)')
ax[1, 0].set_ylabel('BER')
h, l = ax[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.015), handlelength=2.2,
           columnspacing=1.2, handletextpad=0.5)
fig.tight_layout(rect=[0, 0.105, 1, 1])
fig.savefig(f'{OUT}/varA1_cnn1d_vs_cvnn_col.png', dpi=300)
fig.savefig(f'{OUT}/varA1_cnn1d_vs_cvnn_col.pdf')

