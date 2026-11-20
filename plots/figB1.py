import os as _os
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repo root (parent of plots/)
OUT = _os.environ.get("FIG_OUT", _os.path.join(_R, "plots", "output"))
_os.makedirs(OUT, exist_ok=True)
import matplotlib.cbook as _cb, pandas as _pd
_c1d = _cb._check_1d
_cb._check_1d = lambda x: _c1d(x.to_numpy() if isinstance(x, (_pd.Series, _pd.Index)) else x)
import numpy as np, pandas as pd, matplotlib.pyplot as plt

SRC = _os.path.join(_R, 'docs', 'exp3_all_results_byexp.xlsx')
raw = pd.ExcelFile(SRC).parse([s for s in pd.ExcelFile(SRC).sheet_names if s.strip()=='B1-W'][0], header=None)
COLS = ['method','gain','psr','pct_mean','pct_max','ber','dec','phases']

def block(hdr, n):
    df = raw.iloc[hdr+1:hdr+1+n, :8].copy()
    df.columns = COLS
    df = df.dropna(subset=['method'])
    for c in COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

unt  = block(3, 78)     # untargeted
targ = block(85, 78)    # targeted, t = 1
BITS_PER_FRAME = 4000          # N = (frames decoded) x 4000 payload bits

styles = {
 'fgsm':      dict(color='#1f77b4', marker='o', ls='-',   label='FGSM'),
 'bim':       dict(color='#d62728', marker='s', ls='--',  label='BIM'),
 'mim':       dict(color='#2ca02c', marker='^', ls='-.',  label='MIM'),
 'pgd_a0.01': dict(color='#ff7f0e', marker='D', ls=':',   label=r'PGD $\alpha$=0.01'),
 'pgd_a0.05': dict(color='#9467bd', marker='v', ls=(0,(3,1,1,1)), label=r'PGD $\alpha$=0.05'),
 'pgd_a0.10': dict(color='#8c564b', marker='P', ls=(0,(5,1)),     label=r'PGD $\alpha$=0.10'),
}
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
                     'legend.fontsize': 7, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'axes.grid': True, 'grid.alpha': 0.3})

fig, ax = plt.subplots(2, 2, figsize=(7.16, 4.4), sharex=True)
for col, df in enumerate([unt, targ]):
    a, b = ax[0, col], ax[1, col]
    for m, g in df.groupby('method', sort=False):
        g = g.sort_values('psr'); s = styles[m]
        a.plot(g.psr, g.pct_mean, ms=3, lw=1.2, **s)
        ok = g.dropna(subset=['ber', 'dec'])
        bound = 3.0 / (BITS_PER_FRAME * ok.dec.clip(lower=1))
        obs = ok.ber.values.astype(float)
        y = np.where(obs > 0, obs, bound)
        cen = obs == 0
        b.plot(ok.psr, y, ms=3, lw=1.2,
               **{k: v for k, v in s.items() if k != 'label'})
        b.plot(ok.psr[cen], y[cen], ls='none', marker=s['marker'], ms=3,
               mfc='none', mec=s['color'])
    a.set_ylim(-3, 103)
    a.axhline(50, color='0.6', lw=0.8, ls=(0, (2, 3)))
    b.set_yscale('log'); b.set_ylim(2e-6, 1)
    b.axhline(1e-2, color='0.6', lw=0.8, ls=(0, (2, 3)))
    b.set_xlabel('PSR (dB)')
    b.set_xticks([-40, -30, -20, -10, 0, 10, 15])
    for c, tag in ((a, '(a)' if col == 0 else '(b)'),
                   (b, '(c)' if col == 0 else '(d)')):
        c.text(0.5, -0.30 if c is b else -0.12, tag,
               transform=c.transAxes, ha='center', fontsize=8)
ax[0, 0].set_ylabel('fooling rate (%)')
ax[0, 1].set_ylabel('target rate (%)')
ax[1, 0].set_ylabel('bit error rate')
ax[1, 1].set_ylabel('bit error rate')
ax[0, 0].legend(loc='lower right', ncol=2, framealpha=0.95,
                handlelength=1.8, labelspacing=0.25, columnspacing=1.0,
                borderpad=0.3, handletextpad=0.4)
fig.tight_layout()
fig.savefig(f'{OUT}/varB1_power_sweep.png', dpi=300)
fig.savefig(f'{OUT}/varB1_power_sweep.pdf')

for name, df in [('untargeted', unt), ('targeted', targ)]:
    print(f'--- {name}')
    for m, g in df.groupby('method', sort=False):
        g = g.sort_values('psr')
        i = np.argmax(g.pct_mean.values >= 50)
        g50 = np.nan if i == 0 else np.interp(
            50, g.pct_mean.values[i-1:i+1], g.psr.values[i-1:i+1])
        gber = g[g.ber > 1e-2].psr.min()
        print(f'  {m:10s} mean50={g50:6.1f} dB   BER>1e-2 at {gber:>6} dB')
