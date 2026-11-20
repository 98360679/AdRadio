import numpy as np, pandas as pd, matplotlib.pyplot as plt

SRC = '/mnt/user-data/uploads/exp3_all_results_byexp.xlsx'
raw = pd.ExcelFile(SRC).parse('A2-ML', header=None)

# "OTA: one delta scored across all 5 models" block
df = raw.iloc[48:64, :6].copy()
df.columns = ['eps', 'cvnn', 'cnn1d', 'freqcvnn', 'svm', 'rf']
df = df.dropna(subset=['eps']).astype(float).sort_values('eps')

series = [
 ('cvnn',     'CVNN (I/Q)',    '#1f77b4', 'o', '-'),
 ('cnn1d',    'CNN1D (real)',  '#9467bd', 's', '-'),
 ('freqcvnn', 'FreqCVNN',    '#2ca02c', '^', '-'),
 ('svm',      'SVM (physics)',    '#d62728', 'D', '--'),
 ('rf',       'RF (physics)',     '#ff7f0e', 'v', '--'),
]

def eps50(e, y):
    e, y = np.asarray(e, float), np.asarray(y, float)
    i = np.argmax(y >= 50)
    if i == 0:
        return np.nan
    return e[i-1] + (e[i]-e[i-1]) * (50-y[i-1]) / (y[i]-y[i-1])

print('OTA eps50 (one delta, scored on each model):')
for k, lab, *_ in series:
    print(f'  {lab:24s} {eps50(df.eps, df[k]):.3f}')
print('\nrate at eps = 0.2 :', {k: float(df[df.eps == 0.2][k].iloc[0]) for k, *_ in series})

plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
                     'legend.fontsize': 6, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'axes.grid': True, 'grid.alpha': 0.3})
fig, ax = plt.subplots(figsize=(3.5, 2.6))

for k, lab, c, mk, ls in series:
    ax.plot(df.eps, df[k], color=c, marker=mk, ls=ls, ms=2.5, lw=1.0, label=lab)
ax.axhline(50, color='0.6', lw=0.8, ls=(0, (2, 3)))
ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_ylim(-3, 103)
ax.set_xlabel(r'$\varepsilon$')
ax.set_ylabel('fooling rate (%)')
ax.legend(loc='lower right', framealpha=0.95,
          handlelength=1.8, labelspacing=0.3, borderpad=0.3, handletextpad=0.4)
fig.tight_layout()
fig.savefig('/mnt/user-data/outputs/varA2_transfer_ota.png', dpi=300)
fig.savefig('/mnt/user-data/outputs/varA2_transfer_ota.pdf')
