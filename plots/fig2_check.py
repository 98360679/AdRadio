import numpy as np, pandas as pd, matplotlib.pyplot as plt

SRC = '/mnt/user-data/uploads/exp3_all_results_byexp.xlsx'
raw = pd.ExcelFile(SRC).parse('A2-transfer-raw', header=None)

def block(hdr, n, cols):
    df = raw.iloc[hdr+1:hdr+1+n, :len(cols)].copy()
    df.columns = cols
    return df.dropna(subset=[cols[0]]).astype(float).sort_values('eps')

neural = block(3, 40, ['eps','cvnn_wb','cvnn_to_cnn1d','cnn1d_to_cvnn','cnn1d_wb'])
feat   = block(47, 16, ['eps','cvnn','svm','rf'])

def eps50(e, y):
    e, y = np.asarray(e, float), np.asarray(y, float)
    i = np.argmax(y >= 0.5)
    if i == 0:
        return e[0]
    return e[i-1] + (e[i]-e[i-1]) * (0.5-y[i-1]) / (y[i]-y[i-1])

# transfer ratio tau = eps50(white-box) / eps50(transferred); tau = 1 is free transfer
t_cvnn  = eps50(neural.eps, neural.cvnn_wb)  / eps50(neural.eps, neural.cnn1d_to_cvnn)
t_cnn1d = eps50(neural.eps, neural.cnn1d_wb) / eps50(neural.eps, neural.cvnn_to_cnn1d)
r_svm = eps50(feat.eps, feat.svm) / eps50(feat.eps, feat.cvnn)
r_rf  = eps50(feat.eps, feat.rf)  / eps50(feat.eps, feat.cvnn)

print(f'eps50 CVNN white-box      {eps50(neural.eps, neural.cvnn_wb):.5f}')
print(f'eps50 CVNN transferred    {eps50(neural.eps, neural.cnn1d_to_cvnn):.5f}  tau={t_cvnn:.3f}')
print(f'eps50 CNN1D white-box     {eps50(neural.eps, neural.cnn1d_wb):.5f}')
print(f'eps50 CNN1D transferred   {eps50(neural.eps, neural.cvnn_to_cnn1d):.5f}  tau={t_cnn1d:.3f}')
print(f'eps50 CVNN (surrogate)    {eps50(feat.eps, feat.cvnn):.5f}')
print(f'eps50 SVM transferred     {eps50(feat.eps, feat.svm):.5f}  x{r_svm:.2f}')
print(f'eps50 RF  transferred     {eps50(feat.eps, feat.rf):.5f}  x{r_rf:.2f}')

plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
                     'legend.fontsize': 7, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'axes.grid': True, 'grid.alpha': 0.3})
fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.16, 2.7))

axL.plot(neural.eps, neural.cvnn_wb,       color='#1f77b4', ls='-',  lw=1.2,
         label='eval CVNN, white-box')
axL.plot(neural.eps, neural.cnn1d_to_cvnn, color='#1f77b4', ls='--', lw=1.2,
         label='eval CVNN, transferred')
axL.plot(neural.eps, neural.cnn1d_wb,      color='#d62728', ls='-',  lw=1.2,
         label='eval CNN1D, white-box')
axL.plot(neural.eps, neural.cvnn_to_cnn1d, color='#d62728', ls='--', lw=1.2,
         label='eval CNN1D, transferred')

axR.plot(feat.eps, feat.cvnn, color='#1f77b4', ls='-',  lw=1.2,
         label='CVNN surrogate, white-box')
axR.plot(feat.eps, feat.svm,  color='#2ca02c', ls='--', lw=1.2, label='SVM, transferred')
axR.plot(feat.eps, feat.rf,   color='#ff7f0e', ls='-.', lw=1.2, label='RF, transferred')

for a in (axL, axR):
    a.axhline(0.5, color='0.6', lw=0.8, ls=(0, (2, 3)))
    a.set_xlim(0, 0.08); a.set_ylim(0, 1.02)
    a.set_xlabel(r'$\varepsilon$')
    a.legend(loc='lower right', framealpha=0.95)
axL.set_ylabel('fooling rate')
for a, tag in ((axL, '(a)'), (axR, '(b)')):
    a.text(0.5, -0.26, tag, transform=a.transAxes, ha='center', fontsize=8)
fig.tight_layout()
fig.savefig('/mnt/user-data/outputs/varA2_transfer_digital.png', dpi=300)
fig.savefig('/mnt/user-data/outputs/varA2_transfer_digital.pdf')
