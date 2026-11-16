#!/usr/bin/env python3
"""B1 0920 CVNN: untargeted vs targeted side-by-side. 2x2: columns = attack mode,
rows = fooling/target rate (top) + BER denial (bottom). -> docs/b1_combined_cvnn.png"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv, os
import os
CAPTURES = os.environ.get('CAPTURES_ROOT', os.path.expanduser('~/captures'))
CAP=f'{CAPTURES}/attack_b1'
PSR_MEAS={20:-4.5,25:1.7,30:6.3,35:9.9,40:10.0}
def g2psr(g): return PSR_MEAS[g] if g in PSR_MEAS else -4.5+(g-20)
BPF=4000
METHODS=[('fgsm','#1f77b4','o','FGSM'),('bim','#d62728','s','BIM'),('mim','#2ca02c','^','MIM'),
         ('pgd_a0.01','#9467bd','D','PGD α0.01'),('pgd_a0.05','#ff7f0e','v','PGD α0.05'),('pgd_a0.10','#17becf','P','PGD α0.10')]
INK,INK2,MUTED,SURF,GRID='#0b0b0b','#52514e','#8a8880','#fcfcfb','#e4e3de'
plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':SURF,'axes.facecolor':SURF})
COLS=[('ber','Untargeted Attack','Fooling Rate (%)'),
      ('tgt','Targeted Attack','Target Rate (%)')]
fig,axes=plt.subplots(2,2,figsize=(15,9.2),sharex='col')
for ci,(tag,coltitle,toplabel) in enumerate(COLS):
    axT,axB=axes[0][ci],axes[1][ci]
    for m,c,mk,lab in METHODS:
        p=f'{CAP}/{tag}_{m}.csv'
        if not os.path.exists(p): continue
        r=list(csv.DictReader(open(p)))
        x=[g2psr(int(round(float(z['adv_gain_db'])))) for z in r]
        axT.plot(x,[float(z['pct_max']) for z in r],marker=mk,color=c,lw=2.0,ms=5,label=lab,solid_capstyle='round')
        bx,by=[],[]
        for z,xi in zip(r,x):
            b=z['ber']; fd=z['frames_decoded']
            try: b=float(b); fd=float(fd)
            except: bx.append(xi); by.append(1.0); continue
            bx.append(xi); by.append(b if b>0 else (3.0/(fd*BPF) if fd>0 else 1.0))
        axB.plot(bx,by,marker=mk,color=c,lw=2.0,ms=5,solid_capstyle='round')
    for ax in (axT,axB):
        ax.set_xlim(-25,11); ax.set_xticks(range(-25,11,5)); ax.grid(True,color=GRID,lw=0.8)
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        for sp in ['left','bottom']: ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2,labelsize=9)
    axT.set_ylim(-3,105); axT.set_title(coltitle,fontsize=13,color=INK,fontweight='bold',pad=8)
    axB.set_yscale('log'); axB.set_ylim(3e-6,1.0); axB.axhline(1e-2,color=MUTED,lw=1.0,ls=(0,(2,3)))
    axB.set_xlabel('received PSR (dB) — adversary / victim',fontsize=11,color=INK)
    if ci==0:
        axT.set_ylabel(toplabel,fontsize=11.5,color=INK); axB.set_ylabel('bit error rate (log) — denial',fontsize=11.5,color=INK)
        axT.legend(loc='upper left',frameon=False,fontsize=8.6,labelcolor=INK2,ncol=2,title='attack method'); axT.legend_.get_title().set_color(INK2)
    else:
        axT.set_ylabel(toplabel,fontsize=11.5,color=INK)
fig.suptitle('B1 power sweep — Untargeted Attack vs Targeted Attack, white-box CVNN OTA (fresh 0920): '
             'fooling / target rate (top) & denial (bottom)',fontsize=12.5,fontweight='bold',color=INK,x=0.01,ha='left')
plt.tight_layout(rect=[0,0,1,0.97])
fig.savefig('docs/b1_combined_cvnn.png',dpi=200,bbox_inches='tight',facecolor=SURF)
fig.savefig('docs/b1_combined_cvnn.svg',bbox_inches='tight',facecolor=SURF)
print('wrote docs/b1_combined_cvnn.png and .svg')
