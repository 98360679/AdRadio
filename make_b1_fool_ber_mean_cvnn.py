#!/usr/bin/env python3
"""B1 power sweep, fresh 0918 CVNN: fooling/overwrite (top) + BER/denial (bottom) vs PSR.
Zero-BER points at rule-of-three 3/N (N = frames_decoded*4000). -> docs/b1_fool_ber_cvnn.png"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv, os
import os
CAPTURES = os.environ.get('CAPTURES_ROOT', os.path.expanduser('~/captures'))
CAP=f'{CAPTURES}/attack_b1'
PSR_MEAS={20:-4.5,25:1.7,30:6.3,35:9.9,40:10.0}
def g2psr(g): return PSR_MEAS[g] if g in PSR_MEAS else -4.5+(g-20)
BITS_PER_FRAME=4000
METHODS=[('fgsm','#1f77b4','o','FGSM'),('bim','#d62728','s','BIM'),('mim','#2ca02c','^','MIM'),
         ('pgd_a0.01','#9467bd','D','PGD α0.01'),('pgd_a0.05','#ff7f0e','v','PGD α0.05'),('pgd_a0.10','#17becf','P','PGD α0.10')]
INK,INK2,MUTED,SURF,GRID='#0b0b0b','#52514e','#8a8880','#fcfcfb','#e4e3de'
TAG=os.environ.get('BER_TAG','ber'); OUT=os.environ.get('BER_OUT','docs/b1_fool_ber_cvnn')
YTOP=os.environ.get('BER_YTOP','Fooling Rate (%)')
plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':SURF,'axes.facecolor':SURF})
fig,(axT,axB)=plt.subplots(2,1,figsize=(9.5,8.4),sharex=True,gridspec_kw={'height_ratios':[1,1]})
for m,c,mk,lab in METHODS:
    p=f'{CAP}/{TAG}_{m}.csv'
    if not os.path.exists(p): continue
    r=list(csv.DictReader(open(p)))
    x=[g2psr(int(round(float(z['adv_gain_db'])))) for z in r]
    fool=[float(z['pct_mean']) for z in r]
    axT.plot(x,fool,marker=mk,color=c,lw=2.0,ms=5,label=lab,solid_capstyle='round')
    bx,by=[],[]
    for z,xi in zip(r,x):
        b=z['ber']; fd=z['frames_decoded']
        try: b=float(b); fd=float(fd)
        except: bx.append(xi); by.append(1.0); continue
        val=b if b>0 else (3.0/(fd*BITS_PER_FRAME) if fd>0 else None)
        bx.append(xi); by.append(val)
    # trailing None (0 frames decoded) = total denial -> pin at 1.0
    by=[(1.0 if v is None else v) for v in by]
    axB.plot(bx,by,marker=mk,color=c,lw=2.0,ms=5,label=lab,solid_capstyle='round')
for ax in (axT,axB):
    ax.axvline(-6,color=MUTED,lw=1.0,ls=(0,(2,3))); ax.set_xlim(-25,11); ax.set_xticks(range(-25,11,5))
    ax.grid(True,color=GRID,lw=0.8)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['left','bottom']: ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2,labelsize=9.5)
axT.set_ylim(-3,105); axT.set_ylabel(YTOP,fontsize=11,color=INK)
axT.set_title('Untargeted Attack — fooling (top) & denial (bottom), white-box CVNN OTA (fresh 0920)',fontsize=12.5,color=INK,fontweight='bold',loc='left',pad=8)
axT.legend(loc='upper left',frameon=False,fontsize=9,labelcolor=INK2,ncol=2,title='attack method'); axT.legend_.get_title().set_color(INK2)
axB.set_yscale('log'); axB.set_ylim(3e-6,1.0); axB.axhline(1e-2,color=MUTED,lw=1.0,ls=(0,(2,3)))
axB.text(-24.5,1.15e-2,'BER 1e-2 (link damage)',fontsize=8.2,color=MUTED,va='bottom')
axB.set_ylabel('bit error rate (log) — denial',fontsize=11,color=INK)
axB.set_xlabel('received PSR (dB) — adversary / victim  [0-error points at 3/N]',fontsize=11.5,color=INK)
plt.tight_layout()
fig.savefig(f'{OUT}.png',dpi=220,bbox_inches='tight',facecolor=SURF)
fig.savefig(f'{OUT}.svg',bbox_inches='tight',facecolor=SURF)
print(f'wrote {OUT}.png and .svg')
