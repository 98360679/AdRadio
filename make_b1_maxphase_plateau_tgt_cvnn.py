#!/usr/bin/env python3
"""B1 0920: MAX-over-phase fooling vs PSR (the professor's original metric), extended below -6
to show where the 100% plateau begins. -> docs/b1_maxphase_plateau_tgt_cvnn.png"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv, os
import os
CAPTURES = os.environ.get('CAPTURES_ROOT', os.path.expanduser('~/captures'))
CAP=f'{CAPTURES}/attack_b1'
PSR_MEAS={20:-4.5,25:1.7,30:6.3,35:9.9,40:10.0}
def g2psr(g): return PSR_MEAS[g] if g in PSR_MEAS else -4.5+(g-20)
METHODS=[('fgsm','#1f77b4','o','FGSM'),('bim','#d62728','s','BIM'),('mim','#2ca02c','^','MIM'),
         ('pgd_a0.01','#9467bd','D','PGD α0.01'),('pgd_a0.05','#ff7f0e','v','PGD α0.05'),('pgd_a0.10','#17becf','P','PGD α0.10')]
INK,INK2,MUTED,SURF,GRID='#0b0b0b','#52514e','#8a8880','#fcfcfb','#e4e3de'
plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':SURF,'axes.facecolor':SURF})
fig,ax=plt.subplots(figsize=(9.5,6))
for m,c,mk,lab in METHODS:
    r=list(csv.DictReader(open(f'{CAP}/tgt_{m}.csv')))
    x=[g2psr(int(round(float(z['adv_gain_db'])))) for z in r]
    y=[float(z['pct_max']) for z in r]
    ax.plot(x,y,marker=mk,color=c,lw=2.0,ms=5.5,label=lab,solid_capstyle='round')
# original professor's measured window -6..+6 (flat 100%)
ax.axvspan(-6,6,color='#b5544e',alpha=0.10,lw=0)
ax.text(0,58,"original sweep window\n(flat 100%, PSR −6..+6)",fontsize=8.6,color='#8a4a45',ha='center',va='center')
# plateau onset ~-18 dB
ax.axvline(-9.5,color=INK2,lw=1.4,ls=(0,(4,2)))
ax.text(-10.0,30,'plateau starts\n~ −9.5 dB PSR',fontsize=9.5,color=INK,fontweight='bold',ha='right',va='center')
ax.axhline(100,color=MUTED,lw=1.0,ls=(0,(2,3)))
ax.set_xlim(-25,11); ax.set_xticks(range(-25,11,5)); ax.set_ylim(-3,105)
ax.set_xlabel('received PSR (dB) — adversary / victim',fontsize=12,color=INK)
ax.set_ylabel('Target Rate (%)',fontsize=12,color=INK)
ax.set_title('Targeted Attack',
             fontsize=12.5,color=INK,fontweight='bold',loc='left',pad=8)
ax.legend(loc='lower right',frameon=False,fontsize=9.5,labelcolor=INK2,ncol=2,title='attack method'); ax.legend_.get_title().set_color(INK2)
ax.grid(True,color=GRID,lw=0.8)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
for sp in ['left','bottom']: ax.spines[sp].set_color(GRID)
ax.tick_params(colors=INK2,labelsize=10)
fig.suptitle('The original −6..+6 flat 100% was only the top of the plateau — it actually starts ~3.5 dB lower',
             fontsize=11,fontweight='bold',color=INK,x=0.01,ha='left')
plt.tight_layout(rect=[0,0,1,0.96])
fig.savefig('docs/b1_maxphase_plateau_tgt_cvnn.png',dpi=220,bbox_inches='tight',facecolor=SURF)
fig.savefig('docs/b1_maxphase_plateau_tgt_cvnn.svg',bbox_inches='tight',facecolor=SURF)
print('wrote docs/b1_maxphase_plateau_tgt_cvnn.png and .svg')
