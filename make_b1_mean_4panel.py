import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, csv, numpy as np
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','DejaVu Serif'],
 'mathtext.fontset':'stix','axes.labelsize':10,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':7.5})
PSR_MEAS={20:-4.5,25:1.7,30:6.3,35:9.9,40:10.0}
def g2psr(g): return PSR_MEAS[int(g)] if int(g) in PSR_MEAS else -4.5+(int(g)-20)
MC={'fgsm':'#1f77b4','bim':'#d62728','mim':'#2ca02c','pgd_a0.01':'#9467bd','pgd_a0.05':'#ff7f0e','pgd_a0.10':'#17becf'}
LAB={'fgsm':'FGSM','bim':'BIM','mim':'MIM','pgd_a0.01':'PGD-0.01','pgd_a0.05':'PGD-0.05','pgd_a0.10':'PGD-0.10'}
METH=list(MC)
GRID,MUTED='#d9d9d9','#777777'
def load(prefix):
    D={}
    for m in METH:
        rows=sorted(csv.DictReader(open(f'docs/{prefix}{m}.csv')),key=lambda r:float(r['adv_gain_db']))
        psr=np.array([g2psr(float(r['adv_gain_db'])) for r in rows])
        rate=np.array([float(r['pct_mean']) for r in rows])
        ber=[]
        for r in rows:
            b=r['ber']; fd=float(r['frames_decoded']) if r['frames_decoded'] not in('','None') else 0
            if b in('','None',None) or fd==0: ber.append(np.nan)
            elif float(b)>0: ber.append(float(b))
            else: ber.append(3.0/(fd*4000))
        D[m]=(psr,rate,np.array(ber))
    return D
U=load('b1_cvnn_foolber_'); T=load('b1_cvnn_tgt_d2_foolber_')
fig,ax=plt.subplots(2,2,figsize=(7.16,5.2),sharex=True)
def rate_ax(a,D,title,ylab):
    for m in METH:
        p,r,_=D[m]; a.plot(p,r,'-o',color=MC[m],lw=1.3,ms=3.2,label=LAB[m])
    a.axhline(50,color=MUTED,lw=.8,ls=(0,(3,3))); a.set_ylim(-4,104); a.set_ylabel(ylab)
    a.set_title(title,fontsize=10,loc='left'); a.grid(True,color=GRID,lw=.5)
    for s in['top','right']: a.spines[s].set_visible(False)
def ber_ax(a,D,title):
    for m in METH:
        p,_,b=D[m]; a.plot(p,b,'-o',color=MC[m],lw=1.3,ms=3.2)
    a.set_yscale('log'); a.set_ylim(3e-6,1); a.axhline(1e-2,color=MUTED,lw=.8,ls=(0,(3,3)))
    a.set_ylabel('bit error rate'); a.set_xlabel('received PSR (dB)'); a.set_title(title,fontsize=10,loc='left')
    a.grid(True,color=GRID,lw=.5)
    for s in['top','right']: a.spines[s].set_visible(False)
rate_ax(ax[0,0],U,'(a) Untargeted fooling rate','fooling rate (%)')
rate_ax(ax[0,1],T,'(b) Targeted (device_2) target rate','target rate (%)')
ber_ax(ax[1,0],U,'(c) Untargeted BER'); ber_ax(ax[1,1],T,'(d) Targeted BER')
ax[0,1].set_xlabel(''); 
for a in ax.flat: a.set_xlim(-26,11); a.tick_params(length=3,colors='#333')
ax[0,0].legend(loc='upper left',frameon=False,ncol=2,handlelength=1.2,columnspacing=.8,labelspacing=.25)
fig.suptitle('B1 power sweep, eps=0.01, MEAN over 16 blind phases (0920 CVNN OTA)',fontsize=10.5,x=.01,ha='left')
fig.tight_layout(rect=[0,0,1,.97])
fig.savefig('docs/b1_mean_4panel.png',dpi=200,bbox_inches='tight')
print('wrote docs/b1_mean_4panel.png')
