"""Publication figures. Inherited curves and new enlarged-family bounds are distinct."""
from pathlib import Path
from fractions import Fraction as F
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def data():
 rows=list(csv.DictReader((ROOT/'data/legacy_rate_bounds.csv').open()))
 for r in rows:r['q']=float(F(r['Q']));r['m']=int(r['live_memory_budget']);r['l']=float(r['lower']);r['u']=float(r['upper']) if r['upper'] else np.nan
 return rows

def save(fig,name):
 fig.savefig(ROOT/'figures'/f'{name}.pdf',bbox_inches='tight')
 fig.savefig(ROOT/'figures'/f'{name}.png',dpi=220,bbox_inches='tight');plt.close(fig)

def rates(model,overview=False):
 fig,ax=plt.subplots(figsize=(3.35,2.28));rows=data();mark=['o','s','^','v','D','P','x','+']
 budgets=range(1,9) if overview else (1,2,3,5,6,8)
 for m in budgets:
  a=sorted([r for r in rows if r['model']==model and r['m']==m],key=lambda r:r['q'])
  xx=np.array([r['q']*100 for r in a]);yy=np.array([r['l'] if r['l']>0 else np.nan for r in a])
  ax.plot(xx,yy,lw=.95,marker=mark[m-1],markersize=2.1,markevery=max(1,len(a)//12),label=fr'$m={m}$')
 prefix='werner' if model=='Werner' else 'bb84'
 for name,label,ls in [(f'{prefix}_linear_Du.csv','Du et al.','--')]+([('werner_linear_Sun_original.csv','Sun et al. (reconstructed)',':')] if model=='Werner' else []):
  d=np.genfromtxt(ROOT/'data/literature'/name,delimiter=',',names=True);good=d['rate']>0
  ax.plot(d['Q'][good]*100,d['rate'][good],ls=ls,lw=1.15,label=label)
 ax.set_xlabel(r'Error rate $Q$ (%)',fontsize=8);ax.set_ylabel('Secret bits per raw pair',fontsize=8)
 ax.tick_params(labelsize=7)
 if overview:ax.set_ylim(0,1.02);ax.set_xlim(0,33 if model=='Werner' else 25)
 else:ax.set_yscale('log');ax.set_ylim(1e-6,.32);ax.set_xlim(10,32 if model=='Werner' else 24.5)
 ax.legend(fontsize=5.8,ncol=2,loc='lower left' if not overview else 'upper right',handlelength=1.65,columnspacing=.7)
 ax.set_title(('(a) Known Werner state' if model=='Werner' else '(b) BB84-compatible family'),fontsize=8,loc='left')
 fig.tight_layout(pad=.35);save(fig,prefix+('_overview' if overview else '_rates'))

def memory(Q):
 rows={r['m']:r for r in data() if r['model']=='BB84_real' and abs(r['q']-Q)<1e-12}
 fig,ax=plt.subplots(figsize=(3.35,2.12));m=np.arange(1,9);low=np.array([rows[i]['l'] for i in m])
 extra=max(5*rows[3]['u'],2**-17,(1-Q)**160/2)
 upper=[max(rows[i]['u'],extra) for i in range(1,8)]
 ax.plot(m,low,marker='o',markersize=3.2,lw=1,label='Achievable rate')
 ax.plot(m[:-1],upper,marker='_',markersize=8,lw=.8,ls='--',label='Upper bound: enlarged family')
 ax.set_yscale('log');ax.set_xticks(m);ax.tick_params(labelsize=7)
 ax.set_xlabel(r'Local memory budget $m$',fontsize=8);ax.set_ylabel('Secret bits per raw pair',fontsize=8)
 ax.set_xlim(.8,8.2)
 if Q==.23:
  ax.set_ylim(3e-5,1.5e-3)
  ax.annotate(r'$N=32$',xy=(5,low[4]),xytext=(3.0,1.2e-3),fontsize=7,arrowprops=dict(arrowstyle='-',lw=.6))
  ax.annotate(r'$N=32$',xy=(6,low[5]),xytext=(6.9,1.2e-3),fontsize=7,arrowprops=dict(arrowstyle='-',lw=.6))
  ax.set_title('(a) BB84, Q = 23%',fontsize=8,loc='left')
 else:
  ax.set_ylim(2e-7,1.8e-4)
  ax.axhline(1e-5,lw=.75,ls=':');ax.text(1.1,1.15e-5,r'Target $10^{-5}$',fontsize=6.5)
  ax.set_title('(b) BB84, Q = 24%',fontsize=8,loc='left')
 ax.legend(fontsize=6.2,loc='lower right',handlelength=1.6)
 fig.tight_layout(pad=.35);save(fig,f'memory_Q{int(round(100*Q))}')

def cost_example():
 fig,ax=plt.subplots(figsize=(4.6,2.8));x=np.arange(3);c=[5/(692/3125),2975/173,11425/692]
 ax.bar(x,c,width=.58);ax.set_xticks(x,['Fixed block\n2 qubits','Early stopping\n2 qubits','Early stopping\n3 qubits'])
 ax.set_ylabel('Mean raw pairs per accepted output');ax.set_ylim(0,27)
 for i,v in enumerate(c):ax.text(i,v+.6,f'{v:.3f}',ha='center',fontsize=9)
 ax.set_title('Same five-input measurement and accepted state',fontsize=10)
 fig.tight_layout();save(fig,'same_measurement_cost')

def main():
 for model in ('Werner','BB84_real'):rates(model);rates(model,True)
 memory(.23);memory(.24);cost_example()
 print('Generated seven separate quantitative figures; protocol schematic is editable TikZ.')
if __name__=='__main__':main()
