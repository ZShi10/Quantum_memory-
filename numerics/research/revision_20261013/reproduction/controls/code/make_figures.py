"""Figures generated from supplied/frozen data and the new certificates.

No fitted values are inserted into the scientific data. Each chart is a separate
matplotlib figure; multi-panel assembly is performed only by LaTeX.
"""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def read(name):
    with (ROOT/'data'/name).open() as f:return list(csv.DictReader(f))
def save(fig,name):
    fig.tight_layout(pad=.45)
    fig.savefig(ROOT/'figures'/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(ROOT/'figures'/f'{name}.png',dpi=200,bbox_inches='tight')
    plt.close(fig)

def frontiers():
    data=read('archived_rate_endpoints.csv')
    for model,name,xlim in [('Werner','werner_rates',(.1,.322)),('BB84_real','bb84_rates',(.1,.245))]:
        fig,ax=plt.subplots(figsize=(3.48,2.72))
        for m in [1,2,3,4,5,6,8]:
            rs=sorted([r for r in data if r['model']==model and int(r['memory'])==m],key=lambda x:float(x['Q']))
            xx=np.array([float(r['Q']) for r in rs]);yy=np.array([float(r['lower']) for r in rs]);yy=np.where(yy>0,yy,np.nan)
            inds=[i for i,r in enumerate(rs) if r['source_status']=='certified_optimum']
            ax.plot(xx,yy,label=fr'$m={m}$',linewidth=1.1,marker='o',markersize=2.6,markevery=inds)
        r=read('literature/'+('werner_linear_Du.csv' if model=='Werner' else 'bb84_linear_Du.csv'))
        ax.plot([float(a['Q']) for a in r],[max(float(a['rate']),1e-30) for a in r],linestyle='--',linewidth=1,label='Du AD')
        if model=='Werner':
            r=read('literature/werner_linear_Sun_n8.csv')
            ax.plot([float(a['Q']) for a in r],[max(float(a['rate']),1e-30) for a in r],linestyle=':',marker='x',markersize=4,linewidth=1,label='4-input ED + common AD')
        ax.set_yscale('log');ax.set_xlim(*xlim);ax.set_ylim(1e-7,.5)
        ax.set_xlabel('Raw error rate $Q$',fontsize=8);ax.set_ylabel('Net secret bits / raw pair',fontsize=8)
        ax.set_title('Known Werner input' if model=='Werner' else 'BB84-compatible input; fixed policy',fontsize=8.5)
        ax.tick_params(labelsize=7);ax.legend(fontsize=6.5,ncol=2,loc='lower left',frameon=False,handlelength=1.9,columnspacing=.9)
        save(fig,name)

def fixed():
    r=read('fixed_block_interval_bounds.csv')
    fig,ax=plt.subplots(figsize=(3.45,2.10))
    for m,mark in [(2,'o'),(3,'s'),(4,'^')]:
        a=[z for z in r if int(z['memory'])==m]
        x=np.array([int(z['Nmax']) for z in a]);lo=np.array([float(z['lower']) for z in a]);hi=np.array([float(z['upper']) for z in a])
        ax.errorbar(x,1e4*(lo+hi)/2,yerr=1e4*(hi-lo)/2,marker=mark,markersize=4,linewidth=1.2,label=fr'$m={m}$')
    ax.set_xlim(.8,8.2);ax.set_ylim(-.05,2.8);ax.set_xticks(range(1,9))
    ax.set_xlabel('Maximum quantum inputs $N$',fontsize=8)
    ax.set_ylabel(r'Key rate $(10^{-4}$ bits/raw pair$)$',fontsize=8)
    ax.tick_params(labelsize=7);ax.legend(fontsize=7,loc='upper left',frameon=False)
    save(fig,'fixed_input_memory')

def budget_controls():
    rs=read('archived_rate_endpoints.csv')
    for Q,target in [(.23,4e-4),(.24,1e-5)]:
        r=[x for x in rs if x['model']=='BB84_real' and float(x['Q'])==Q]
        mm=np.array([int(x['memory']) for x in r]);lo=np.array([float(x['lower']) for x in r]);up=np.array([max(float(x['registered_upper']),5*float(x['block_upper'])) for x in r])
        fig,ax=plt.subplots(figsize=(4.9,2.7))
        ax.plot(mm,np.maximum(lo,1e-14),marker='o',label='Archived feasible lower')
        ax.plot(mm,up,marker='s',linestyle='--',label='Stopping-allowed upper relaxation')
        ax.axhline(target,linestyle=':',linewidth=1,label=f'Target {target:g}')
        ax.set_yscale('log');ax.set_ylim(1e-8,.003);ax.set_xticks(range(1,9));ax.set_xlabel('Occupied local qubits $m$');ax.set_ylabel('Secret bits / raw pair');ax.set_title(f'BB84 Q = {Q:.2f}')
        ax.legend(fontsize=8,loc='lower right');save(fig,'budget_'+str(Q).replace('.',''))

def literature():
    fig,ax=plt.subplots(figsize=(5.0,3.0))
    for name,label in [('werner_linear_Sun_author.csv','Supplied Sun author curve (original convention)'),('werner_linear_Sun_original.csv','Sun formula (original convention)'),('werner_linear_Sun_n8.csv','4-input ED with common AD/cost'),('werner_linear_Du.csv','Du AD')]:
        r=read('literature/'+name);ax.plot([float(a['Q']) for a in r],[max(float(a['rate']),1e-20) for a in r],label=label,linewidth=1)
    ax.set_yscale('log');ax.set_xlim(.1,.32);ax.set_ylim(1e-7,.5);ax.set_xlabel('Raw error rate $Q$');ax.set_ylabel('Rate under recorded source convention')
    ax.legend(fontsize=7.2,loc='lower left');save(fig,'literature_conventions')

def main():
    (ROOT/'figures').mkdir(exist_ok=True)
    frontiers();fixed();budget_controls();literature()
    print('Generated six independent vector charts and PNG previews from the frozen data.')
if __name__=='__main__':main()
