"""Regenerate vector manuscript figures from traceable numerical tables.

No spline or synthetic optimum is introduced. Global connecting segments
are guides through the archived achieved coordinates; certificate windows
are plotted directly from their exact-endpoint data.
"""
from pathlib import Path
from fractions import Fraction as F
import csv
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator,LogLocator,LogFormatterSciNotation

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'paper/figures';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
    'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'lines.linewidth':1.3,
    'savefig.facecolor':'white'})
COL=['#707070','#466F9D','#2F8EAA','#2E9375','#91A241','#CC9B39','#CE6A35','#A53D55']
records=[]
def table(path):
    with (ROOT/path).open(encoding='utf-8') as f:return list(csv.DictReader(f))
def save(fig,name,sources):
    for ext in ('pdf','png'):
        path=OUT/(name+'.'+ext);fig.savefig(path,dpi=200,bbox_inches='tight',pad_inches=.07)
    records.append(dict(figure=name,source_files=sources,vector_pdf_sha256=hashlib.sha256((OUT/(name+'.pdf')).read_bytes()).hexdigest()))
    plt.close(fig)
def box(ax,x,y,w,h,label,color='#edf2f7',fs=9):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=0.06',
                 facecolor=color,edgecolor='#566779',linewidth=.8))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fs)
def arrow(ax,a,b,color='#566779'):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1,color=color))

def protocol():
    fig,ax=plt.subplots(figsize=(7.2,3.65));ax.set_xlim(0,10);ax.set_ylim(0,5);ax.axis('off')
    labels=['Raw Bell\npairs',r'$N$-input'+'\nClifford ED','Measure\n'+r'$N-1$ check'+'\nqubits', 'Retained\nBell pair','Measure\nkey qubit','Classical\nAD','Secret\nkey']
    xs=[.03,1.43,2.83,4.23,5.63,7.03,8.43]
    for i,(x,label) in enumerate(zip(xs,labels)):
        box(ax,x,3.66,1.23,.79,label,'#eaf2f8' if i<5 else '#eaf3ea',8.5)
        if i<6:arrow(ax,(x+1.24,4.055),(xs[i+1]-.025,4.055))
    ax.text(4.9,4.76,'Logical protocol',ha='center',fontsize=11,fontweight='bold')
    ax.text(4.95,3.32,r'$N$: successful ancestry     $m$: peak local qubits     $C$: mean raw consumption',ha='center',fontsize=9)
    ax.plot([0,9.75],[3.05,3.05],color='#ccd3da',lw=.8)
    ax.text(.1,2.70,'Release and reuse',fontweight='bold',fontsize=10)
    ax.text(5.2,2.70,'Retain intermediate success',fontweight='bold',fontsize=10)
    ax.text(.05,2.15,'Slot 1',fontsize=8);ax.text(.05,1.35,'Slot 2',fontsize=8)
    box(ax,.85,1.9,3.55,.54,'Unresolved quantum state','#dbeaf5',8)
    for x,lab in [( .85,'Load'),(2.2,'Read'),(3.55,'Reload')]:box(ax,x,1.08,.8,.57,lab,'#f0f3f6',8)
    arrow(ax,(1.68,1.365),(2.15,1.365));arrow(ax,(3.03,1.365),(3.50,1.365))
    ax.text(2.7,.72,'Save outcomes classically; reuse the slot.',ha='center',fontsize=8)
    box(ax,5.2,1.9,2.00,.54,'Hold successful\nchild L','#dbeaf5',8)
    box(ax,5.2,1.08,2.00,.57,'Retry failed\nchild R','#f0f3f6',8)
    box(ax,8.13,1.43,1.56,.61,'Combine\nL and R','#eaf3ea',8)
    arrow(ax,(7.22,2.17),(8.10,1.87));arrow(ax,(7.22,1.365),(8.10,1.60))
    ax.text(7.4,.72,'Avoid repeating the successful child.',ha='center',fontsize=8)
    ax.text(4.9,.12,r'Net yield: $R=r_{\rm cl}/C$ secret bits per raw Bell pair',ha='center',fontsize=10)
    save(fig,'figure1_protocol',['paper/sections/s1_model.tex','paper/sections/s3_production.tex'])

def landscape():
    rows=table('data/global_rate_landscape.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.4),sharey=True)
    for ax,model,title,xmax in zip(axes,['Werner','BB84_real'],['(a) Known Werner source','(b) BB84-compatible source'],[1/3,.25]):
        for m in range(8,0,-1):
            rr=sorted([r for r in rows if r['source_model']==model and int(r['memory_budget'])==m],key=lambda r:float(F(r['Q'])))
            x=np.array([float(F(r['Q'])) for r in rr]);y=np.array([float(r['lower']) for r in rr]);y[y<=0]=np.nan
            ax.plot(x,y,color=COL[m-1],lw=1.2,marker='.',ms=2.4,alpha=.93,label=f'$m={m}$',zorder=10-m)
        lit='werner' if model=='Werner' else 'bb84'
        rr=table(f'reproduction/extensions/data/literature/{lit}_linear_Du.csv')
        x=np.array([float(r['Q']) for r in rr]);y=np.array([float(r['rate']) for r in rr]);y[y<=0]=np.nan
        ax.plot(x,y,'--',color='#151515',lw=1.2,label='Du: supplied reference')
        if model=='Werner':
            rr=table('reproduction/extensions/data/literature/werner_linear_Sun_original.csv')
            x=np.array([float(r['Q']) for r in rr]);y=np.array([float(r['rate']) for r in rr]);y[y<=0]=np.nan
            ax.plot(x,y,':',color='#66517c',lw=1.6,label='Sun: reconstructed menu')
            rr=table('reproduction/extensions/data/literature/werner_linear_Sun_author.csv')
            ax.scatter([float(r['Q']) for r in rr],[float(r['rate']) for r in rr],s=10,
                  facecolors='none',edgecolors='#66517c',linewidths=.7,zorder=20)
        ax.set_yscale('log');ax.set_ylim(1e-7,1.3);ax.set_xlim(0,xmax)
        ax.set_xlabel('Raw bit error $Q$');ax.set_title(title,loc='left')
        ax.grid(axis='y',which='major',lw=.4,alpha=.3)
        ax.xaxis.set_major_locator(MultipleLocator(.05))
        ax.text(.03,.05,'Achieved points; finite search scope',transform=ax.transAxes,fontsize=7.5)
    axes[0].set_ylabel('Net secret bits per raw pair')
    handles=[Line2D([],[],color=COL[m-1],label=f'$m={m}$') for m in range(1,9)]
    handles += [Line2D([],[],ls='--',color='#151515',label='Du reference'),
                Line2D([],[],ls=':',marker='o',mfc='none',ms=3,color='#66517c',label='Sun menu / author points')]
    fig.legend(handles=handles,loc='lower center',ncol=5,bbox_to_anchor=(.51,-.015),columnspacing=1.25,handlelength=2.2,frameon=False)
    fig.subplots_adjust(left=.09,right=.99,top=.92,bottom=.25,wspace=.16)
    save(fig,'figure2_landscape',['data/global_rate_landscape.csv','reproduction/extensions/data/literature/'])

def matched():
    a=table('data/figure3a_matched_control.csv');b=table('data/figure3b_same_measurement.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.12))
    x=[int(r['memory_budget']) for r in a];y=[float(r['rate'])*1e4 for r in a]
    axes[0].bar(x,y,width=.55,color=[COL[m-1] for m in x],alpha=.92)
    for xi,yi in zip(x,y):axes[0].text(xi,yi+.04,f'{yi:.4f}',ha='center',fontsize=8)
    axes[0].set_ylim(0,3.13);axes[0].set_xticks(x);axes[0].set_xlabel('Allowed local memory $m$')
    axes[0].set_ylabel(r'Rate $R$ ($10^{-4}$ bits / raw pair)')
    axes[0].set_title('(a) Matched six-input control',loc='left')
    axes[0].text(.05,.93,r'$Q=0.28$; same B/Watanabe menu',transform=axes[0].transAxes,fontsize=8)
    axes[0].text(.05,.84,r'$R_4/U_3 > 1.0409$',transform=axes[0].transAxes,fontsize=9)
    axes[1].bar([2,3],[float(r['C']) for r in b],width=.55,color=[COL[1],COL[2]],alpha=.92)
    for r in b:
        xi=int(r['memory_budget']);yi=float(r['C']);axes[1].text(xi,yi+.3,f'{yi:.4f}',ha='center',fontsize=8)
    axes[1].set_ylim(0,22);axes[1].set_xticks([2,3]);axes[1].set_xlabel('Allowed local memory $m$')
    axes[1].set_ylabel('Mean raw cost $C$')
    axes[1].set_title('(b) Same five-input measurement',loc='left')
    axes[1].text(.05,.93,r'$Q=0.20$; same output and terminal',transform=axes[1].transAxes,fontsize=8)
    axes[1].text(.05,.84,r'$R_3/R_2=C_2/C_3=476/457$',transform=axes[1].transAxes,fontsize=8.8)
    for ax in axes:ax.grid(axis='y',lw=.4,alpha=.25);ax.set_axisbelow(True)
    fig.subplots_adjust(wspace=.35,left=.085,right=.99,bottom=.18,top=.89)
    save(fig,'figure3_matched',['data/figure3a_matched_control.csv','data/figure3b_same_measurement.csv'])

def targets():
    rows=table('data/figure4_target_windows.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.9))
    for ax,q,title in zip(axes,['23/100','6/25'],['(a) BB84: $Q=0.23$','(b) BB84: $Q=0.24$']):
        rr=[r for r in rows if r['Q']==q]
        for r in rr:
            m=int(r['memory']);lo=float(r['upper_low']);hi=float(r['lower_high']);target=float(r['target']);col=COL[m-1]
            ax.plot([lo,hi],[m,m],color=col,lw=3.5,solid_capstyle='butt')
            ax.plot(lo,m,'o',color=col,mfc='white',ms=5,mew=1.2);ax.plot(hi,m,'o',color=col,ms=5)
            ax.plot(target,m,'*',color='#151515',ms=7,zorder=10)
            ax.annotate(f'{target:.2g}',(target,m),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
        ax.set_xscale('log');ax.set_yticks([int(r['memory']) for r in rr]);ax.set_ylim(min(int(r['memory']) for r in rr)-.4,max(int(r['memory']) for r in rr)+.5)
        ax.set_xlabel(r'Target yield $R_*$ (bits / raw pair)');ax.set_title(title,loc='left')
        ax.grid(axis='x',which='both',lw=.4,alpha=.25)
    axes[0].set_xlim(4e-4,1.02e-3);axes[0].set_xticks([4e-4,6e-4,8e-4,1e-3],labels=[r'$4\!\times\!10^{-4}$',r'$6\!\times\!10^{-4}$',r'$8\!\times\!10^{-4}$',r'$10^{-3}$']);axes[0].minorticks_off()
    axes[1].set_xscale('linear');axes[1].set_xlim(0,1.05e-4)
    axes[1].set_xticks([0,2e-5,4e-5,6e-5,8e-5,1e-4],labels=['0','2','4','6','8','10'])
    axes[1].set_xlabel(r'Target yield $R_*$ ($10^{-5}$ bits / raw pair)')
    axes[0].set_ylabel(r'Certified minimum memory $m_*$')
    fig.subplots_adjust(left=.085,right=.99,wspace=.27,bottom=.2,top=.88)
    save(fig,'figure4_targets',['data/figure4_target_windows.csv'])

def decomposition():
    rows=table('data/bb84_policy_decomposition.csv')
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.55))
    for pol,m in [('BB23_m5',5),('BB23_m6',6)]:
        rr=[r for r in rows if r['policy']==pol];z=[float(r['z']) for r in rr]
        for ax,key in zip(axes,['C','r_cl','rate']):ax.plot(z,[float(r[key])*(1000 if key=='rate' else 1) for r in rr],color=COL[m-1],label=f'$m={m}$')
    labels=['Mean raw cost $C$',r'Classical yield $r_{\rm cl}$',r'Net rate $R$ ($10^{-3}$)']
    for ax,lab in zip(axes,labels):
        ax.set_title(lab,loc='left',fontsize=9);ax.set_xlabel('Hidden correlation $z$');ax.grid(lw=.4,alpha=.25);ax.set_xlim(0,.23)
        ax.set_xticks([0,.1,.2]);ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    axes[0].legend(frameon=False,loc='best')
    fig.subplots_adjust(left=.06,right=.99,wspace=.43,bottom=.24,top=.86)
    save(fig,'figureS1_decomposition',['data/bb84_policy_decomposition.csv'])

def main():
    protocol();landscape();matched();targets();decomposition()
    (ROOT/'validation/figure_provenance.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps({'status':'passed','figures':[r['figure'] for r in records],'format':'vector PDF + PNG previews; all numerical panel data in data/'},indent=2))
if __name__=='__main__':main()
