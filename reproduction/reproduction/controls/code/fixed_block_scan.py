"""Memory/input-size controls for a fully specified common classical menu.

All supplied graph representatives, every nonzero logical dual, all ED flags
with full-block consumption, OR zero-syndrome selection with optimal fixed-order
stopping. Common menu: 0..16 pooled B rounds followed by Watanabe's protocol.
This is not the full general-code menu and uses floating arithmetic for discovery.
Selected structural identities are independently checked with exact rationals.
"""
from __future__ import annotations
from pathlib import Path
import json,csv,time
import numpy as np
from numba import njit
ROOT=Path(__file__).resolve().parents[1]

@njit(cache=True)
def rank_binary(rows):
    a=rows.copy();rank=0
    for bit in range(15,-1,-1):
        pivot=-1
        for i in range(rank,len(a)):
            if (a[i]>>bit)&1:pivot=i;break
        if pivot<0:continue
        t=a[rank];a[rank]=a[pivot];a[pivot]=t
        for i in range(rank+1,len(a)):
            if (a[i]>>bit)&1:a[i]^=a[rank]
        rank+=1
    return rank

@njit(cache=True)
def fwht(a):
    out=a.copy();n=out.shape[1];h=1
    while h<n:
        for row in range(out.shape[0]):
            for j in range(0,n,2*h):
                for k in range(j,j+h):
                    x=out[row,k];y=out[row,k+h]
                    out[row,k]=x+y;out[row,k+h]=x-y
        h*=2
    return out

@njit(cache=True)
def entropy4(q):
    h=0.
    for v in q:
        if v>0:h-=v*np.log2(v)
    return h

@njit(cache=True)
def h2(x):
    if x<=0 or x>=1:return 0.
    return -x*np.log2(x)-(1-x)*np.log2(1-x)

@njit(cache=True)
def terminal_menu(q,Bmax=16):
    y=1.;best=0.;bestB=0
    if np.max(q)<=.5:return best,bestB
    for bround in range(Bmax+1):
        if np.max(q)<=.5 or y<1e-20:break
        a=q[0]+q[3];b=q[1]+q[2];s=a*a+b*b
        raw=np.array([q[0]*q[0]+q[3]*q[3],q[1]*q[1]+q[2]*q[2],2*q[1]*q[2],2*q[0]*q[3]])
        if s<=0:return best,bestB
        qb=raw/s
        w1=1-entropy4(q)
        if a*b>0:w1+=a*b*h2((q[0]*q[1]+q[3]*q[2])/(a*b))
        w=max(0.,w1,s*(1-entropy4(qb))/2)
        if y*w>best:best=y*w;bestB=bround
        q=qb;y*=s/2
    return best,bestB

@njit(cache=True)
def optimize_orders(cuts,probabilities,mmax):
    # rows: all nonzero logical duals; columns: loaded subsets.
    V,S=probabilities.shape;n=0
    while (1<<n)<S:n+=1
    result=np.full((V,mmax),np.inf)
    for v in range(V):
        for m in range(1,mmax+1):
            dp=np.full(S,np.inf);dp[0]=0.
            for a in range(1,S):
                if a!=S-1 and cuts[a]>=m:continue
                best=np.inf
                for i in range(n):
                    if (a>>i)&1:
                        prev=a^(1<<i);cost=dp[prev]+probabilities[v,prev]
                        if cost<best:best=cost
                dp[a]=best
            result[v,m-1]=dp[-1]
    return result

@njit(cache=True)
def rates_all_duals(P,D,N,Bmax=16):
    S=len(P);zeros=np.zeros(S);allflags=np.zeros(S);Bzeros=np.zeros(S,dtype=np.int32)
    for v in range(1,S):
        pivot=0
        while not ((v>>pivot)&1):pivot+=1
        for y in range(S):
            if (y>>pivot)&1:continue
            yy=y^v;p=P[y]+P[yy]
            if p<1e-280:continue
            q=np.array([(P[y]+D[v,y])/2,(P[yy]+D[v,yy])/2,
                        (P[yy]-D[v,yy])/2,(P[y]-D[v,y])/2])
            for k in range(4):q[k]=max(0.,q[k])
            q/=np.sum(q)
            r,br=terminal_menu(q,Bmax)
            allflags[v]+=p*r/N
            if y==0:zeros[v]=p*r;Bzeros[v]=br
    return zeros,allflags,Bzeros


def preprocess(graph):
    n=graph['N'];S=1<<n;adj=np.zeros(n,dtype=np.int64)
    for i,j in graph['edges']:adj[i]|=1<<j;adj[j]|=1<<i
    au=np.zeros(S,dtype=np.int64)
    for u in range(1,S):
        bit=u&-u;i=bit.bit_length()-1;au[u]=au[u^bit]^adj[i]
    u=np.arange(S,dtype=np.int64);support=u|au
    pc=np.array([i.bit_count() for i in range(S)])
    weights=pc[(au[None,:]^u[:,None])|u[None,:]]
    subsets=np.asarray((support[None,:] & (S-1-u[:,None]))==0,dtype=float)
    cnt=subsets.sum(axis=1)[:,None]
    counts=(cnt+fwht(subsets))/2
    cuts=np.array([rank_binary(adj[((u0>>np.arange(n))&1).astype(bool)]&(S-1-u0)) for u0 in range(S)],dtype=np.int32)
    widths=np.zeros(S,dtype=np.int32)
    for a in range(1,S):widths[a]=min(max(widths[a^(1<<i)],cuts[a]) for i in range(n) if (a>>i)&1)
    assert 1+widths[-1]==graph['original_min_memory']
    return n,weights,subsets,counts,cuts,int(1+widths[-1])


def scan(qs=(.15,.20,.23,.25,.27,.28,.29,.30),Bmax=16):
    catalog=json.loads((ROOT/'data/graph_representatives.json').read_text())['graphs']
    results=[];details=[];start=time.time()
    tables={q:{'fixed':np.zeros((9,5)),'early':np.zeros((9,5)),'choice':{}} for q in qs}
    for gi,(name,g) in enumerate(catalog.items()):
        n,weights,subsets,counts,cuts,mmin=preprocess(g);S=1<<n
        for Q in qs:
            factors=(1-2*Q)**weights
            D=fwht(factors)/S;P=D[0]
            # Walsh formula for supported public hyperplanes.
            vals=subsets*factors[0][None,:]
            numer=(vals.sum(axis=1)[:,None]+fwht(vals))/2
            probs=(numer/counts).T
            probs=np.clip(probs,0,1)
            probs[:,0]=1.
            costs=optimize_orders(cuts,probs[1:],4)
            numerator,full,bsel=rates_all_duals(P,D,n,Bmax)
            for mm in range(1,5):
                if mm>=mmin:
                    j=int(np.argmax(full[1:]))+1;value=full[j]
                    if value>tables[Q]['fixed'][n,mm]:
                        tables[Q]['fixed'][n,mm]=value;tables[Q]['choice'][f'fixed_{n}_{mm}']={'graph':name,'dual':j}
                r=numerator[1:]/costs[:,mm-1];j=int(np.argmax(r))+1;value=r[j-1]
                if value>tables[Q]['early'][n,mm]:
                    tables[Q]['early'][n,mm]=value
                    tables[Q]['choice'][f'early_{n}_{mm}']={'graph':name,'dual':j,'B':int(bsel[j]),'mean_attempt_cost':float(costs[j-1,mm-1]),'acceptance':float(P[0]+P[j])}
        if gi%40==0:print('graphs',gi+1,'/',len(catalog),'seconds',round(time.time()-start,2),flush=True)
    for Q,t in tables.items():
        # at-most input budgets. No extra resource is charged for choosing smaller N.
        for mode in ['fixed','early']:
            cumulative=np.maximum.accumulate(t[mode],axis=0)
            for n in range(1,9):
                for mm in range(1,5):
                    results.append({'Q':Q,'Nmax':n,'memory':mm,'implementation':mode,'rate':float(cumulative[n,mm]),'fixed_N_rate':float(t[mode][n,mm])})
        details.append({'Q':Q,'choices':t['choice']})
    with (ROOT/'data/fixed_N_memory_control.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=results[0]);w.writeheader();w.writerows(results)
    (ROOT/'data/fixed_N_control_choices.json').write_text(json.dumps(details,indent=2))
    report={'status':'floating_complete_scan_of_explicit_control_family','graph_representatives':len(catalog),
      'logical_duals_before_cost_quotient':sum((1<<g['N'])-1 for g in catalog.values()),'Q':list(qs),'B_max':Bmax,
      'classical_menu':'B prefixes followed by Watanabe; same menu at every N and memory',
      'implementations':'all flags/full cost OR zero relative syndrome/optimal deterministic stopping order',
      'seconds':time.time()-start,'not_claimed':'No full-code-menu optimum; no interval numerical optimality certificate for this control scan.'}
    (ROOT/'verification/fixed_N_scan.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
    for Q,t in tables.items():
        print('Q',Q,'max full',np.max(t['fixed'],axis=0),'max early',np.max(t['early'],axis=0),flush=True)

if __name__=='__main__':scan()
