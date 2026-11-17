"""Exact probabilities and memory-constrained stopping for complete Clifford readout.

For symmetric Bell input, graph bases up to local H cover the real basis family.
The adaptive algorithm reveals only designated public checks. It optimizes
stopping and input order in this specified complete-readout compiler class,
not arbitrary low-memory quantum instruments.
"""
from __future__ import annotations
from fractions import Fraction as F
from functools import lru_cache
from itertools import permutations
import math
import numpy as np

def rank(rows):
    piv={}
    for w in rows:
        w=int(w)
        while w:
            i=w.bit_length()-1
            if i in piv:w^=piv[i]
            else:piv[i]=w;break
    return len(piv)

def basis(rows):
    piv={}
    for w in rows:
        w=int(w)
        while w:
            i=w.bit_length()-1
            if i in piv:w^=piv[i]
            else:piv[i]=w;break
    return tuple(piv.values())

def graph_data(adj):
    n=len(adj);size=1<<n;full=size-1
    z=[0]*size
    for a in range(1,size):
        low=a&-a;z[a]=z[a^low]^adj[low.bit_length()-1]
    support=[a|z[a] for a in range(size)]
    cuts=[rank([adj[i]&(full^a) for i in range(n) if a>>i&1]) for a in range(size)]
    d=[n]*size;d[0]=0
    for a in range(1,size):d[a]=min(max(d[a^(1<<i)],cuts[a]) for i in range(n) if a>>i&1)
    return z,support,cuts,1+d[-1]

def fwht(values):
    a=list(values);h=1
    while h<len(a):
        for i in range(0,len(a),2*h):
            for j in range(i,i+h):x,y=a[j],a[j+h];a[j],a[j+h]=x+y,x-y
        h*=2
    return a

def branch_weights(adj,dual,p):
    """Unnormalized Bell laws for all original public relative syndromes.

    Exact Fraction inputs give exact results. Independent checks use 4**n
    Pauli enumeration. Graph generators are X_i prod_neighbors Z_j;
    the logical conjugate is Z**dual. Bell order I,X,Y,Z is fixed.
    """
    n=len(adj);size=1<<n;z,_,_,_=graph_data(adj)
    char=[1,1-2*(p[2]+p[3]),1-2*(p[1]+p[2]),1-2*(p[1]+p[3])]
    # index is x + 2*z: I, X, Z, Y
    chars=[]
    for c in (0,1):
        for a in range(size):
            zz=z[a]^(dual if c else 0);val=1
            for i in range(n):val*=char[((a>>i)&1)+2*((zz>>i)&1)]
            chars.append(val)
    joint=[x/(2*size) for x in fwht(chars)]
    pivot=(dual&-dual).bit_length()-1;other=[i for i in range(n) if i!=pivot]
    out=[[0,0,0,0] for _ in range(size//2)]
    index={(0,0):0,(1,0):1,(1,1):2,(0,1):3}
    for y in range(size):
        bit=(y>>pivot)&1
        s=sum(((((y>>i)&1)^(((dual>>i)&1)*bit)))<<j for j,i in enumerate(other))
        for phase in (0,1):out[s][index[bit,phase]]+=joint[y+phase*size]
    return out

def public_partitions(adj,dual):
    n=len(adj);full=(1<<n)-1;z,support,cuts,memory=graph_data(adj)
    pivot=(dual&-dual).bit_length()-1;other=[i for i in range(n) if i!=pivot]
    ys=[sum(((s>>j)&1)<<i for j,i in enumerate(other)) for s in range(1<<(n-1))]
    parts=[]
    for a in range(full+1):
        checks=basis([b for b in range(1,full+1) if not(support[b]&(full^a)) and not((b&dual).bit_count()%2)])
        groups={}
        for s,y in enumerate(ys):
            label=sum(((b&y).bit_count()%2)<<j for j,b in enumerate(checks))
            groups.setdefault(label,[]).append(s)
        parts.append(tuple(tuple(v) for v in groups.values()))
    return parts,cuts,memory

def stopping_test(adj,dual,mass,reward,price,memory=99):
    """Bellman test: some allowed policy has reward/cost > price iff root>0.

    All values are unnormalized probabilities. Each non-final node can stop
    without reward, or consume one more raw input, revealing newly available
    public checks. At the final node its specified branch reward is collected.
    Fractions make the comparison exact, given exact branch reward bounds.
    """
    n=len(adj);full=(1<<n)-1;parts,cuts,m_min=public_partitions(adj,dual)
    zero=mass[0]*0
    val=[None]*(full+1)
    val[full]={g:sum(reward[s] for s in g) for g in parts[full]}
    for a in range(full-1,-1,-1):
        if a and cuts[a]>memory-1:continue
        current={}
        for g in parts[a]:
            best=zero;P=sum(mass[s] for s in g);gs=set(g)
            for v in range(n):
                if a>>v&1:continue
                b=a|(1<<v)
                if val[b] is None:continue
                candidate=sum(x for h,x in val[b].items() if h[0] in gs)-price*P
                best=max(best,candidate)
            current[g]=best
        val[a]=current
    return next(iter(val[0].values()))

def optimum_stopping(adj,dual,mass,reward,memory=99,tol=1e-11):
    lo=0.;hi=min(1.,sum(reward))
    if graph_data(adj)[-1]>memory:return 0.,0.
    while hi-lo>tol:
        mid=(lo+hi)/2
        if stopping_test(adj,dual,mass,reward,mid,memory)>0:lo=mid
        else:hi=mid
    return lo,hi

def zero_flag_cost(adj,dual,p,memory):
    weights=branch_weights(adj,dual,p);mass=[sum(w) for w in weights]
    parts,cuts,m_min=public_partitions(adj,dual);n=len(adj);full=(1<<n)-1
    if m_min>memory:return None,None
    survival=[sum(mass[s] for s in next(g for g in gs if 0 in g)) for gs in parts]
    dp=[None]*(full+1);parent=[None]*(full+1);dp[0]=mass[0]*0
    for a in range(1,full+1):
        if a!=full and cuts[a]>memory-1:continue
        for v in range(n):
            if not(a>>v&1):continue
            b=a^(1<<v)
            if dp[b] is None:continue
            q=dp[b]+survival[b]
            if dp[a] is None or q<dp[a]:dp[a],parent[a]=q,(b,v)
    if dp[-1] is None:return None,None
    order=[];a=full
    while a:a,v=parent[a];order.append(v)
    return dp[-1]/mass[0],order[::-1]

def source_reward(weights,exact=False):
    """Upper reward for arbitrary downstream key extraction of each Bell law.
    exact=True uses rational square bound (2f-1)_+**2 on E_R; otherwise E_R.
    """
    from bell_core import h2
    out=[]
    for w in weights:
        P=sum(w)
        if P<=0:out.append(P*0);continue
        excess=max(0,2*max(w)-P)
        if exact:out.append(excess*excess/P)
        else:out.append(float(P)*(1-h2((1+float(excess/P))/2)))
    return out
