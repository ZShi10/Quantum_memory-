"""Independent checks of selected formulas in the uploaded QKD manuscript.

This is NOT a replication of the authors' exhaustive catalog or an interval
certificate. Gate propagation and exact rational costs are independent of the
manuscript's saved software. Numerical minimizations are exploratory, not proofs
of a minimum over a continuum or of whole-class optimality.

Dependencies: numpy, scipy; optional mpmath for high-precision entropy evaluation.
"""
from fractions import Fraction as F
from math import log2, factorial
from functools import lru_cache
import itertools
import json
import numpy as np
from scipy.special import logsumexp, xlogy
from scipy.optimize import minimize_scalar

# Bell order: I, X, Y, Z; Pauli error coordinates (bit, phase).
BITS = [(0,0),(1,0),(1,1),(0,1)]
INDEX = {x:i for i,x in enumerate(BITS)}
GATES = {
    'Z': [('C',0,1),('C',1,0)],
    'X': [('H',0),('H',1),('C',0,1),('C',1,0)],
    'R': [('C',0,1),('H',0),('C',0,1),('C',1,0)],
    'C': [('H',0),('C',0,1),('C',1,0)]
}

def propagate(d, i, j):
    x=[BITS[i][0],BITS[j][0]]; z=[BITS[i][1],BITS[j][1]]
    for g in GATES[d]:
        if g[0]=='H':
            a=g[1]; x[a],z[a]=z[a],x[a]
        else:
            a,b=g[1:]; x[b]^=x[a]; z[a]^=z[b]
    return None if x[1] else INDEX[(x[0],z[0])]
MAPS={d:[(i,j,propagate(d,i,j)) for i in range(4) for j in range(4)] for d in GATES}

def decoder(d, pl, pr):
    q=[0,0,0,0]
    for i,j,k in MAPS[d]:
        if k is not None: q[k]+=pl[i]*pr[j]
    s=sum(q)
    if s<=0: raise ValueError('Decoder acceptance is zero.')
    return [v/s for v in q],s

def axis(p, a):
    ix={'Z':[0,1,2,3], 'X':[0,3,2,1], 'Y':[0,1,3,2]}[a]
    return [p[i] for i in ix]

def entropy(p):
    return -sum(float(v)*log2(float(v)) for v in p if v>0)

def h2(x):
    x=min(1.,max(0.,float(x)))
    return entropy([x,1-x])

def B(p):
    a,b,c,d=p
    q=[a*a+d*d,b*b+c*c,2*b*c,2*a*d]; s=sum(q)
    return [v/s for v in q],s

def watanabe(p):
    a=p[0]+p[3]; c=p[1]+p[2]
    w1=1-entropy(p)+(float(a*c)*h2((p[0]*p[1]+p[3]*p[2])/(a*c)) if a*c else 0)
    bp,s=B(p)
    return max(0,w1,float(s)/2*(1-entropy(bp)))

def raw_pump(p,d,m):
    q=p[:]; cost=1
    for _ in range(m-1):
        q,s=decoder(d,q,p); cost=(cost+1)/s
    return q,cost

def homogeneous(p,word):
    q=p[:]; cost=1
    for d in word:
        q,s=decoder(d,q,q); cost=2*cost/s
    return q,cost

@lru_cache(None)
def type_table(n):
    counts=[]; logs=[]; parity=[]
    from scipy.special import gammaln
    for i in range(n+1):
        for j in range(n-i+1):
            for k in range(n-i-j+1):
                t=[i,j,k,n-i-j-k]
                counts.append(t); logs.append(gammaln(n+1)-sum(gammaln(v+1) for v in t)); parity.append((j+k)%2)
    return np.array(counts),np.array(logs),np.array(parity)

def spc_nootp(p,n):
    if n<2: raise ValueError('SPC length must be >=2.')
    p=np.array(p,dtype=float)
    t, lm, flags=type_table(n)
    # xlogy defines 0*log(0)=0 at boundary sources.
    lp=xlogy(t,p).sum(axis=1)
    lq=xlogy(t,p[[3,2,1,0]]).sum(axis=1)
    w=np.exp(lm+lp)
    merged=np.logaddexp(lp,lq)
    rate=0.
    for s in [0,1]:
        ii=(flags==s)&(w>0)
        P=w[ii].sum()
        if P:
            Hweighted=-(w[ii]*merged[ii]).sum()/np.log(2)+P*np.log2(P)
            rate+=max(0.,(n-1)*P-Hweighted)/n
    return rate

def evaluate_policy(p,kind,word_or_m,d,final_axis,brounds,terminal):
    if kind=='pump': q,cost=raw_pump(p,d,word_or_m)
    elif kind=='homogeneous': q,cost=homogeneous(p,word_or_m)
    else: raise ValueError(kind)
    q=axis(q,final_axis); y=1/float(cost)
    for _ in range(brounds):
        q,s=B(q); y*=float(s)/2
    r=watanabe(q) if terminal=='W' else spc_nootp(q,int(terminal))
    return y*r

def exploratory_minimum(fn,lo,hi,points=1001):
    xx=np.unique(np.r_[np.linspace(lo,hi,points), lo+(hi-lo)*np.geomspace(1e-9,0.02,121), hi-(hi-lo)*np.geomspace(1e-9,0.02,121)]); yy=np.array([fn(x) for x in xx])
    candidates=[(float(yy[0]),lo),(float(yy[-1]),hi)]
    for i in range(1,len(xx)-1):
        if yy[i]<=yy[i-1] and yy[i]<=yy[i+1] and (yy[i]<yy[i-1] or yy[i]<yy[i+1]):
            sol=minimize_scalar(fn,bounds=(xx[i-1],xx[i+1]),method='bounded',options={'xatol':1e-14})
            candidates.append((float(sol.fun),float(sol.x)))
    return sorted(candidates)

