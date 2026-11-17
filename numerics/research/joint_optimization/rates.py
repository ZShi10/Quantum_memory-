"""IID no-OTP code rates from complete (I,X,Y,Z) enumerators.

The syndrome-resolved entropy polynomials are compressed exactly, by integer
equality only. This compression preserves rates for every Bell distribution.
"""
from pathlib import Path
from collections import Counter
import numpy as np
from algebra import span,parity,compositions,physical_types


def xlog2x(x):
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.where(x>0,x*np.log2(x),0.)


def entropy(p):return -xlog2x(p).sum(axis=-1)


def monomials(states,types):
    states=np.asarray(states)
    return np.prod(states[...,None,:]**np.asarray(types),axis=-1)


class CodeKernel:
    def __init__(self,entry):
        self.entry=entry;self.n=n=entry['n'];self.k=k=entry['k']
        rows=entry['rows'];words=span(rows);types=compositions(n)
        lookup={p:i for i,p in enumerate(types)}
        counts=np.zeros((1<<(n+k),len(types)),dtype=np.int16)
        for physical,typ in enumerate(physical_types(n)):
            x=physical&((1<<n)-1);z=physical>>n
            phase=sum(parity(z&g)<<j for j,g in enumerate(rows))
            counts[(x<<k)|phase,lookup[typ]]+=1
        unique={};syndromes={}
        for x in range(1<<n):syndromes.setdefault(min(x^w for w in words),[]).append(x)
        histograms=[]
        for rep,xs in sorted(syndromes.items()):
            hist=Counter()
            for x in xs:
                for row in counts[x<<k:(x+1)<<k]:
                    if not row.any():continue
                    key=tuple(map(int,row))
                    index=unique.setdefault(key,len(unique));hist[index]+=1
            histograms.append((rep,hist))
        hist_classes={};representatives={}
        for rep,hist in histograms:
            key=tuple(sorted(hist.items()))
            hist_classes[key]=hist_classes.get(key,0)+1
            representatives.setdefault(key,[]).append(rep)
        self.types=np.asarray(types)
        self.polys=np.array(list(unique),dtype=float)
        self.hist=np.zeros((len(hist_classes),len(unique)))
        self.multiplicity=np.array(list(hist_classes.values()))
        self.syndrome_classes=[]
        for j,key in enumerate(hist_classes):
            for i,mult in key:self.hist[j,i]=mult
            self.syndrome_classes.append(representatives[key])
        self.marginal=self.hist@self.polys

    def components_from_monomials(self,powers):
        joint=powers@self.polys.T
        ps=powers@self.marginal.T
        hmass=-(xlog2x(joint)@self.hist.T)+xlog2x(ps)
        return ps,hmass

    def components(self,states):
        return self.components_from_monomials(monomials(states,self.types))

    def evaluate_powers(self,powers,all_syndromes=True):
        ps,hmass=self.components_from_monomials(powers)
        gain=np.maximum(self.k*ps-hmass,0.)
        if not all_syndromes:
            j=next(j for j,s in enumerate(self.syndrome_classes) if 0 in s)
            return gain[...,j]/self.n
        return (gain@self.multiplicity)/self.n

    def evaluate(self,states):
        return self.evaluate_powers(monomials(states,self.types))


def best_codes(states,kernels,batch_size=2048):
    """Exact finite enumeration of rate formulas, with deterministic ties."""
    states=np.asarray(states).reshape(-1,4)
    rates=np.zeros(len(states));selected=np.full(len(states),-1,dtype=int)
    endpoints=np.zeros(len(states));end_selected=selected.copy()
    identity=np.maximum(1-entropy(states),0.)
    active=np.flatnonzero(states.max(axis=1)>0.5+1e-14)
    for start in range(0,len(active),batch_size):
        idx=active[start:start+batch_size];ss=states[idx]
        powers={n:monomials(ss,compositions(n)) for n in range(1,max(k.n for k in kernels)+1)}
        rb=np.zeros(len(idx));cb=np.full(len(idx),-1,dtype=int)
        re=np.zeros(len(idx));ce=cb.copy()
        for ci,kernel in enumerate(kernels):
            r=kernel.evaluate_powers(powers[kernel.n])
            r=np.where(r>1e-12,r,0.)
            take=r>rb+1e-14;rb[take]=r[take];cb[take]=ci
            if kernel.entry['endpoint']:
                take=r>re+1e-14;re[take]=r[take];ce[take]=ci
        rates[idx]=rb;selected[idx]=cb;endpoints[idx]=re;end_selected[idx]=ce
    return dict(rate=rates,code=selected,endpoint_rate=endpoints,endpoint_code=end_selected,oneway=identity)
