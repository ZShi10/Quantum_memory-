"""Zero-ED-syndrome probabilities via Pauli characters, O(m 2**m).

Character enumerators are signed integer polynomials, divided by 2**(m+1).
They work for arbitrary IID Bell inputs. No Werner twirl is used.
"""
from common import *
from functools import lru_cache
from fractions import Fraction as F
from math import comb
import numpy as np
from algebra import compositions
from bernstein import monomial_coefficients


@lru_cache(None)
def base_arrays(m,edges):
    t=np.arange(1<<m,dtype=np.int64);z=np.zeros_like(t)
    for a,b in edges:z^=((t>>a)&1)<<b;z^=((t>>b)&1)<<a
    pop=np.array([i.bit_count() for i in range(1<<m)])
    return t,z,pop


def enumerator(m,edges,v,rotations=None):
    t,z,pop=base_arrays(m,tuple(map(tuple,edges)))
    types=compositions(m);lookup=np.full((m+1,)*3,-1,dtype=int)
    for j,typ in enumerate(types):lookup[typ[1],typ[2],typ[3]]=j
    sign=1-2*(pop[t&v]%2);terms=[]
    for h in (0,1):
        phase=z^(h*v)
        if rotations is None:
            x=pop[t&~phase];y=pop[t&phase];zz=pop[phase&~t]
        else:
            counts=np.zeros((3,len(t)),dtype=np.int16)
            for j,perm in enumerate(rotations):
                xx=(t>>j)&1;pp=(phase>>j)&1
                label=np.where(pp,3-xx,xx)
                mapped=np.asarray(perm)[label]
                for a in range(1,4):counts[a-1]+=(mapped==a)
            x,y,zz=counts
        indices=lookup[x,y,zz]
        terms.append(np.bincount(indices,minlength=len(types)).astype(np.int64))
        terms.append(np.bincount(indices,weights=sign,minlength=len(types)).astype(np.int64))
    s0,sv,t0,tv=terms
    return np.stack((s0+t0,sv+tv,sv-tv,s0-t0))


def evaluate(entry,p):
    p=np.asarray(p);r=np.stack((np.ones(p.shape[:-1]),1-2*(p[...,2]+p[...,3]),
                              1-2*(p[...,1]+p[...,3]),1-2*(p[...,1]+p[...,2])),axis=-1)
    typ=np.asarray(compositions(entry['m']))
    mon=np.prod(r[...,None,:]**typ,axis=-1)
    raw=mon@np.asarray(entry['characters']).T/(2**(entry['m']+1))
    assert np.min(raw)>-1e-12
    return np.maximum(raw,0.)


def bb84_polynomials(entry,eb,ep,exact=False):
    m=entry['m'];t=min(eb,ep)
    if not exact:
        chars=np.array([[1,1],[1-2*ep]*2,[1-2*(eb+ep),1-2*(eb+ep)+4*t],[1-2*eb]*2])
        chars=chars[entry.get('axis_permutation',[0,1,2,3])]
        raw=np.asarray(entry['characters'])@monomial_coefficients(chars,np.asarray(compositions(m)))/(2**(m+1))
        assert raw.min()>-2e-12
        return np.maximum(raw,0.)
    from interval_certificate import mono,matpoly
    eb=F(str(eb));ep=F(str(ep));t=min(eb,ep)
    chars=[[F(1)]*2,[1-2*ep]*2,[1-2*(eb+ep),1-2*(eb+ep)+4*t],[1-2*eb]*2]
    chars=[chars[j] for j in entry.get('axis_permutation',[0,1,2,3])]
    raw=matpoly(entry['characters'],mono(chars,compositions(m)))
    raw=[[v/(2**(m+1)) for v in row] for row in raw]
    assert all(v>=0 for row in raw for v in row)
    return raw
