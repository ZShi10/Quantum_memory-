"""Cheap rigorous SPC value enclosure before the full ambiguity walk.

The phase-complement ambiguity in either syndrome has unnormalized
entropy in [0,P_s]. No derivative of that unknown ambiguity is asserted.
"""
from core import *
import interval_jets as j


def bound(p,n):
    entropy=j.h(p);H=entropy.sum()
    hp=entropy.dot(np.array([1,0,0,1]))-entropy.dot(np.array([0,1,1,0]))
    d=1-2*(p[...,1]+p[...,2]);low=[];high=[]
    for sign in (1,-1):
        mass=((1+sign*d**n)/2).clip()
        physical=n*(H+sign*d**(n-1)*hp)/2
        gain=(n-1)*mass-physical+j.h(mass)
        low.append(j.positive(gain));high.append(j.positive(gain+mass))
    lower=((low[0]+low[1])/n).v[0]
    upper=((high[0]+high[1])/n).v[1]
    return j.Jet((lower,upper),(np.full_like(lower,-np.inf),np.full_like(upper,np.inf)))


def verify():
    from fractions import Fraction as F
    from spc_interval_dynamic import all_bounds
    rng=np.random.default_rng(26117);points=rng.dirichlet(np.ones(4),16)
    # Independent exact ambiguity walks, including positive-rate and zero-rate inputs.
    points=np.r_[points,[[.91,.01,.02,.06],[.76,.03,.02,.19]]]
    boxes=(np.nextafter(points,-np.inf),np.nextafter(points,np.inf))
    p=j.Jet(boxes);count=0
    for n in (8,17,60):
        cheap=bound(p,n).v
        full=all_bounds(boxes,np.array([n]),np.ones((len(points),1),bool))
        assert np.all(cheap[0]<=full[0][:,0]+1e-13)
        assert np.all(cheap[1]>=full[1][:,0]-1e-13)
        count+=len(points)
    atomic_json(DATA/'coarse_spc_checks.json',dict(status='passed',independent_walk_comparisons=count,
              justification='In a two-element phase-complement class the entropy lost by merging is at least zero and at most one bit; multiplied by the syndrome probability its interval is [0,P_s].'))
    print('Coarse SPC:',count,'independent walk comparisons passed.',flush=True)


if __name__=='__main__':verify()
