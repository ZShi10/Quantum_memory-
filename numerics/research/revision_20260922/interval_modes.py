"""Outward probability and net-rate enclosures for all three Du modes.

Uses the audited binary64 engine with an analytic logarithm remainder.
No libm log accuracy or optimization solver tolerance is assumed here.
"""
from core import *
from fractions import Fraction as F
import stable_intervals as iv


def signed_mul(a,b):
    lows=[iv.down(x*y) for x in a for y in b]
    highs=[iv.up(x*y) for x in a for y in b]
    return np.minimum.reduce(lows),np.maximum.reduce(highs)


def signed_dot(a,weights):
    pos=iv.positive_dot((np.maximum(0,a[0]),np.maximum(0,a[1])),weights)
    neg=iv.positive_dot((np.maximum(0,-a[1]),np.maximum(0,-a[0])),weights)
    return iv.sub(pos,neg)


def signed_sum(a,axis=-1):
    pos=iv.positive_sum((np.maximum(0,a[0]),np.maximum(0,a[1])),axis)
    neg=iv.positive_sum((np.maximum(0,-a[1]),np.maximum(0,-a[0])),axis)
    return iv.sub(pos,neg)


def components(powers,kernel):
    joint=iv.positive_dot(powers,kernel.polys.T)
    ps=iv.positive_dot(powers,kernel.marginal.T)
    hj=iv.entropy_interval(joint);hs=iv.entropy_interval(ps)
    # The exact probabilities are in [0,1], including pure-state endpoints.
    hj=(np.maximum(0,hj[0]),np.maximum(0,hj[1]))
    hs=(np.maximum(0,hs[0]),np.maximum(0,hs[1]))
    total=iv.positive_dot(hj,kernel.hist.T)
    gain=iv.add(iv.sub(iv.scale(ps,kernel.k),total),hs)
    gain=signed_mul(gain,iv.rational(F(1,kernel.n)))
    gain=signed_mul(gain,(kernel.multiplicity,kernel.multiplicity))
    cost=iv.scale(iv.positive_dot(hs,kernel.multiplicity),F(1,kernel.n))
    return gain,cost


def best_short_bounds(state,kernels,physical,batch=256):
    lower=np.zeros(len(state[0]));upper=lower.copy()
    for start in range(0,len(lower),batch):
        stop=min(start+batch,len(lower));local=(state[0][start:stop],state[1][start:stop])
        powers={n:iv.monomial_bounds(local,np.asarray(compositions(n))) for n in {k.n for k in kernels}}
        for k in kernels:
            gain,_=components(powers[k.n],k)
            no=iv.positive_sum((np.maximum(0,gain[0]),np.maximum(0,gain[1])))
            gain,cost=components(powers[k.n],physical[k.entry['id']])
            gross=iv.positive_sum((np.maximum(0,gain[0]),np.maximum(0,gain[1])))
            plain=iv.sub(gross,iv.rational(F(k.n-k.k,k.n)))
            compressed=iv.sub(gross,cost)
            for lo,hi in (no,plain,compressed):
                lower[start:stop]=np.maximum(lower[start:stop],lo)
                upper[start:stop]=np.maximum(upper[start:stop],hi)
    return lower,upper


def menu_values(menu,entry,b,p,ts):
    b,p=F(str(b)),F(str(p));initial=[]
    for t in ts:
        z=min(b,p)*F(str(t));initial.append([1-b-p+z,b-z,z,p-z])
    initial_bounds=[[(iv.rational(v)) for v in row] for row in initial]
    bell=(np.array([[v[0] for v in row] for row in initial_bounds]),
          np.array([[v[1] for v in row] for row in initial_bounds]))
    raw_powers=iv.monomial_bounds(bell,np.asarray(compositions(entry['m'])))
    lows=np.zeros((len(ts),menu.count));highs=lows.copy()
    for branch,tensor in enumerate(entry['tensor']):
        raw=iv.positive_dot(raw_powers,np.asarray(tensor).T)
        mass=iv.positive_sum(raw)
        active=mass[0]>0
        safe=(np.where(active,mass[0],1),np.where(active,mass[1],1))
        state=iv.div(raw,(safe[0][:,None],safe[1][:,None]))
        powers={n:iv.monomial_bounds(state,np.asarray(compositions(n))) for n in {k.n for k in menu.kernels}}
        scale=iv.scale(mass,F(1,entry['m']))
        for o in menu.options:
            if o['branch']!=branch:continue
            k=o['kernel'];kk=o['entropy_kernel']
            gain,cost=components(powers[k.n],kk)
            g=signed_mul(gain,(scale[0][:,None],scale[1][:,None]))
            lows[:,o['beta']],highs[:,o['beta']]=g
            if o['mode']=='no_otp':cost=(np.zeros(len(ts)),np.zeros(len(ts)))
            elif o['mode']=='otp_plain':cost=iv.rational(F(k.n-k.k,k.n))
            cost=iv.mul(cost,scale)
            lows[:,o['alpha']],highs[:,o['alpha']]=-cost[1],-cost[0]
            if not active.all():
                # If a branch is exactly absent, all of its contributions vanish.
                assert np.all(mass[1][~active]<1e-300)
                lows[~active,o['alpha']]=-mass[1][~active]
                highs[~active,o['alpha']]=mass[1][~active]
                lows[np.ix_(~active,o['beta'])]=-mass[1][~active,None]
                highs[np.ix_(~active,o['beta'])]=mass[1][~active,None]
    return lows,highs


def dual_upper(menu,entry,b,p,ts,dual):
    weights=[max(F(0),F(str(v))) for v in dual]
    total=sum(weights);assert total>0
    weights=[v/total for v in weights]
    ids=[i for i,v in enumerate(weights) if v]
    values=menu_values(menu,entry,b,p,[ts[i] for i in ids])
    wb=[iv.rational(weights[i]) for i in ids]
    factor=(np.array([v[0] for v in wb])[:,None],np.array([v[1] for v in wb])[:,None])
    average=signed_sum(signed_mul(values,factor),axis=0)
    best=np.zeros(len(menu.raw))
    for o in menu.options:
        positive=(np.maximum(0,average[0][o['beta']]),np.maximum(0,average[1][o['beta']]))
        gain=iv.positive_sum(positive)
        value=iv.add(gain,(average[0][o['alpha']],average[1][o['alpha']]))
        best[o['branch']]=max(best[o['branch']],float(value[1]))
    upper=iv.positive_sum((best,best))[1]
    return dict(upper=float(upper),test_parameters=[str(F(str(ts[i]))) for i in ids],
                dual_weights_exact=[str(weights[i]) for i in ids],
                arithmetic='Outward binary64 operations; gamma_n dot bounds; analytic atanh log remainder; exact feasible rational dual weights.')
