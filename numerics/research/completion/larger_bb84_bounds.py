"""All-candidate BB84 minimax upper bounds for the frozen m=15..20 family.

At a single compatible source, any processing after an accepted ED branch
is bounded by its Bell relative entropy, in turn bounded by concurrence^2.
The bound allows arbitrary subsequent LOCC, not only the searched AD codes.
It can exclude a proposed larger circuit without evaluating long codes.
"""
from common import *
from fractions import Fraction as F
import json
import numpy as np
import float_enclosures as iv
from algebra import compositions
from libraries import code_kernels


def bounds(entries,b,p,t,ks,target):
    z=min(b,p)*t;chars=[F(1),1-2*p,1-2*(b+p)+4*z,1-2*b];out=[]
    for m in sorted({e['m'] for e in entries}):
        es=[e for e in entries if e['m']==m];powers=[]
        for c in chars:
            pp=[(np.array(1.),np.array(1.))]
            for _ in range(m):pp.append(iv.mul(pp[-1],iv.rational(c)))
            powers.append(pp)
        mon=[]
        for typ in compositions(m):
            v=(np.array(1.),np.array(1.))
            for i,n in enumerate(typ):v=iv.mul(v,powers[i][n])
            mon.append(v)
        mon=(np.array([[x[0] for x in mon]]),np.array([[x[1] for x in mon]]))
        counts=np.array([e['characters'] for e in es]).reshape(-1,len(mon[0][0])).T
        a=iv.positive_dot(mon,np.maximum(counts,0));c=iv.positive_dot(mon,np.maximum(-counts,0));v=iv.sub(a,c)
        raw=(np.maximum(0,v[0].reshape(-1,4))/(2**(m+1)),np.maximum(0,v[1].reshape(-1,4))/(2**(m+1)))
        ps=iv.positive_sum(raw);num=iv.up(2*raw[1].max(1)-ps[0]);num=np.maximum(0,num)
        upper=iv.up(iv.up(num*num)/iv.down(ps[0]*m))
        finite=upper.copy();active=np.flatnonzero(upper>target)
        if len(active):
            pp=iv.div((raw[0][active],raw[1][active]),(ps[0][active,None],ps[1][active,None]))
            _,rate_hi,_=iv.best_code_bounds(pp,ks)
            finite[active]=iv.up(iv.up(ps[1][active]*rate_hi)/m)
        out.extend(dict(ED=e['id'],m=m,M=e['M'],upper=float(u),single_AD_n7_upper=float(f)) for e,u,f in zip(es,upper,finite))
    return out


def main():
    entries=json.loads((DATA/'memory_eight_graphs.json').read_text())['entries'];results=[];ks=code_kernels()
    for path in sorted(DATA.glob('bb84_upper_*.json')):
        src=json.loads(path.read_text());b,p=F(src['eb']),F(src['ep'])
        for mode in ('real_M2','full_M2'):
            v=src['results'][mode];t=F(v['test_t']);records=bounds(entries,b,p,t,ks,v['upper']);modes={}
            for M in range(2,9):
                ub=max((x['upper'] for x in records if x['M']<=M),default=0.)
                short=max((x['single_AD_n7_upper'] for x in records if x['M']<=M),default=0.)
                modes[f'M{M}']=dict(new_family_arbitrary_postprocessing_upper=ub,new_family_single_AD_n7_upper=short,
                                    union_single_AD_n7_upper=max(v['upper'],short))
            results.append(dict(eb=float(b),ep=float(p),mode=mode,test_t=str(t),budgets=modes,all_candidates=records))
            print('Larger BB bound',float(b),float(p),mode,'M8',modes['M8'],flush=True)
    (DATA/'larger_bb84_bounds.json').write_text(json.dumps(dict(status='passed',scope='Frozen 108 zero-syndrome graph instruments with m=15..20 and actual memory through8. Certified all-candidate rate upper at one admissible BB84 source; permits arbitrary post-ED LOCC. Union bounds refer to the complete m<=4, n<=7 single-round pilot core plus this new family. Other m=5..14 constructions are separate achievable records.',results=results),separators=(',',':')))


if __name__=='__main__':main()
