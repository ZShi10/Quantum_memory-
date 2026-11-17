"""A uniform-in-Q separation for the common small-block Werner control.

Outward interval Walsh transforms enclose every graph/dual state and every
fixed-order stopping cost. A SINGLE six-input four-slot witness is compared
against the union of every <=8-input, <=3-slot control throughout a Q interval.
No interpolation between isolated Q certificates is used.
"""
from pathlib import Path
from fractions import Fraction as F
import json
import sys
import time
import numpy as np
from numba import njit
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reproduction/controls/code'))
import float_intervals as iv
from fixed_block_scan import preprocess
from certify_fixed_blocks import clean,H,h2,take,maximum

def walsh(a):
    lo=a[0].copy();hi=a[1].copy();h=1
    while h<lo.shape[-1]:
        for i in range(0,lo.shape[-1],2*h):
            l0=lo[...,i:i+h].copy();u0=hi[...,i:i+h].copy()
            l1=lo[...,i+h:i+2*h].copy();u1=hi[...,i+h:i+2*h].copy()
            lo[...,i:i+h]=iv.down(l0+l1);hi[...,i:i+h]=iv.up(u0+u1)
            lo[...,i+h:i+2*h]=iv.down(l0-u1);hi[...,i+h:i+2*h]=iv.up(u0-l1)
        h*=2
    return lo,hi

@njit(cache=True)
def minimum_cost_lower(cuts,probabilities,memory):
    V,S=probabilities.shape;n=0
    while (1<<n)<S:n+=1
    out=np.full(V,np.inf)
    for v in range(V):
        dp=np.full(S,np.inf);dp[0]=0.
        for a in range(1,S):
            if a!=S-1 and cuts[a]>=memory:continue
            for i in range(n):
                if (a>>i)&1:
                    prev=a^(1<<i)
                    if np.isfinite(dp[prev]):
                        cost=np.nextafter(dp[prev]+probabilities[v,prev],-np.inf)
                        if cost<dp[a]:dp[a]=cost
        out[v]=dp[-1]
    return out

def menu_enclosure(p):
    n=len(p[0]);y=(np.ones(n),np.ones(n));best=(np.zeros(n),np.zeros(n));choice=np.zeros(n,dtype=int)
    zero=(np.zeros(n),np.zeros(n));one=(np.ones(n),np.ones(n))
    # Zero-syndrome Werner blocks have P(error=0)>=P(error=v), hence a>=b.
    # Closed B powers avoid dependency blow-up from 16 normalized recursions.
    aa=clean(iv.add(take(p,0),take(p,3)));aa=(np.maximum(.5,aa[0]),aa[1])
    bb=clean(iv.add(take(p,1),take(p,2)));bb=(bb[0],np.minimum(.5,bb[1]))
    beta=clean(iv.div(bb,aa))
    def signed_ratio(num,den):
        safe=(np.maximum(np.nextafter(0.,1.),den[0]),np.maximum(np.nextafter(0.,1.),den[1]))
        with np.errstate(over='ignore',invalid='raise'):
            quotients=np.stack([num[i]/safe[j] for i in (0,1) for j in (0,1)])
        return np.maximum(-1,iv.down(quotients.min(axis=0))),np.minimum(1,iv.up(quotients.max(axis=0)))
    c0=signed_ratio(iv.sub(take(p,0),take(p,3)),aa)
    c1=signed_ratio(iv.sub(take(p,1),take(p,2)),bb)
    c1=(np.where(bb[0]<=0,-1,c1[0]),np.where(bb[0]<=0,1,c1[1]))
    def square(v):
        lower=np.minimum(v[0]*v[0],v[1]*v[1]);upper=np.maximum(v[0]*v[0],v[1]*v[1])
        return np.where((v[0]<=0)&(v[1]>=0),0,np.maximum(0,iv.down(lower))),np.minimum(1,iv.up(upper))
    for j in range(17):
        a=clean(iv.add(take(p,0),take(p,3)));b=clean(iv.add(take(p,1),take(p,2)));ab=iv.mul(a,b)
        numerator=clean(iv.add(iv.mul(take(p,0),take(p,1)),iv.mul(take(p,3),take(p,2))))
        ratio=clean(iv.div(numerator,(np.maximum(ab[0],np.nextafter(0.,1.)),np.maximum(ab[1],np.nextafter(0.,1.)))))
        h=h2(ratio);term=iv.mul(ab,(np.maximum(0,h[0]),np.maximum(0,h[1])))
        first=iv.add(iv.sub(one,H(p)),term)
        next_beta=square(beta);c0=square(c0);c1=square(c1)
        denominator=iv.scale(iv.add(one,next_beta),2)
        plus0=iv.add(one,c0);minus0=iv.sub(one,c0);minus0=(np.maximum(0,minus0[0]),minus0[1])
        plus1=iv.add(one,c1);minus1=iv.sub(one,c1);minus1=(np.maximum(0,minus1[0]),minus1[1])
        raw=[plus0,iv.mul(next_beta,plus1),iv.mul(next_beta,minus1),minus0]
        qb=clean(iv.div((np.stack([x[0] for x in raw],axis=-1),np.stack([x[1] for x in raw],axis=-1)),
                        (denominator[0][:,None],denominator[1][:,None])))
        old_normalization=iv.add(one,beta)
        s=iv.div(iv.add(one,next_beta),iv.mul(old_normalization,old_normalization))
        s=(np.maximum(.5,s[0]),np.minimum(1.,s[1]))
        assert np.all(s[0]<=s[1])
        beta=next_beta
        gain=iv.sub(one,H(qb))
        second=(iv.down(s[0]*np.maximum(0,gain[0])/2),iv.up(s[1]*np.maximum(0,gain[1])/2))
        w=maximum([zero,first,second]);w=(np.maximum(0,w[0]),np.minimum(1,w[1]))
        rate=iv.mul(y,w);sel=rate[0]>best[0];choice[sel]=j;best=maximum([best,rate])
        y=iv.scale(iv.mul(y,s),F(1,2))
        # Every B round retains at most half its input; this exact bound also
        # controls interval overestimation after many squarings.
        y=(np.maximum(0,y[0]),np.minimum(2.**(-j-1),y[1]))
        p=qb
    return best,choice

def evaluate(qleft,qright,catalog,winner):
    rlo=iv.rational(1-2*qright)[0];rhi=iv.rational(1-2*qleft)[1]
    power=[(np.array(1.),np.array(1.))]
    for i in range(8):power.append(iv.mul(power[-1],(rlo,rhi)))
    U=0.;L=None;bound_winner=None;cases=0;high_meta=None
    for name,g in catalog.items():
        n,weights,subsets,counts,cuts,mmin=preprocess(g);S=1<<n
        powers=(np.array([x[0] for x in power]),np.array([x[1] for x in power]))
        factors=(powers[0][weights],powers[1][weights])
        d=walsh(factors);d=(iv.down(d[0]/S),iv.up(d[1]/S))
        p0=(d[0][0],d[1][0])
        acceptance=clean(iv.add((p0[0][0],p0[1][0]),(p0[0][1:],p0[1][1:])))
        diagonal=(np.diag(d[0])[1:],np.diag(d[1])[1:])
        vv=(d[0][1:,0],d[1][1:,0])
        raw=[iv.add((p0[0][0],p0[1][0]),vv),iv.add((p0[0][1:],p0[1][1:]),diagonal),
             iv.sub((p0[0][1:],p0[1][1:]),diagonal),iv.sub((p0[0][0],p0[1][0]),vv)]
        raw=(np.maximum(0,iv.down(np.stack([x[0] for x in raw],axis=-1)/2)),
             np.maximum(0,iv.up(np.stack([x[1] for x in raw],axis=-1)/2)))
        assert np.all(acceptance[0]>0)
        p=clean(iv.div(raw,(acceptance[0][:,None],acceptance[1][:,None])))
        vals=(subsets*factors[0][0][None,:],subsets*factors[1][0][None,:])
        transformed=walsh(vals);summed=iv.positive_sum(vals)
        numerator=iv.add((summed[0][:,None],summed[1][:,None]),transformed)
        probabilities=clean((iv.down(numerator[0]/(2*counts)),iv.up(numerator[1]/(2*counts))))
        probabilities=(probabilities[0].T[1:],probabilities[1].T[1:])
        probabilities[0][:,0]=1.;probabilities[1][:,0]=1.
        reward,choice=menu_enclosure(p);cases+=S-1
        if mmin<=3:
            costs=minimum_cost_lower(cuts,probabilities[0],3)
            rate_upper=iv.up(iv.up(acceptance[1]/costs)*reward[1])
            j=int(np.argmax(rate_upper))
            if rate_upper[j]>U:U=float(rate_upper[j]);bound_winner=dict(graph=name,dual=j+1,N=n)
        if name==winner['graph']:
            v=winner['logical_dual']-1;mask=0;cost=0.
            for input_ in winner['input_order']:
                cost=float(iv.up(cost+probabilities[1][v,mask]));mask|=1<<input_
            L=max(0.,float(iv.down(iv.down(acceptance[0][v]/cost)*reward[0][v])))
            high_meta=dict(graph=name,dual=v+1,N=n,memory=4,input_order=winner['input_order'],
                B_rounds=int(choice[v]),classical='Watanabe',attempt_cost_upper=cost,
                acceptance_lower=float(acceptance[0][v]),classical_yield_lower=float(reward[0][v]))
    assert L is not None and cases==56000
    return dict(Q_left=str(qleft),Q_right=str(qright),low_memory_upper=U,high_memory_lower=L,
                strict_gap_lower=float(iv.down(L-U)),cases=cases,high_policy=high_meta,largest_upper_case=bound_winner)

def main():
    start=time.monotonic();cat=json.loads((ROOT/'reproduction/controls/data/graph_representatives.json').read_text())['graphs']
    winner=next(w for w in json.loads((ROOT/'reproduction/controls/data/executable_control_winners.json').read_text()) if w['peak_memory']==4)
    trials=[]
    for digits in range(4,10):
        half=F(1,10**digits)
        row=evaluate(F(7,25)-half,F(7,25)+half,cat,winner);trials.append(row)
        print('Uniform Q box',row['Q_left'],row['Q_right'],'gap',row['strict_gap_lower'],flush=True)
        if row['strict_gap_lower']>0:break
    else:raise RuntimeError('No uniform separation closed at the attempted precision')
    report=dict(status='passed',certificate=row,trials=trials,elapsed_seconds=time.monotonic()-start,
        arithmetic='Outward binary64 Walsh transforms, probability bounds, and minimum-cost DP; closed B-power intervals; analytic atanh-series entropy enclosure; no fast-math.',
        scope='All 290 graph representatives, every nonzero dual, zero syndrome, all fixed eager-stopping input orders, N<=8/m<=3 upper; one fixed N=6/m=4 witness; common B<=16 and Watanabe menu.',
        interpretation='The entire reported closed Q interval is enclosed directly. No interpolation and no unrestricted ED/AD capacity claim.')
    (ROOT/'validation/matched_noise_interval.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
