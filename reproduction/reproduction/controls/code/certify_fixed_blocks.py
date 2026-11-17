"""Exact integer stopping DP plus outward entropy enclosures, Werner Q=7/25.

Replays the entire explicitly frozen graph/dual zero-syndrome control class.
No floating optimum is promoted to a bound: probabilities and attempt costs are
integer rational quantities, and every evaluated classical menu action is bounded.
The graph-representative completeness reduction is inherited from the supplied
catalog; all 56000 graph/dual pairs are evaluated without rate-based compression.
"""
from pathlib import Path
import json,csv,time,math
import numpy as np
from numba import njit
import float_intervals as iv
from fixed_block_scan import preprocess,fwht
ROOT=Path(__file__).resolve().parents[1]

@njit(cache=True)
def integer_fwht(a):
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
def integer_dp(cuts,pnums,mmax):
    V,S=pnums.shape;n=0
    while (1<<n)<S:n+=1
    huge=np.int64(4_000_000_000_000_000_000)
    out=np.full((V,mmax),huge,dtype=np.int64)
    for v in range(V):
        for m in range(1,mmax+1):
            dp=np.full(S,huge,dtype=np.int64);dp[0]=0
            for a in range(1,S):
                if a!=S-1 and cuts[a]>=m:continue
                best=huge
                for i in range(n):
                    if (a>>i)&1:
                        b=a^(1<<i)
                        if dp[b]!=huge:
                            d=dp[b]+pnums[v,b]
                            if d<best:best=d
                dp[a]=best
            out[v,m-1]=dp[-1]
    return out

def ratio_int(a,b):
    # These anchor counts are <2**53 and convert to binary64 exactly.
    assert np.max(np.abs(a))<2**53 and np.max(np.abs(b))<2**53
    value=np.asarray(a,dtype=float)/np.asarray(b,dtype=float)
    return np.maximum(0,iv.down(value)),iv.up(value)

def clean(p):return np.maximum(0,p[0]),np.minimum(1,p[1])
def H(p):
    a,b=iv.entropy_interval(clean(p))
    return iv.down(a.sum(axis=-1)-8*np.finfo(float).eps),iv.up(b.sum(axis=-1)+8*np.finfo(float).eps)
def h2(p):return H((np.stack([p[0],iv.down(1-p[1])],axis=-1),np.stack([p[1],iv.up(1-p[0])],axis=-1)))
def take(p,i):return p[0][:,i],p[1][:,i]
def psum(p):return iv.positive_sum(p)
def maximum(vals):return np.maximum.reduce([v[0] for v in vals]),np.maximum.reduce([v[1] for v in vals])
def bstep(p):
    a,b,c,d=[take(p,i) for i in range(4)]
    q=[iv.add(iv.mul(a,a),iv.mul(d,d)),iv.add(iv.mul(b,b),iv.mul(c,c)),iv.scale(iv.mul(b,c),2),iv.scale(iv.mul(a,d),2)]
    q=(np.maximum(0,np.stack([x[0] for x in q],axis=-1)),np.stack([x[1] for x in q],axis=-1));s=psum(q)
    return clean(iv.div(q,(s[0][:,None],s[1][:,None]))),s

def terminal_bounds(p,Bmax=16):
    num=len(p[0]);y=(np.ones(num),np.ones(num));best=(np.zeros(num),np.zeros(num));choice=np.zeros(num,dtype=int)
    zero=(np.zeros(num),np.zeros(num));one=(np.ones(num),np.ones(num))
    for j in range(Bmax+1):
        a=clean(iv.add(take(p,0),take(p,3)));b=clean(iv.add(take(p,1),take(p,2)));ab=iv.mul(a,b)
        product=clean(iv.add(iv.mul(take(p,0),take(p,1)),iv.mul(take(p,3),take(p,2))))
        ratio=clean(iv.div(product,(np.maximum(ab[0],np.nextafter(0.,1.)),np.maximum(ab[1],np.nextafter(0.,1.)))))
        h=h2(ratio);term=iv.mul(ab,(np.maximum(0,h[0]),np.maximum(0,h[1])))
        first=iv.add(iv.sub(one,H(p)),term)
        qb,s=bstep(p)
        gain=iv.sub(one,H(qb));second=(iv.down(s[0]*np.maximum(0,gain[0])/2),iv.up(s[1]*np.maximum(0,gain[1])/2))
        w=maximum([zero,first,second]);w=(np.maximum(0,w[0]),np.minimum(1,w[1]))
        rate=iv.mul(y,w);sel=rate[0]>best[0];choice[sel]=j
        best=maximum([best,rate]);y=iv.scale(iv.mul(y,s),.5);p=qb
    return best,choice

def certify():
    start=time.time();cat=json.loads((ROOT/'data/graph_representatives.json').read_text())['graphs']
    state_counts=[];paccs=[];costs=[];meta=[];counts_checked=0
    for name,g in cat.items():
        n,weights,subsets,counts,cuts,mmin=preprocess(g);S=1<<n;den=25**n
        # F_v(u)=(11/25)^weight; exact common-denominator character transform.
        powers=np.asarray([11**w*25**(n-w) for w in range(n+1)],dtype=np.int64)
        factors=powers[weights];D=integer_fwht(factors);P=D[0]
        sf=(subsets.astype(np.int64)*factors[0][None,:])
        nums=sf.sum(axis=1)[:,None]+integer_fwht(sf)
        twocount=(2*counts).astype(np.int64)
        # p_A = nums/(twocount*25**N). Use common denom S*25**N.
        assert np.all(((2*S)%twocount)==0)
        pn=(nums*((2*S)//twocount)).T
        assert np.all(pn>=0) and np.all(pn<=2*S*den)
        dc=integer_dp(cuts,pn[1:],4)
        for v in range(1,S):
            q=np.asarray([P[0]+D[v,0],P[v]+D[v,v],P[v]-D[v,v],P[0]-D[v,0]],dtype=np.int64)
            assert np.all(q>=0)
            pa=P[0]+P[v]
            assert sum(q)==2*pa
            counts_checked+=1
            # Separable Bell outputs cannot yield secret key under any AD.
            if 2*np.max(q)<=np.sum(q):continue
            state_counts.append(q);paccs.append(2*pa);costs.append(dc[v-1]);meta.append({'graph':name,'N':n,'dual':v,'minimum_memory':mmin})
    q=np.asarray(state_counts,dtype=np.int64);c=np.asarray(costs,dtype=np.int64);pa=np.asarray(paccs,dtype=np.int64)
    p=ratio_int(q,q.sum(axis=1)[:,None]);bounds,choices=terminal_bounds(p)
    rows=[];winning={}
    for N in range(1,9):
        for m in range(1,5):
            allowed=np.array([x['N']<=N and x['minimum_memory']<=m for x in meta])
            if not np.any(allowed):lower=upper=0.;winner=None
            else:
                ids=np.flatnonzero(allowed)
                y=ratio_int(pa[ids],c[ids,m-1]);rates=iv.mul(y,(bounds[0][ids],bounds[1][ids]))
                ii=int(np.argmax(rates[0]));lower=float(rates[0][ii]);upper=float(np.max(rates[1]));k=int(ids[ii])
                winner={**meta[k],'B_rounds':int(choices[k]),'Bell_counts':q[k].tolist(),'acceptance_numerator':int(pa[k]),'attempt_cost_numerator':int(c[k,m-1]),'common_denominator':2*(1<<meta[k]['N'])*25**meta[k]['N']}
            rows.append({'Q':'7/25','Nmax':N,'memory':m,'lower':lower,'upper':upper});winning[f'N{N}_m{m}']=winner
    with (ROOT/'data/fixed_block_interval_bounds.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    rec={'status':'complete_interval_evaluation_of_frozen_zero_syndrome_control_family','Q':'7/25',
      'graph_count':len(cat),'graph_dual_pairs':counts_checked,'nonseparable_outputs':len(meta),
      'B_max':16,'classical_terminal':'Watanabe','arithmetic':'Integer Walsh transforms and exact integer stopping DP; outward binary64 operations and atanh-series entropy bounds; no unsafe fast-math.',
      'scope':'All catalog graph representatives N<=8 and every nonzero logical dual. Only zero relative syndrome retained; all legal deterministic eager-stopping input orders. No arbitrary ED, no arbitrary retained subsets, and no general-code terminal optimum.',
      'catalog_coverage':'Graph representatives supplied by source; cross-cut ranks and all costs recomputed independently.',
      'winners':winning,'seconds':time.time()-start}
    (ROOT/'verification/fixed_block_interval_certificate.json').write_text(json.dumps(rec,indent=2))
    print('DONE',counts_checked,'graph-duals,',len(meta),'entangled outputs',round(time.time()-start,2),'seconds')
    for r in rows:
        if r['Nmax'] in [6,8]:print(r)
    # Strict fixed-input-size and larger-input low-memory comparisons.
    a=next(x for x in rows if x['Nmax']==6 and x['memory']==4)
    b=next(x for x in rows if x['Nmax']==8 and x['memory']==3)
    assert a['lower']>b['upper']
    print('SIX-input FOUR-memory lower exceeds ALL N<=8 THREE-memory upper by',a['lower']-b['upper'])

if __name__=='__main__':certify()
