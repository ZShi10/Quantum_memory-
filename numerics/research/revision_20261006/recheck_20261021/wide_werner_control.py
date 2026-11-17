"""Tight polynomial Q boxes for the exhaustive matched Werner control.

Signed Walsh transforms are performed on exact integer coefficients first.
Only the resulting degree-N polynomials are enclosed; this avoids treating
canceling character terms as independent intervals. The protocol class and
the B/Watanabe menu are unchanged from matched_interval.py.
"""
from pathlib import Path
from fractions import Fraction as F
from math import comb
import argparse, functools, json, os, sys, time
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
ROOT=PROJECT/'research/revision_20261013'
sys.path[:0]=[str(ROOT/'numerics'),str(ROOT/'reproduction/controls/code')]
import numpy as np
import matched_interval as old
import float_intervals as iv
from fixed_block_scan import preprocess,fwht


@functools.lru_cache(None)
def transform(n,a,b):
    # c_i x^i -> Bernstein coefficients on the exact rational interval [a,b].
    values=[[sum(F(comb(i,j))*a**(i-j)*(b-a)**j*F(comb(k,j),comb(n,j))
                  for j in range(min(i,k)+1)) for k in range(n+1)]
            for i in range(n+1)]
    return tuple(np.array([[iv.rational(v)[side] for v in row] for row in values])
                 for side in (0,1))


def polynomial_bounds(coeff,a,b,probability=True):
    n=coeff.shape[-1]-1;T=transform(n,a,b)
    positive=np.maximum(0,coeff);negative=np.maximum(0,-coeff)
    # Each stored coefficient is a dyadic rational and exactly representable.
    first=iv.positive_dot((T[0].T,T[1].T),positive.T)
    second=iv.positive_dot((T[0].T,T[1].T),negative.T)
    lo=iv.down(first[0]-second[1]);hi=iv.up(first[1]-second[0])
    lower,upper=lo.min(0),hi.max(0)
    return (np.maximum(0,lower),np.minimum(1,upper)) if probability else (lower,upper)


def coefficients(g):
    n,w,subsets,counts,cuts,mmin=preprocess(g);S=1<<n
    raw=np.zeros((S-1,4,n+1));public=np.zeros((S-1,S,n+1))
    for degree in range(n+1):
        d=fwht((w==degree).astype(float))/S
        p00=d[0,0];p0v=d[0,1:];dv0=d[1:,0];dvv=np.diag(d)[1:]
        raw[:,:,degree]=np.stack((p00+dv0,p0v+dvv,p0v-dvv,p00-dv0),-1)/2
        terms=subsets*(w[0]==degree)[None,:]
        numerator=(terms.sum(1)[:,None]+fwht(terms))/2
        public[:,:,degree]=(numerator/counts).T[1:]
    assert np.all(public>=0)
    assert np.array_equal(raw.sum(1),public[:,-1,:])
    # Proves q_I is maximal for every lambda>=0, as required by the
    # closed-power concurrence cap in derivative_menu.
    assert np.all(raw[:,0:1,:]-raw[:,1:,:]>=0)
    return n,cuts,mmin,raw,public


def derivative_menu(co,left,right,cost,midcost,n):
    """Mean-value enclosure for the maximum of the fixed B-prefix rates.

    For any legal order, each prefix probability is a polynomial in lambda
    with nonnegative coefficients summing to one and degree <=n. Thus its
    Q derivative is in [-2n,0], and every attempt-cost derivative is in
    [-2n(n-1),0]. This also bounds the piecewise derivative of the minimum
    over fixed orders. No smoothness of the maximizing order is assumed.
    """
    sys.path.insert(0,str(PROJECT/'research/revision_20260922'))
    import interval_jets as j
    a,b=1-2*right,1-2*left;mid=(a+b)/2
    shape=co.shape[:2]
    c=co.reshape(-1,n+1)
    value=tuple(x.reshape(shape) for x in polynomial_bounds(c,a,b))
    deriv=tuple(x.reshape(shape) for x in polynomial_bounds(-2*c[:,1:]*np.arange(1,n+1),a,b,False))
    point=tuple(x.reshape(shape) for x in polynomial_bounds(c,mid,mid))
    raw=j.Jet(value,deriv);raw0=j.Jet(point)
    mass=raw.sum().clip();mass0=raw0.sum().clip()
    p=(raw/mass[...,None]).clip();p0=(raw0/mass0[...,None]).clip()
    numerator=polynomial_bounds(co[:,0,:]-co[:,3,:],a,b)
    denominator=polynomial_bounds(co[:,0,:]+co[:,3,:],a,b)
    bias=np.minimum(1,iv.up(numerator[1]/denominator[0]))
    bias2=iv.up(bias*bias)
    D=j.Jet(cost,(np.full(len(c)//4,-2*n*(n-1)),np.zeros(len(c)//4)))
    D0=j.Jet(midcost)
    y=j.Jet((np.ones(shape[0]),np.ones(shape[0])));y0=j.Jet((np.ones(shape[0]),np.ones(shape[0])))
    bestlo=np.zeros(shape[0]);besthi=np.zeros(shape[0]);choice=np.zeros(shape[0],int)
    radius=float(iv.rational((right-left)/2)[1])
    for B in range(17):
        r=mass*y*j.watanabe(p)/D;r0=mass0*y0*j.watanabe(p0)/D0
        slope=np.maximum(np.abs(r.d[0]),np.abs(r.d[1]));error=iv.up(radius*slope)
        lo=np.maximum(0,np.maximum(r.v[0],iv.down(r0.v[0]-error)))
        hi=np.minimum(r.v[1],iv.up(r0.v[1]+error))
        hi=np.minimum(hi,iv.up(iv.up(mass.v[1]/cost[0])*2.**(-B)))
        # In a zero-flag Werner graph block q_I is maximal. If q_I<=1/2
        # its Bell state is separable. Otherwise, after length ell=2**B,
        # concurrence=(c0**ell-beta**ell)/(1+beta**ell)<=c0**ell.
        # The Bell relative-entropy bound is <=concurrence**2. This avoids
        # spurious long-B upper rates caused by interval normalization.
        hi=np.minimum(hi,iv.up(iv.up(iv.up(mass.v[1]/cost[0])*2.**(-B))*bias2))
        assert np.all(lo<=hi+1e-14)
        take=lo>bestlo;choice[take]=B
        bestlo=np.maximum(bestlo,lo);besthi=np.maximum(besthi,hi)
        p,s=j.step(p,p,'Z');p0,s0=j.step(p0,p0,'Z')
        y=(y*s/2).clip(0,2.**(-B-1));y0=(y0*s0/2).clip(0,2.**(-B-1))
        bias2=np.minimum(1,iv.up(bias2*bias2))
    return bestlo,besthi,choice


def evaluate(left,right,catalog,winner):
    a,b=1-2*right,1-2*left
    powers=(np.array([iv.rational(a**i)[0] for i in range(9)]),
            np.array([iv.rational(b**i)[1] for i in range(9)]))
    mid=(a+b)/2
    mpowers=tuple(np.array([iv.rational(mid**i)[k] for i in range(9)]) for k in (0,1))
    U=0.;L=None;cases=0;worst=None;meta=None
    for name,g in catalog.items():
        n,cuts,mmin,co,public=coefficients(g)
        S=1<<n;cases+=S-1
        if mmin>3 and name!=winner['graph']:continue
        # transpose: polynomial_bounds consumes (degree, Bernstein-index).
        raw=polynomial_bounds(co.reshape(-1,n+1),a,b)
        raw=tuple(x.reshape(S-1,4) for x in raw)
        acc=iv.positive_sum(raw)
        assert np.all(acc[0]>0)
        p=old.clean(iv.div(raw,(acc[0][:,None],acc[1][:,None])))
        value=iv.positive_dot((powers[0][:n+1],powers[1][:n+1]),public.reshape(-1,n+1).T)
        probs=tuple(x.reshape(S-1,S) for x in value)
        probs[0][:,0]=1.;probs[1][:,0]=1.
        mv=iv.positive_dot((mpowers[0][:n+1],mpowers[1][:n+1]),public.reshape(-1,n+1).T)
        mprobs=tuple(x.reshape(S-1,S) for x in mv)
        mprobs[0][:,0]=1.;mprobs[1][:,0]=1.
        if mmin<=3:
            costs=old.minimum_cost_lower(cuts,probs[0],3)
            midcosts=old.minimum_cost_lower(cuts,mprobs[0],3)
            _,rate,_=derivative_menu(co,left,right,(costs,np.full_like(costs,n)),(midcosts,np.full_like(costs,n)),n)
            j=int(rate.argmax())
            if rate[j]>U:U=float(rate[j]);worst=dict(graph=name,dual=j+1,N=n)
        if name==winner['graph']:
            v=winner['logical_dual']-1;mask=0;cost=0.;clo=0.;cmlo=0.;cmhi=0.
            for i in winner['input_order']:
                cost=float(iv.up(cost+probs[1][v,mask]));clo=float(iv.down(clo+probs[0][v,mask]))
                cmlo=float(iv.down(cmlo+mprobs[0][v,mask]));cmhi=float(iv.up(cmhi+mprobs[1][v,mask]));mask|=1<<i
            lows,_,choice=derivative_menu(co[v:v+1],left,right,(np.array([clo]),np.array([cost])),(np.array([cmlo]),np.array([cmhi])),n)
            L=float(lows[0])
            meta=dict(graph=name,dual=v+1,N=n,memory=4,input_order=winner['input_order'],B=int(choice[0]))
    assert cases==56000 and L is not None
    return dict(Q_left=str(left),Q_right=str(right),low_memory_upper=U,
                high_memory_lower=L,gap=float(iv.down(L-U)),cases=cases,
                witness=meta,largest_upper_case=worst)


def main(args):
    catalog=json.loads((ROOT/'reproduction/controls/data/graph_representatives.json').read_bytes())['graphs']
    winner=next(w for w in json.loads((ROOT/'reproduction/controls/data/executable_control_winners.json').read_bytes()) if w['peak_memory']==4)
    left,right=F(args.left),F(args.right);start=time.monotonic()
    folder=HERE/'wide_werner_derivative_v2';folder.mkdir(exist_ok=True)
    rows=[]
    for j in range(args.cells):
        lo=left+(right-left)*F(j,args.cells);hi=left+(right-left)*F(j+1,args.cells)
        file=folder/(str(lo).replace('/','-')+'_'+str(hi).replace('/','-')+'.json')
        if file.exists():row=json.loads(file.read_bytes())
        else:
            row=evaluate(lo,hi,catalog,winner)
            file.write_text(json.dumps(row,indent=2)+'\n')
        rows.append(row);print('WIDE CONTROL',row,flush=True)
    out=dict(complete=True,strict=all(x['gap']>0 for x in rows),
             interval=[str(left),str(right)],cells=rows,seconds=time.monotonic()-start,
             scope='All N<=8 graph representatives, all duals, zero flag, all fixed eager-loading orders; m<=3 upper versus an N=6 m=4 witness; common B<=16 and Watanabe.')
    (folder/'latest.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--left',required=True);p.add_argument('--right',required=True)
    p.add_argument('--cells',type=int,default=1);main(p.parse_args())
