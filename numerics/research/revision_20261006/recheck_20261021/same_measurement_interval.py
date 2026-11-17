"""Exact all-order cost comparison on a nonzero Werner noise interval.

Only the single five-input measurement used in Figure 3b is compared.
This does not bound other lower-memory measurements or protocols.
"""
from pathlib import Path
from fractions import Fraction as F
from itertools import permutations
from math import comb
import csv, hashlib, json, sys
import numpy as np

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
CODE=PROJECT/'revised/reproduction/reproduction/controls/code'
sys.path.insert(0,str(CODE))
from fixed_block_scan import preprocess
from certify_fixed_blocks import H
import float_intervals as iv

def plus(*ps): return [sum(p[j] for p in ps) for j in range(6)]
def scale(p,c): return [v*c for v in p]
def sub(p,q): return [a-b for a,b in zip(p,q)]
def exact(p,x): return sum(a*x**i for i,a in enumerate(p))
def bernstein(p,a,b):
    n=len(p)-1
    t=[sum(p[i]*comb(i,j)*a**(i-j)*(b-a)**j for i in range(j,n+1)) for j in range(n+1)]
    return [sum(t[j]*F(comb(k,j),comb(n,j)) for j in range(k+1)) for k in range(n+1)]
def enclosure(p,a,b):
    coefficients=bernstein(p,a,b)
    return (iv.rational(min(coefficients))[0],iv.rational(max(coefficients))[1])

def main():
    graph=dict(N=5,edges=[[0,1],[0,4],[1,2],[2,3]],original_min_memory=2)
    n,w,subsets,counts,cuts,mmin=preprocess(graph)
    v=5;S=1<<n
    probs=[]
    for A in range(S):
        allowed=[u for u in range(S) if subsets[A,u] and (u&v).bit_count()%2==0]
        p=[F(0)]*6
        for u in allowed:p[int(w[0,u])]+=F(1,len(allowed))
        probs.append(p)
    orders=[]
    for order in permutations(range(n)):
        A=0;cost=[F(0)]*6;peak=1
        for i in order:
            cost=plus(cost,probs[A]);peak=max(peak,1+int(cuts[A]));A|=1<<i
        orders.append(dict(order=order,memory=peak,cost=cost))
    picks={2:(3,2,1,4,0),3:(3,2,4,0,1)}
    chosen={m:next(o for o in orders if o['order']==picks[m]) for m in picks}
    def transform(row,y):
        p=[F(0)]*6
        for u in range(S):p[int(w[row,u])]+=F((-1)**((u&y).bit_count()%2),S)
        return p
    P0,Pv,V0,Vv=transform(0,0),transform(0,v),transform(v,0),transform(v,v)
    raw=[scale(plus(P0,V0),F(1,2)),scale(plus(Pv,Vv),F(1,2)),
         scale(sub(Pv,Vv),F(1,2)),scale(sub(P0,V0),F(1,2))]
    accept=plus(P0,Pv)
    assert accept==probs[-1]
    # A substantial-rate comparison, rather than a neighborhood of a zero rate.
    left,right=F(3,20),F(1,5)
    a,b=1-2*right,1-2*left
    cost_checks=[]
    for m,c in chosen.items():
        eligible=[o for o in orders if o['memory']<=m]
        mins=[min(bernstein(sub(o['cost'],c['cost']),a,b)) for o in eligible]
        assert min(mins)>=0
        cost_checks.append(dict(memory=m,orders=len(eligible),nonnegative_difference_bernstein_min=str(min(mins))))
    diff=sub(chosen[2]['cost'],chosen[3]['cost'])
    assert min(bernstein(diff,a,b))>0
    at=F(3,5)
    C2=exact(chosen[2]['cost'],at)/exact(accept,at)
    C3=exact(chosen[3]['cost'],at)/exact(accept,at)
    assert C2==F(2975,173) and C3==F(11425,692)
    leaves=[]
    for j in range(100):
        ql=left+(right-left)*F(j,100);qr=left+(right-left)*F(j+1,100)
        aa,bb=1-2*qr,1-2*ql
        pa=enclosure(accept,aa,bb)
        rr=[enclosure(p,aa,bb) for p in raw]
        state=iv.div((np.array([[x[0] for x in rr]]),np.array([[x[1] for x in rr]])),
                     (np.array([[pa[0]]]),np.array([[pa[1]]])))
        hh=H(state);rlo=float(iv.down(1-hh[1][0]));assert rlo>0
        d2=enclosure(chosen[2]['cost'],aa,bb);d3=enclosure(chosen[3]['cost'],aa,bb)
        delta=enclosure(diff,aa,bb)
        gap=float(iv.down(iv.down(iv.down(pa[0]*rlo)*delta[0])/iv.up(d2[1]*d3[1])))
        ratio=float(iv.down(d2[0]/d3[1]));assert gap>0 and ratio>1
        lowrate=float(iv.down(iv.down(pa[0]*rlo)/d3[1]))
        leaves.append(dict(Q_left=str(ql),Q_right=str(qr),rate_3_lower=lowrate,
                           rate_gap_lower=gap,rate_ratio_lower=ratio,classical_coefficient_lower=rlo))
    output=HERE/'same_measurement_interval';output.mkdir(exist_ok=True)
    rows=[]
    for j in range(201):
        Q=left+(right-left)*F(j,200);lam=1-2*Q;p=float(exact(accept,lam))
        state=np.array([float(exact(vv,lam))/p for vv in raw])
        rc=1+float(np.sum(np.where(state>0,state*np.log2(state),0)))
        c2=float(exact(chosen[2]['cost'],lam))/p;c3=float(exact(chosen[3]['cost'],lam))/p
        rows.append(dict(Q=float(Q),R2=rc/c2,R3=rc/c3,C2=c2,C3=c3,ratio=c2/c3,gap=rc/c3-rc/c2))
    with (output/'curve.csv').open('w',newline='',encoding='utf-8') as f:
        cw=csv.DictWriter(f,fieldnames=list(rows[0]));cw.writeheader();cw.writerows(rows)
    report=dict(status='certified',source='Werner',Q_interval=[str(left),str(right)],
        input_N=5,graph=graph,dual=v,all_orders_checked=len(orders),order_checks=cost_checks,
        selected={str(m):dict(order=c['order'],cost_polynomial_lambda=list(map(str,c['cost']))) for m,c in chosen.items()},
        difference_polynomial_lambda=list(map(str,diff)),acceptance_polynomial_lambda=list(map(str,accept)),
        rate_gap_lower=min(v['rate_gap_lower'] for v in leaves),
        rate_ratio_lower=min(v['rate_ratio_lower'] for v in leaves),
        rate_3_lower=min(v['rate_3_lower'] for v in leaves),leaves=leaves,
        scope='One fixed complete terminal measurement, zero syndrome, all legal fixed loading orders, eager rejection, same one-way 1-H terminal. Not an upper bound on other low-memory measurements.',
        arithmetic='Exact rational cost polynomials and Bernstein nonnegativity; outward probability and analytic entropy enclosures on 100 exactly covering Q boxes.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'certificate.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('leaves','selected')},indent=2))

if __name__=='__main__':main()
