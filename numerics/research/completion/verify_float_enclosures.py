from common import *
import json,time
from decimal import Decimal as D,localcontext
from fractions import Fraction as F
import numpy as np
import float_enclosures as iv
from libraries import code_kernels
from algebra import compositions


def main():
    start=time.time();rng=np.random.default_rng(429618)
    values=np.concatenate((np.exp(rng.uniform(-740,0,2000)),np.ldexp(np.ones(100),np.arange(-99,1)),np.array([np.nextafter(0.,1.),1.,.5,1/3])))
    lo,hi=iv.log2_positive(values)
    with localcontext() as ctx:
        ctx.prec=100;ln2=D(2).ln()
        for x,a,b in zip(values,lo,hi):
            exact=D(float(x)).ln()/ln2
            assert D(float(a))<=exact<=D(float(b)),(x,a,str(exact),b)
        peak=1/(D(1).exp()*ln2);assert peak<D('0.5307378454230432')
        ks=code_kernels();cases=[[F(1,2),F(1,6),F(1,6),F(1,6)],
              [F(3,5),F(1,10),F(1,20),F(1,4)],[F(1),F(0),F(0),F(0)],
              [F(1,2),F(0),F(0),F(1,2)],[F(97,100),F(1,100),F(1,100),F(1,100)]]
        verified=0;worst=0.
        for case in cases:
            bounds=[iv.rational(x) for x in case];p=(np.array([[v[0] for v in bounds]]),np.array([[v[1] for v in bounds]]))
            powers={n:iv.monomial_bounds(p,np.array(compositions(n))) for n in range(1,8)}
            dp=[D(v.numerator)/D(v.denominator) for v in case]
            def h(x):return -x*x.ln()/ln2 if x else D(0)
            for k in ks:
                mono=[]
                for typ in k.types:
                    value=D(1)
                    for x,n in zip(dp,typ):
                        if n:value*=x**int(n)
                    mono.append(value)
                joints=[sum((D(int(c))*mono[j] for j,c in enumerate(row) if c),D(0)) for row in k.polys]
                total=D(0)
                for hist,mult in zip(k.hist,k.multiplicity):
                    ps=sum((D(int(c))*x for c,x in zip(hist,joints) if c),D(0))
                    g=k.k*ps+h(ps)-sum((D(int(c))*h(x) for c,x in zip(hist,joints) if c),D(0))
                    total+=int(mult)*max(D(0),g)
                total/=k.n
                a,b=iv.code_bounds(powers[k.n],k);a=float(a[0]);b=float(b[0])
                # A 100-digit reference can straddle exact zero by 1e-99.
                assert D(a)-D('1e-90')<=total<=D(b)+D('1e-90'),(case,k.entry['id'],a,str(total),b)
                worst=max(worst,b-a);verified+=1
    report=dict(status='passed',logarithm_comparisons=len(values),code_rate_comparisons=verified,
                reference='100-digit Decimal, independent direct entropy sums',max_code_interval_width=worst,seconds=time.time()-start)
    (DATA/'float_enclosure_validation.json').write_text(json.dumps(report,indent=2));print(report,flush=True)


if __name__=='__main__':main()
