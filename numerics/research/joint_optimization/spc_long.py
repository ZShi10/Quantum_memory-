"""Single parity-check AD by phase-complement type orbits, O(n^3)."""
from math import lgamma
from functools import lru_cache
import numpy as np
from algebra import compositions
from rates import xlog2x

@lru_cache(None)
def table(n):
    types=np.asarray(compositions(n));syndrome=(types[:,1]+types[:,2])%2
    logmult=np.array([lgamma(n+1)-sum(lgamma(int(j)+1) for j in t) for t in types])
    return types,syndrome,logmult

def spc_rate(p,n):
    p=np.asarray(p,dtype=float);types,syndrome,logmult=table(n)
    with np.errstate(divide='ignore',invalid='ignore'):
        lp=np.log(p)
        first=np.where(types>0,types*lp,0.).sum(1)
        second=np.where(types>0,types*lp[[3,2,1,0]],0.).sum(1)
    logw=np.logaddexp(first,second)
    mass=np.exp(logmult+logw-np.log(2.))
    with np.errstate(invalid='ignore'):
        h=np.bincount(syndrome,weights=np.where(mass>0,-mass*logw/np.log(2.),0.),minlength=2)
    ps=np.bincount(syndrome,weights=mass,minlength=2)
    gain=(n-1)*ps-h-xlog2x(ps)
    return float(np.maximum(gain,0.).sum()/n)

def spc_rate_decimal(p,n,precision=100):
    """Independent high-precision type-orbit entropy sum."""
    from decimal import Decimal,localcontext
    from math import factorial
    with localcontext() as ctx:
        ctx.prec=precision
        p=[Decimal(str(x)) for x in p];ln2=Decimal(2).ln()
        powers=[[v**r if r else Decimal(1) for r in range(n+1)] for v in p]
        ps=[Decimal(0),Decimal(0)];h=[Decimal(0),Decimal(0)]
        facts=[factorial(i) for i in range(n+1)]
        for typ in compositions(n):
            a=b=Decimal(1);den=1
            for j,t in enumerate(typ):a*=powers[j][t];b*=powers[3-j][t];den*=facts[t]
            w=a+b
            if not w:continue
            mass=Decimal(facts[n]//den)*w/2;s=(typ[1]+typ[2])%2
            ps[s]+=mass;h[s]-=mass*w.ln()/ln2
        return sum(max(Decimal(0),(n-1)*s-entropy-s*s.ln()/ln2) for s,entropy in zip(ps,h) if s)/n
