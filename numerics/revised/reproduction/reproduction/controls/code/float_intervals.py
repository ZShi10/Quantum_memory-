"""Vector probability/entropy enclosures using IEEE binary64 arithmetic.

Positive dot products use standard gamma_(2N) forward-error bounds; every
elementary interval operation is enlarged by one representable neighbor.
Logarithms use a finite atanh series and an analytic tail, not libm accuracy.
The floating environment must be round-to-nearest binary64, with no unsafe
fast-math transformations. Independent Decimal comparisons test this engine.
"""
import numpy as np
from fractions import Fraction

U=2.**-53
TINY=np.nextafter(0.,1.)
LN2LO=np.nextafter(float('0.693147180559945309417232121458176568'),-np.inf)
LN2HI=np.nextafter(float('0.693147180559945309417232121458176568'),np.inf)


def down(x):return np.nextafter(x,-np.inf)
def up(x):return np.nextafter(x,np.inf)
def gamma(n):return up((n*U)/(1-n*U))
def point(x):
    a=np.asarray(x,dtype=float);return down(a),up(a)
def rational(x):
    x=Fraction(x);f=float(x);ff=Fraction(f)
    return (np.array(f if ff<=x else down(f)),np.array(f if ff>=x else up(f)))
def add(a,b):return down(a[0]+b[0]),up(a[1]+b[1])
def sub(a,b):return down(a[0]-b[1]),up(a[1]-b[0])
def mul(a,b):
    # All uses in the probability engine have nonnegative operands.
    assert np.all(a[0]>=0) and np.all(b[0]>=0)
    return np.maximum(0,down(a[0]*b[0])),up(a[1]*b[1])
def div(a,b):
    assert np.all(a[0]>=0) and np.all(b[0]>0)
    return np.maximum(0,down(a[0]/b[1])),up(a[1]/b[0])
def scale(a,x):return mul(a,rational(x))
def positive_dot(a,b):
    b=np.asarray(b,dtype=float);assert np.all(b>=0)
    n=b.shape[0];g=gamma(2*n+2)
    lo=a[0]@b;hi=a[1]@b
    return np.maximum(0,down(lo/(1+g)-n*TINY)),up(hi/(1-g)+n*TINY)
def positive_sum(a,axis=-1):
    n=a[0].shape[axis];g=gamma(n+1)
    return np.maximum(0,down(a[0].sum(axis)/(1+g)-n*TINY)),up(a[1].sum(axis)/(1-g)+n*TINY)


def log2_positive(x):
    """Certified bounds for log2 of an EXACT positive binary64 x."""
    x=np.asarray(x,dtype=float);assert np.all(x>0)
    m,e=np.frexp(x);m=m*2;e=e-1
    # m-1 is exact by Sterbenz. The positive quotient has relative error
    # <=gamma2. Every power in the sum has exponent <=31, so gamma256
    # covers its propagation as well as the Horner arithmetic below.
    num=m-1;den=m+1;r=num/den
    rh=up(r/(1-gamma(2)))
    def series(r):
        rr=r*r;value=np.full_like(r,1/31)
        for j in range(14,-1,-1):value=1/(2*j+1)+rr*value
        return 2*r*value
    gl=gamma(256);value=series(r)
    sl=np.maximum(0,down(value/(1+gl)-128*TINY))
    tail=up((2*rh**33/(33*(1-rh*rh)))/(1-gamma(128)))
    sh=up(value/(1-gl)+tail+128*TINY)
    return down(e+down(sl/LN2HI)),up(e+up(sh/LN2LO))


def entropy_exact_float(x):
    x=np.asarray(x,dtype=float);positive=x>0
    safe=np.where(positive,x,1.);l,h=log2_positive(safe)
    return np.where(positive,down(-x*h),0.),np.where(positive,up(-x*l),0.)


def entropy_interval(x):
    lo,hi=x;assert np.all(lo>=0)
    # A single log evaluation at hi, followed by a mean-value bound.
    # ln(hi/lo)<= (hi-lo)/lo; this avoids evaluating the same logarithm
    # independently at two almost-identical interval endpoints.
    positive=hi>0;safe=np.where(positive,hi,1.)
    ll,lh=log2_positive(safe);bl=np.where(positive,down(-hi*lh),0.);bh=np.where(positive,up(-hi*ll),0.)
    width=up(hi-lo)
    with np.errstate(over='ignore',invalid='ignore'):
        lower_log=down(ll-up(up(width/np.maximum(lo,TINY))/LN2LO))
        slope=up(np.maximum(np.abs(lower_log),np.abs(lh))+up(1/LN2LO))
    error=up(width*slope);lower=down(bl-error);upper=up(bh+error)
    zero=lo==0
    # On [0,hi] entropy is nonnegative for hi<=1 and monotone until 1/e.
    lower=np.where(zero,np.minimum(0,bl),lower)
    upper=np.where(zero,bh,upper)
    # The unique maximum of -x log2 x is 1/(e ln2), enclosed constants.
    cross=(lo<=float('0.3678794411714424'))&(hi>=float('0.3678794411714422'))
    upper=np.where(cross,np.maximum(upper,float('0.5307378454230432')),upper)
    return lower,upper


def monomial_bounds(p,types):
    n=int(np.max(np.sum(types,axis=1)));cache=[]
    for j in range(4):
        powers=[(np.ones(len(p[0])),np.ones(len(p[0])))]
        for i in range(n):powers.append(mul(powers[-1],(p[0][:,j],p[1][:,j])))
        cache.append(powers)
    out=(np.ones((len(p[0]),len(types))),np.ones((len(p[0]),len(types))))
    for j in range(4):
        a=(np.stack([cache[j][int(t)][0] for t in types[:,j]],-1),np.stack([cache[j][int(t)][1] for t in types[:,j]],-1))
        out=mul(out,a)
    return out


def code_bounds(powers,kernel):
    joint=positive_dot(powers,kernel.polys.T)
    ps=positive_dot(powers,kernel.marginal.T)
    hj=entropy_interval(joint);hs=entropy_interval(ps)
    # hmass=sum h(joint)-h(flag); all h terms are nonnegative since the
    # probability upper enclosures can exceed one by a few ulps only.
    hsum=positive_dot((np.maximum(0,hj[0]),np.maximum(0,hj[1])),kernel.hist.T)
    gain=add(sub(scale(ps,kernel.k),hsum),hs)
    # Exact maximum is the per-flag positive part; each retained set that
    # is positive at the lower endpoint realizes the returned lower rate.
    clipped=(np.maximum(0,gain[0]),np.maximum(0,gain[1]))
    return scale(positive_dot(clipped,kernel.multiplicity),Fraction(1,kernel.n))


def best_code_bounds(p,kernels,batch=512):
    from algebra import compositions
    lower=np.zeros(len(p[0]));upper=lower.copy();choices=np.full(len(lower),-1,dtype=int)
    for start in range(0,len(lower),batch):
        end=min(start+batch,len(lower));local=(p[0][start:end],p[1][start:end]);powers={}
        for n in sorted({k.n for k in kernels}):powers[n]=monomial_bounds(local,np.asarray(compositions(n)))
        for j,k in enumerate(kernels):
            lo,hi=code_bounds(powers[k.n],k);take=lo>lower[start:end]
            lower[start:end]=np.maximum(lower[start:end],lo);upper[start:end]=np.maximum(upper[start:end],hi)
            choices[start:end][take]=j
    return lower,upper,choices
