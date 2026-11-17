"""Positive Bernstein probability polynomials and entropy bounds.

Every input probability is affine and nonnegative on the full uncertainty
interval. Positive Pauli enumerators therefore have nonnegative Bernstein
coefficients. Conditional entropy bounds use convexity/concavity, not a grid
assumption about the location of the worst input.
"""
from math import comb
from functools import lru_cache
import heapq
import numpy as np
from rates import xlog2x


@lru_cache(None)
def binomial(n):return np.array([comb(n,j) for j in range(n+1)],dtype=float)


def multiply(a,b):
    n=len(a)-1;m=len(b)-1
    return np.convolve(np.asarray(a)*binomial(n),np.asarray(b)*binomial(m))/binomial(n+m)


def power(a,n):
    result=np.ones(1)
    for _ in range(n):result=multiply(result,a)
    return result


def evaluate(coeff,t):
    coeff=np.asarray(coeff);n=coeff.shape[-1]-1
    weights=binomial(n)*t**np.arange(n+1)*(1-t)**np.arange(n,-1,-1)
    return coeff@weights


def split(coeff,t=.5):
    work=np.array(coeff,copy=True);n=work.shape[-1]-1
    left=np.empty_like(work);right=np.empty_like(work)
    left[...,0]=work[...,0];right[...,-1]=work[...,-1]
    for j in range(1,n+1):
        work=(1-t)*work[...,:-1]+t*work[...,1:]
        left[...,j]=work[...,0];right[...,n-j]=work[...,-1]
    return left,right


def monomial_coefficients(components,types):
    """components: 4 by (degree+1), return type by (total_degree+1)."""
    n=sum(types[0]);cache={(j,r):power(components[j],r) for j in range(4) for r in range(n+1)}
    result=[]
    for typ in types:
        v=np.ones(1)
        for j,r in enumerate(typ):v=multiply(v,cache[j,int(r)])
        result.append(v)
    return np.asarray(result)


def gain_coefficients(joint,hist,k):
    """Bounds for g(w)=k*sum(w)+sum(w log w)-sum(w)log(sum(w)).

    g is convex and positively homogeneous. Coefficientwise Jensen gives
    an upper polynomial. Its tangent at the midpoint gives a lower one.
    hist maps distinct joint probabilities to syndrome classes, counting
    repeated equal logical outcomes exactly.
    """
    ps=hist@joint
    upper=k*ps+hist@xlog2x(joint)-xlog2x(ps)
    mid=evaluate(joint,.5);total=hist@mid
    with np.errstate(divide='ignore',invalid='ignore'):
        logjoint=np.where(mid>0,np.log2(mid),0.)
        logtotal=np.where(total>0,np.log2(total),0.)
    lower=k*ps+(hist*logjoint[None,:])@joint-logtotal[:,None]*ps
    return lower,upper


class GainCurve:
    def __init__(self,raw_ED,kernel,mask=None,scale=1.):
        self.kernel=kernel
        self.joint=kernel.polys@monomial_coefficients(raw_ED,kernel.types)
        ps=raw_ED.sum(0)
        self.denominator=multiply(power(ps,kernel.n-1),np.ones(len(raw_ED[0])))
        self.mask=np.ones(len(kernel.hist)) if mask is None else np.asarray(mask,dtype=float)
        self.scale=scale/kernel.n

    def value(self,t):
        joint=evaluate(self.joint,t);ps=self.kernel.hist@joint
        gain=self.kernel.k*ps+self.kernel.hist@xlog2x(joint)-xlog2x(ps)
        denominator=evaluate(self.denominator,t)
        if denominator==0:return 0.
        return self.scale*np.dot(self.mask*self.kernel.multiplicity,gain)/denominator

    def interval(self,joint=None,denominator=None):
        if joint is None:joint=self.joint
        if denominator is None:denominator=self.denominator
        lower,upper=gain_coefficients(joint,self.kernel.hist,self.kernel.k)
        weights=self.scale*self.mask*self.kernel.multiplicity
        lo=weights@lower;hi=weights@upper
        active=denominator>0
        if not active.any():return (0.,0.)
        return float(np.min(lo[active]/denominator[active])),float(np.max(hi[active]/denominator[active]))


def certify_minimum(curves,atol=1e-10,rtol=1e-7,max_cells=20000):
    """Branch and bound minimum of a sum of fixed, signed entropy gains.

    Curves have nonnegative mixing weights. The reported numerical bounds
    include conservative roundoff padding; final witnesses need an independent
    high-precision/interval check before being called rigorous certificates.
    """
    samples=[(sum(c.value(t) for c in curves),t) for t in (0.,.5,1.)]
    incumbent,argmin=min(samples)
    counter=0;heap=[];evaluations=3
    def add(a,b,arrays):
        nonlocal counter,incumbent,argmin,evaluations
        bound=0.
        for c,(joint,den) in zip(curves,arrays):bound+=c.interval(joint,den)[0]
        midpoint=(a+b)/2
        value=sum(c.value(midpoint) for c in curves);evaluations+=1
        if value<incumbent:incumbent,argmin=value,midpoint
        padding=5e-12*max(1.,sum(abs(c.scale) for c in curves))
        heapq.heappush(heap,(bound-padding,counter,a,b,arrays));counter+=1
    add(0.,1.,[(c.joint,c.denominator) for c in curves])
    while heap and counter<max_cells:
        tolerance=atol+rtol*max(0.,abs(incumbent))
        if incumbent-heap[0][0]<=tolerance:break
        _,_,a,b,arrays=heapq.heappop(heap)
        pairs=[(split(j),split(d)) for j,d in arrays]
        middle=(a+b)/2
        add(a,middle,[(j[0],d[0]) for j,d in pairs])
        add(middle,b,[(j[1],d[1]) for j,d in pairs])
    lower=heap[0][0] if heap else incumbent
    return dict(lower=lower,upper=incumbent,argmin=argmin,cells=counter,
                converged=incumbent-lower<=atol+rtol*max(0.,abs(incumbent)),evaluations=evaluations)
