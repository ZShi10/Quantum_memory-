"""Stable long repetition AD and continuous-input enclosures.

Only low-degree ED probabilities are represented as Bernstein polynomials.
Long AD uses bias formulas, so an effective block of size 2**100 does not
require expanding a polynomial of that degree. These are numerical bounds;
high-precision witnesses are checked separately.
"""
from dataclasses import dataclass
from math import log
import heapq
import numpy as np
from bernstein import power,evaluate,split,multiply

LN2=np.log(2.)

def binary_entropy(x):
    x=np.clip(np.asarray(x,dtype=float),0.,1.)
    with np.errstate(divide='ignore',invalid='ignore'):
        return -np.where(x>0,x*np.log(x),0.)/LN2-np.where(x<1,(1-x)*np.log1p(-x),0.)/LN2

def deficit(c):
    """1-h((1-c)/2), without cancellation at small bias."""
    c=np.clip(np.abs(np.asarray(c,dtype=float)),0.,1.)
    small=c<.25;v=c*c;series=np.zeros_like(c);term=v.copy()
    for j in range(1,15):
        series+=term/((2*j)*(2*j-1)*LN2);term*=v
    with np.errstate(divide='ignore',invalid='ignore'):
        direct=((1+c)*np.log1p(c)+np.where(c<1,(1-c)*np.log1p(-c),0.))/(2*LN2)
    return np.where(small,series,direct)

def repeated_parameters(epsilon,phi0,phi1,N):
    epsilon=np.clip(np.asarray(epsilon),0.,1.)
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        betaN=np.exp(-2*N*np.arctanh(epsilon))
    e=betaN/(1+betaN)
    ds=[]
    for phi in (phi0,phi1):
        phi=np.minimum(np.clip(phi,0.,1.),1-np.clip(phi,0.,1.))
        with np.errstate(divide='ignore',invalid='ignore'):
            bias=np.exp(N*np.log1p(-2*phi))
        ds.append(deficit(bias))
    return e,ds[0],ds[1]

def pooled_yield(epsilon,k,n):
    epsilon=np.clip(np.asarray(epsilon),0.,1.)
    with np.errstate(divide='ignore',invalid='ignore'):
        alpha=np.arctanh(epsilon)
    result=np.ones_like(epsilon)
    for j in range(k):result*=(1+np.tanh((2.**j)*alpha)**2)/4
    e=np.tanh((2.**k)*alpha)
    result*=(((1+e)/2)**n+((1-e)/2)**n)/n
    return result

def rate(epsilon,phi0,phi1,k,n):
    N=(2.**k)*n
    e,d0,d1=repeated_parameters(epsilon,phi0,phi1,N)
    gain=(1-e)*d0+e*d1-binary_entropy(e)
    return pooled_yield(epsilon,k,n)*np.maximum(gain,0.)

def ratio_range(numerator,denominator):
    active=denominator>0
    if not active.any():return (0.,0.)
    ratios=numerator[active]/denominator[active]
    return float(ratios.min()),float(ratios.max())

@dataclass
class EDParameters:
    name:str
    # (epsilon, phi0, phi1, ED yield), each a same-degree ratio
    ratios:list
    memory:int
    metadata:dict

    def values(self,t):
        values=[]
        for numerator,denominator in self.ratios:
            den=evaluate(denominator,t)
            values.append(evaluate(numerator,t)/den if den>0 else 0.)
        return values

    def value(self,t,k,n):
        eps,p0,p1,y=self.values(t)
        return float(y*rate(eps,p0,p1,k,n))

    def lower(self,arrays,k,n):
        (elo,ehi),(p0lo,p0hi),(p1lo,p1hi),(ylo,yhi)=[ratio_range(*a) for a in arrays]
        # Enclose min(phi,1-phi) even when phi crosses one half.
        phis=[]
        for lo,hi in ((p0lo,p0hi),(p1lo,p1hi)):
            phis.append(.5 if lo<=.5<=hi else max(min(lo,1-lo),min(hi,1-hi)))
        N=(2.**k)*n
        emax,d0,d1=repeated_parameters(max(0.,elo),*phis,N)
        emin,_,_=repeated_parameters(min(1.,ehi),*phis,N)
        # The entropy loss is concave in e; its maximum can be interior.
        stationary=1/(1+2**(float(d1)-float(d0)))
        e=np.clip(stationary,emin,emax)
        gain=(1-e)*d0+e*d1-binary_entropy(e)
        # Relative margin avoids declaring roundoff-sized gains positive.
        margin=3e-13*((1-float(e))*abs(float(d0))+float(e)*abs(float(d1))+abs(float(binary_entropy(e))))
        return float(max(0.,ylo)*pooled_yield(max(0.,elo),k,n)*max(0.,float(gain)-margin))

    def certify(self,k,n,rtol=1e-5,atol=0.,max_cells=6000):
        samples=[(self.value(t,k,n),t) for t in (0.,.25,.5,.75,1.)]
        incumbent,argmin=min(samples);heap=[];counter=0
        def add(a,b,arrays):
            nonlocal counter,incumbent,argmin
            value=self.value((a+b)/2,k,n)
            if value<incumbent:incumbent,argmin=value,(a+b)/2
            lower=self.lower(arrays,k,n)
            heapq.heappush(heap,(lower,counter,a,b,arrays));counter+=1
        add(0.,1.,self.ratios)
        while counter<max_cells:
            lower=heap[0][0]
            if incumbent-lower<=atol+rtol*incumbent:break
            _,_,a,b,arrays=heapq.heappop(heap)
            sub=[(split(x),split(y)) for x,y in arrays]
            add(a,(a+b)/2,[(x[0],y[0]) for x,y in sub])
            add((a+b)/2,b,[(x[1],y[1]) for x,y in sub])
        lower=heap[0][0]
        return dict(lower=lower,upper=incumbent,argmin=argmin,cells=counter,
                    relative_gap=(incumbent-lower)/incumbent if incumbent else 0.,converged=incumbent-lower<=atol+rtol*incumbent)

def pumping(eb,ep,r,axis=0):
    """r raw inputs on the successful path; fixed complementary key axis."""
    if axis:eb,ep=ep,eb
    zmax=min(eb,ep)
    A=(1-eb)**r;B=eb**r
    V=power(np.array([1-eb-2*ep,1-eb-2*ep+2*zmax]),r)
    W=power(np.array([eb,eb-2*zmax]),r)
    one=np.ones(r+1)
    C=(1+sum((1-eb)**j+eb**j for j in range(1,r)))/(A+B)
    # Expressions preserve an exponentially small positive bit bias.
    ratios=[(V+W,(A+B)*one),
            (B*one+W,(A+B)*one+V+W),
            (B*one-W,(A+B)*one-V-W),
            (one/C,one)]
    return EDParameters(f'pump_{axis}_{r}',ratios,2,dict(family='pumping',r=r,axis=axis,expected_raw_cost=C))

def from_raw(name,raw,yield_numerator=None,yield_denominator=None,memory=1,metadata=None):
    """Nonnegative ED polynomials; sufficiently biased I-majority outputs."""
    a,x,y,z=raw;total=raw.sum(0)
    if yield_numerator is None:yield_numerator=total
    if yield_denominator is None:yield_denominator=np.ones(len(total))
    ratios=[(a+z-x-y,total),(z,a+z),(y,x+y),(yield_numerator,yield_denominator)]
    return EDParameters(name,ratios,memory,{} if metadata is None else metadata)

def direct(eb,ep,axis):
    t=min(eb,ep);s=eb+ep
    p=np.array([[1-s,eb,0.,ep],[1-s+t,eb-t,t,ep-t]]).T
    permutation=((0,1,2,3),(0,3,2,1),(0,1,3,2))[axis]
    return from_raw(f'direct_{axis}',p[list(permutation)],metadata=dict(family='direct',axis=axis))
