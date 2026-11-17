"""Derivative of the Hellinger walk, avoiding a separate type sum per n."""
from spc_interval_dynamic import *
import interval_jets as j


def sqrt_jet(a):
    value=sqrt_bound(a.v);den=scale(value,2)
    safe=(np.maximum(den[0],iv.TINY),np.maximum(den[1],iv.TINY))
    reciprocal=(iv.down(1/safe[1]),iv.up(1/safe[0]));derivative=j.smul(a.d,reciprocal)
    return j.Jet(value,derivative)


def logarithm_ratio(a,b):
    value=log_ratio(a.v,b.v)
    aa=j.Jet(a.d)/j.Jet(a.v);bb=j.Jet(b.d)/j.Jet(b.v)
    derivative=(aa-bb)/j.Jet((iv.LN2LO,iv.LN2HI))
    return j.Jet(value,derivative.v)


def abs_jet(g):
    lo,hi=g.v;dl,dh=g.d;radius=np.maximum(np.abs(dl),np.abs(dh))
    vl=np.where((lo<=0)&(hi>=0),0.,np.minimum(np.abs(lo),np.abs(hi)));vh=np.maximum(np.abs(lo),np.abs(hi))
    low=np.where(lo>=0,dl,np.where(hi<=0,-dh,-radius));high=np.where(lo>=0,dh,np.where(hi<=0,-dl,radius))
    return j.Jet((vl,vh),(low,high))


def ambiguity_jet(L):
    value=(ambiguity_point(L.v[1])[0],ambiguity_point(L.v[0])[1])
    tl=exp2_point(-L.v[1])[0];th=exp2_point(-L.v[0])[1]
    root=(exp2_point(-L.v[1]/2)[0],exp2_point(-L.v[0]/2)[1])
    ml=log1p_ratio_point(th)[0];mh=log1p_ratio_point(tl)[1]
    positive_part=iv.mul(positive(iv.sub((np.ones_like(tl),np.ones_like(tl)),(tl,th))),(ml,mh))
    negative_part=iv.mul(L.v,(np.full_like(tl,iv.LN2LO),np.full_like(tl,iv.LN2HI)))
    slope=scale(j.smul(root,iv.sub(positive_part,negative_part)),F(1,4))
    sl=np.maximum(-.185,np.where(np.isnan(slope[0]),-np.inf,slope[0]));sh=np.minimum(0.,np.where(np.isnan(slope[1]),np.inf,slope[1]))
    large=L.v[0]>1024;start=np.minimum(L.v[0],1e6);r=exp2_point(-start/2)[1]
    bound=iv.up(iv.up(r*start)*iv.LN2HI/4)
    sl=np.where(large,-bound,sl);sh=np.where(large,0.,sh)
    return j.Jet(value,j.smul((sl,sh),L.d))


def positive_source(p,n):
    count=len(p.v[0]);g0=sqrt_jet(p[...,0]*p[...,3]);g1=sqrt_jet(p[...,1]*p[...,2])
    u=logarithm_ratio(p[...,0],p[...,3]);v=logarithm_ratio(p[...,1],p[...,2])
    distribution=(np.ones((count,1,1)),np.ones((count,1,1)))
    # D_n=(g0*A+g1*B)^n. Differentiate this polynomial only once:
    # D_n'=n*(g0'*A+g1'*B)*D_(n-1). Propagating derivative boxes
    # through all n layers is slower and introduces avoidable dependency.
    for nn in range(1,n+1):
        shape=(count,nn+1,nn+1);nxt=(np.zeros(shape),np.zeros(shape))
        previous=distribution
        aa=iv.mul((g0.v[0][:,None,None],g0.v[1][:,None,None]),previous)
        bb=iv.mul((g1.v[0][:,None,None],g1.v[1][:,None,None]),previous)
        for term,sl in ((aa,(slice(None),slice(None,-1),slice(None,-1))),(aa,(slice(None),slice(1,None),slice(1,None))),
                        (bb,(slice(None),slice(1,None),slice(None,-1))),(bb,(slice(None),slice(None,-1),slice(1,None)))):
            val=iv.add((nxt[0][sl],nxt[1][sl]),term)
            for i in (0,1):nxt[i][sl]=np.maximum(0,val[i])
        distribution=nxt
    derivative=(np.zeros_like(distribution[0]),np.zeros_like(distribution[1]))
    da=scale(j.smul((g0.d[0][:,None,None],g0.d[1][:,None,None]),previous),n)
    db=scale(j.smul((g1.d[0][:,None,None],g1.d[1][:,None,None]),previous),n)
    for term,sl in ((da,(slice(None),slice(None,-1),slice(None,-1))),(da,(slice(None),slice(1,None),slice(1,None))),
                    (db,(slice(None),slice(1,None),slice(None,-1))),(db,(slice(None),slice(None,-1),slice(1,None)))):
        val=iv.add((derivative[0][sl],derivative[1][sl]),term)
        for i in (0,1):derivative[i][sl]=val[i]
    distribution=j.Jet(distribution,derivative)
    ii=np.arange(n+1)[:,None];jj=np.arange(n+1)[None,:];a=n-ii-jj;b=jj-ii
    signed=u[...,None,None]*j.Jet((a,a))+v[...,None,None]*j.Jet((b,b))
    ambiguity=ambiguity_jet(abs_jet(signed));weighted=distribution*ambiguity
    entropy=j.h(p);hp=entropy.dot(np.array([1,0,0,1]))-entropy.dot(np.array([0,1,1,0]));H=entropy.sum()
    d=1-2*(p[...,1]+p[...,2]);result=j.Jet((np.zeros(count),np.zeros(count)))
    for sign,parity in ((1,0),(-1,1)):
        loss=weighted[:,(ii+jj)%2==parity].sum();mass=((1+sign*d**n)/2).clip()
        physical=n*(H+sign*d**(n-1)*hp)/2
        result=result+j.positive((n-1)*mass-physical+j.h(mass)+loss)/n
    return result


def spc_jet(p,n):
    # Log derivatives require positive probabilities. At zero-touching cells
    # use a value enclosure and an unrestricted derivative, then subdivision.
    safe=np.all(p.v[0]>1e-150,axis=-1);count=len(safe)
    value=(np.zeros(count),np.zeros(count));derivative=(np.full(count,-np.inf),np.full(count,np.inf))
    if safe.any():
        ids=np.flatnonzero(safe);result=positive_source(p[ids],n)
        for i in (0,1):value[i][ids]=result.v[i];derivative[i][ids]=result.d[i]
    if not safe.all():
        ids=np.flatnonzero(~safe);lo,hi=all_bounds((p.v[0][ids],p.v[1][ids]),np.array([n]),np.ones((len(ids),1),bool))
        value[0][ids]=lo[:,0];value[1][ids]=hi[:,0]
    return j.Jet(value,derivative)
