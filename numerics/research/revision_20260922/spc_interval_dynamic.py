"""O(N^3) outward Hellinger-walk bounds for every requested SPC length."""
from core import *
from fractions import Fraction as F
from math import factorial
from interval_modes import signed_mul
from long_code_bounds import sqrt_bound,h,clip,positive,scale,column,powers
import stable_intervals as iv


def exp2_point(x):
    """Enclose 2**x for exact binary64 x<=0 by a positive Taylor series."""
    x=np.asarray(x);safe=np.maximum(-1100.,np.minimum(0.,x));k=np.floor(safe)
    f=safe-k;fl=np.maximum(0.,iv.down(f));fh=np.minimum(1.,iv.up(f))
    al=np.maximum(0.,iv.down(fl*iv.LN2LO));ah=iv.up(fh*iv.LN2HI)
    def poly(a):
        v=np.full_like(a,1/factorial(20))
        for j in range(19,-1,-1):v=1/factorial(j)+a*v
        return v
    g=iv.gamma(256);lo=iv.down(poly(al)/(1+g));hi=iv.up(poly(ah)/(1-g))
    tail=iv.up(3*ah**21/factorial(21)/(1-iv.gamma(128))+128*iv.TINY);hi=iv.up(hi+tail)
    with np.errstate(under='ignore'):
        lo=np.maximum(0.,iv.down(np.ldexp(lo,k.astype(int))));hi=iv.up(np.ldexp(hi,k.astype(int)))
    return np.where(x<-1100,0.,lo),np.where(x<-1100,iv.TINY,hi)


def log1p_ratio_point(t):
    """ln(1+t)/t, including t=0, without cancellation in 1+t."""
    t=np.asarray(t);den=2+t;r=t/den;g=iv.gamma(512)
    def series(r):
        rr=r*r;v=np.full_like(r,1/31)
        for k in range(14,-1,-1):v=1/(2*k+1)+rr*v
        return 2/den*v
    v=series(r);rh=iv.up(r/(1-iv.gamma(3)))
    tail=iv.up(2/iv.down(den)*rh**32/(33*(1-rh*rh))/(1-iv.gamma(128))+128*iv.TINY)
    return np.maximum(0,iv.down(v/(1+g))),iv.up(v/(1-g)+tail)


def ambiguity_point(L):
    """The entropy ambiguity divided by the Hellinger weight; decreasing L."""
    safe=np.minimum(1024.,np.maximum(0.,L));t=exp2_point(-safe);root=exp2_point(-safe/2)
    # ln(1+t)/t decreases with t, so use the opposite interval endpoints.
    low,_=log1p_ratio_point(t[1]);_,high=log1p_ratio_point(t[0]);ratio=(low,high)
    one=np.ones_like(safe);first=iv.mul(iv.add((one,one),t),ratio)
    first=iv.div(first,(np.full_like(safe,iv.LN2LO),np.full_like(safe,iv.LN2HI)))
    result=iv.scale(iv.mul(root,iv.add(first,(safe,safe))),F(1,2))
    # For L>=1024, f(L)<=2**(-L/4). This avoids underflow ratios and
    # remains far below any rate relevant to the finite-menu certificate.
    tail=exp2_point(-np.minimum(L,1e6)/4)[1]
    lo=np.where(L>1024,0.,result[0]);hi=np.where(L>1024,tail,result[1])
    return np.where(np.isinf(L),0.,lo),np.where(np.isinf(L),0.,hi)


def log_ratio(a,b):
    al,ah=iv.log2_positive(np.where(a[0]>0,a[0],1.))[0],iv.log2_positive(np.where(a[1]>0,a[1],1.))[1]
    bl,bh=iv.log2_positive(np.where(b[0]>0,b[0],1.))[0],iv.log2_positive(np.where(b[1]>0,b[1],1.))[1]
    lo=iv.down(al-bh);hi=iv.up(ah-bl)
    lo=np.where(a[0]==0,-np.inf,lo);hi=np.where(b[0]==0,np.inf,hi)
    return lo,hi


def likelihood(u,v,a,b):
    def signed_integer(term,coefficient):
        with np.errstate(invalid='ignore'):
            lo=np.where(coefficient>=0,term[0][:,None,None]*coefficient,term[1][:,None,None]*coefficient)
            hi=np.where(coefficient>=0,term[1][:,None,None]*coefficient,term[0][:,None,None]*coefficient)
        return np.where(coefficient==0,0.,iv.down(lo)),np.where(coefficient==0,0.,iv.up(hi))
    with np.errstate(invalid='ignore'):
        ll,lh=iv.add(signed_integer(u,a),signed_integer(v,b))
    bad=np.isnan(ll)|np.isnan(lh);ll=np.where(bad,-np.inf,ll);lh=np.where(bad,np.inf,lh)
    return np.where((ll<=0)&(lh>=0),0.,np.minimum(np.abs(ll),np.abs(lh))),np.maximum(np.abs(ll),np.abs(lh))


def all_bounds(p,lengths,needed,batch=8):
    lengths=np.asarray(lengths,int);lo=np.zeros(needed.shape);hi=lo.copy();column_index={n:i for i,n in enumerate(lengths)}
    maxneeded=np.max(np.where(needed,lengths[None,:],0),-1);indices=np.flatnonzero(maxneeded>0);indices=indices[np.argsort(maxneeded[indices])]
    for start in range(0,len(indices),batch):
        ids=indices[start:start+batch];q=(p[0][ids],p[1][ids]);g0=sqrt_bound(iv.mul(column(q,0),column(q,3)));g1=sqrt_bound(iv.mul(column(q,1),column(q,2)))
        u=log_ratio(column(q,0),column(q,3));v=log_ratio(column(q,1),column(q,2));entropy=h(q);H=iv.positive_sum(entropy)
        Hp=iv.sub(iv.add(column(entropy,0),column(entropy,3)),iv.add(column(entropy,1),column(entropy,2)))
        d=iv.sub((np.ones(len(ids)),np.ones(len(ids))),scale(clip(iv.add(column(q,1),column(q,2))),2));dp=powers(d,int(maxneeded[ids].max()),True)
        distribution=(np.ones((len(ids),1,1)),np.ones((len(ids),1,1)))
        largest=int(maxneeded[ids].max());cached=None
        requested=[int(n) for n in lengths if needed[ids,column_index[n]].any()]
        if sum((n+1)**2 for n in requested)>3*(2*largest+1)**2:
            # The ambiguity depends only on the two integer displacements,
            # not the walk length. Reuse it across every requested n.
            aa=np.arange(-largest,largest+1)[:,None];bb=np.arange(-largest,largest+1)[None,:]
            grid=likelihood(u,v,aa,bb)
            cached=(ambiguity_point(grid[1])[0],ambiguity_point(grid[0])[1])
        for n in range(1,int(maxneeded[ids].max())+1):
            nxt=(np.zeros((len(ids),n+1,n+1)),np.zeros((len(ids),n+1,n+1)))
            aa=iv.mul((g0[0][:,None,None],g0[1][:,None,None]),distribution);bb=iv.mul((g1[0][:,None,None],g1[1][:,None,None]),distribution)
            for a,sl in ((aa,(slice(None),slice(None,-1),slice(None,-1))),(aa,(slice(None),slice(1,None),slice(1,None))),
                         (bb,(slice(None),slice(1,None),slice(None,-1))),(bb,(slice(None),slice(None,-1),slice(1,None)))):
                out=iv.add((nxt[0][sl],nxt[1][sl]),a)
                for k in (0,1):nxt[k][sl]=np.maximum(0,out[k])
            distribution=nxt
            if n not in column_index:continue
            col=column_index[n];take=np.flatnonzero(needed[ids,col])
            if not len(take):continue
            ii=np.arange(n+1)[:,None];jj=np.arange(n+1)[None,:];a=n-ii-jj;b=jj-ii
            if cached is None:
                L=likelihood((u[0][take],u[1][take]),(v[0][take],v[1][take]),a,b)
                al=ambiguity_point(L[1])[0];ah=ambiguity_point(L[0])[1]
            else:
                al=cached[0][take[:,None,None],a[None,:,:]+largest,b[None,:,:]+largest]
                ah=cached[1][take[:,None,None],a[None,:,:]+largest,b[None,:,:]+largest]
            weighted=iv.mul((distribution[0][take],distribution[1][take]),(al,ah));gains=[]
            for sign,parity in ((1,0),(-1,1)):
                loss=iv.positive_sum((weighted[0][:,(ii+jj)%2==parity],weighted[1][:,(ii+jj)%2==parity]))
                power=(dp[n][0][take],dp[n][1][take]);mass=clip(scale(iv.add((np.ones(len(take)),np.ones(len(take))),scale(power,sign)),F(1,2)))
                old=(dp[n-1][0][take],dp[n-1][1][take]);hh=(H[0][take],H[1][take]);hp=(Hp[0][take],Hp[1][take])
                physical=scale(iv.add(hh,scale(signed_mul(old,hp),sign)),F(n,2))
                gains.append(positive(iv.add(iv.add(iv.sub(scale(mass,n-1),physical),h(mass)),loss)))
            rate=scale(iv.add(*gains),F(1,n));lo[ids[take],col]=rate[0];hi[ids[take],col]=rate[1]
    return lo,hi
