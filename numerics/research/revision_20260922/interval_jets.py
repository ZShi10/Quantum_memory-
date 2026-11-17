"""Outward interval values and first derivatives for BB84 rate certificates.

The derivative encloses the whole cell, including all active branches of a
positive part. A midpoint mean-value bound then retains cancellations near
stationary points. Endpoint logarithmic singularities fall back to boxes.
"""
from core import *
from fractions import Fraction as F
from recurrence_v2 import MAPS
from recursive_library import PRE
from interval_modes import signed_sum,signed_dot
import stable_intervals as iv


def smul(a,b):
    lows=[];highs=[]
    for x in a:
        for y in b:
            with np.errstate(invalid='ignore',over='ignore'):v=x*y
            v=np.where((x==0)|(y==0),0.,v)
            lows.append(iv.down(v));highs.append(iv.up(v))
    lo,hi=np.minimum.reduce(lows),np.maximum.reduce(highs)
    zero=((a[0]==0)&(a[1]==0))|((b[0]==0)&(b[1]==0))
    return np.where(zero,0.,lo),np.where(zero,0.,hi)


class Jet:
    def __init__(self,val,der=None):
        self.v=tuple(np.asarray(x,dtype=float) for x in val)
        self.d=(np.zeros_like(self.v[0]),np.zeros_like(self.v[1])) if der is None else tuple(np.asarray(x,dtype=float) for x in der)
        # Extended interval arithmetic can encounter infinity-infinity on a
        # coarse box. It means no finite bound, never permission to emit NaN.
        self.v=(np.where(np.isnan(self.v[0]),-np.inf,self.v[0]),np.where(np.isnan(self.v[1]),np.inf,self.v[1]))
        self.d=(np.where(np.isnan(self.d[0]),-np.inf,self.d[0]),np.where(np.isnan(self.d[1]),np.inf,self.d[1]))
    @classmethod
    def constant(cls,x):return cls(iv.rational(F(x)))
    def __add__(self,other):
        b=other if isinstance(other,Jet) else Jet.constant(other)
        return Jet(iv.add(self.v,b.v),iv.add(self.d,b.d))
    __radd__=__add__
    def __neg__(self):return Jet((-self.v[1],-self.v[0]),(-self.d[1],-self.d[0]))
    def __sub__(self,other):return self+-(other if isinstance(other,Jet) else Jet.constant(other))
    def __rsub__(self,other):return Jet.constant(other)+-self
    def __mul__(self,other):
        b=other if isinstance(other,Jet) else Jet.constant(other)
        return Jet(smul(self.v,b.v),iv.add(smul(self.d,b.v),smul(self.v,b.d)))
    __rmul__=__mul__
    def __truediv__(self,other):
        b=other if isinstance(other,Jet) else Jet.constant(other)
        good=b.v[0]>0;lo=np.where(good,b.v[0],1.);hi=np.where(good,b.v[1],1.)
        reciprocal=(iv.down(1/hi),iv.up(1/lo))
        value=smul(self.v,reciprocal)
        deriv=smul(iv.sub(smul(self.d,b.v),smul(self.v,b.d)),smul(reciprocal,reciprocal))
        return Jet((np.where(good,value[0],-np.inf),np.where(good,value[1],np.inf)),
                   (np.where(good,deriv[0],-np.inf),np.where(good,deriv[1],np.inf)))
    def __pow__(self,n):
        assert isinstance(n,int) and n>=0
        result=Jet.constant(1);base=self
        while n:
            if n&1:result=result*base
            n//=2
            if n:base=base*base
        return result
    def clip(self,lo=0.,hi=1.):
        # Intersect a known range of the exact function; do not differentiate
        # the numerical bound intersection as if it were a physical clipping.
        return Jet((np.maximum(lo,self.v[0]),np.minimum(hi,self.v[1])),self.d)
    def __getitem__(self,key):return Jet((self.v[0][key],self.v[1][key]),(self.d[0][key],self.d[1][key]))
    def sum(self,axis=-1):return Jet(signed_sum(self.v,axis),signed_sum(self.d,axis))
    def dot(self,weights):return Jet(signed_dot(self.v,weights),signed_dot(self.d,weights))


def stack(items,axis=-1):
    return Jet(tuple(np.stack([x.v[i] for x in items],axis) for i in (0,1)),
               tuple(np.stack([x.d[i] for x in items],axis) for i in (0,1)))


def maximum(items):
    lo=np.maximum.reduce([x.v[0] for x in items]);hi=np.maximum.reduce([x.v[1] for x in items])
    dl=np.minimum.reduce([np.where(x.v[1]>=lo,x.d[0],np.inf) for x in items])
    dh=np.maximum.reduce([np.where(x.v[1]>=lo,x.d[1],-np.inf) for x in items])
    return Jet((lo,hi),(dl,dh))


def positive(a):return maximum([a,Jet((np.zeros_like(a.v[0]),np.zeros_like(a.v[1])))])


def h(a):
    a=a.clip();value=iv.entropy_interval(a.v)
    value=(np.maximum(0,value[0]),np.maximum(0,value[1]))
    lo,hi=a.v;sl=np.where(lo>0,lo,1.);sh=np.where(hi>0,hi,1.)
    ll,_=iv.log2_positive(sl);_,lh=iv.log2_positive(sh)
    slope=(iv.down(-lh-iv.up(1/iv.LN2LO)),np.where(lo>0,iv.up(-ll-iv.down(1/iv.LN2HI)),np.inf))
    derivative=smul(slope,a.d)
    return Jet(value,derivative)


def source(b,p,intervals):
    b,p=F(str(b)),F(str(p));scale=min(b,p);lows=[];highs=[]
    for left,right in intervals:
        lo,hi=scale*F(left),scale*F(right)
        lows.append([1-b-p+lo,b-hi,lo,p-hi]);highs.append([1-b-p+hi,b-lo,hi,p-lo])
    bounds=(np.array([[iv.rational(x)[0] for x in row] for row in lows]),
            np.array([[iv.rational(x)[1] for x in row] for row in highs]))
    slope=[iv.rational(scale*s) for s in (1,-1,1,-1)]
    derivative=(np.broadcast_to([x[0] for x in slope],bounds[0].shape),np.broadcast_to([x[1] for x in slope],bounds[1].shape))
    return Jet(bounds,derivative)


def step(p,q,primitive):
    zero=Jet((np.zeros(p.v[0].shape[:-1]),np.zeros(p.v[0].shape[:-1])))
    raw=[zero,zero,zero,zero]
    for i,j,k in MAPS[primitive]:raw[k]=raw[k]+p[...,i]*q[...,j]
    raw=stack(raw).clip();mass=raw.sum().clip()
    return (raw/mass[...,None]).clip(),mass


def raw_step(p,q,primitive):
    zero=Jet((np.zeros(p.v[0].shape[:-1]),np.zeros(p.v[0].shape[:-1])))
    raw=[zero,zero,zero,zero]
    for i,j,k in MAPS[primitive]:raw[k]=raw[k]+p[...,i]*q[...,j]
    return stack(raw).clip()


def prepare(b,p,intervals,record,rounds=0):
    initial=source(b,p,intervals);state=initial;y=Jet((np.ones(len(intervals)),np.ones(len(intervals))))
    family=record['family']
    if family=='real_decoder_recurrence':
        raw=state;den=Jet((np.ones(len(intervals)),np.ones(len(intervals))))
        for level,primitive in enumerate(record['path']):
            raw=raw_step(raw,raw,primitive);mass=raw.sum().clip()
            # y_d=P_d/(2^d product_{j<d} P_j), where P_j is the
            # all-success fixed-tree probability. Normalize only once.
            y=(mass/(2*den)).clip()
            if level+1<len(record['path']):den=2*den*mass
        if record['path']:state=(raw/mass[...,None]).clip()
    elif family=='real_decoder_pumping':
        raw=initial;mass=Jet((np.ones(len(intervals)),np.ones(len(intervals))))
        den=Jet((np.ones(len(intervals)),np.ones(len(intervals))))
        for _ in range(record['r']-1):
            den=den+mass;raw=raw_step(raw,initial,record['primitive']);mass=raw.sum().clip()
        state=(raw/mass[...,None]).clip();y=(mass/den).clip()
    elif family=='homogeneous_recurrence':
        for axis in record['path']:
            state=state[...,list(PRE[axis])];state,s=step(state,state,'Z');y=(y*s/2).clip()
    else:
        initial=initial[...,list(PRE[record['preparation_axis']])];state=initial
        for _ in range(record['r']-1):state,s=step(state,initial,'Z');y=(y*s/(1+y)).clip()
    state=state[...,list(PRE[record['final_axis']])]
    for _ in range(rounds):state,s=step(state,state,'Z');y=(y*s/2).clip()
    return state,y


def watanabe(p):
    a=(p[...,0]+p[...,3]).clip();b=(p[...,1]+p[...,2]).clip()
    v=(a*b).clip(0,.25);u=(p[...,0]*p[...,1]+p[...,3]*p[...,2]).clip(0,.25)
    # Use the positive polynomial for v-u. Subtracting nearly equal terms
    # destroys relative precision of tiny phase probabilities after pumping
    # and produces an artificial logarithmic derivative singularity.
    rest=(p[...,0]*p[...,2]+p[...,3]*p[...,1]).clip(0,.25)
    extra=h(u)+h(rest)-h(v)
    first=1-h(p).sum()+extra
    out,s=step(p,p,'Z');second=s/2*(1-h(out).sum())
    return positive(maximum([first,second]))


def policy(b,p,intervals,record):
    state,y=prepare(b,p,intervals,record['ED'],record['AD']['classical_B_rounds'])
    return y*watanabe(state)


def mean_value_lower(box,middle,intervals):
    widths=np.array([iv.rational((F(c)-F(a))/2)[1] for a,c in intervals])
    slope=np.maximum(np.abs(box.d[0]),np.abs(box.d[1]));error=iv.up(widths*slope)
    return np.maximum(box.v[0],iv.down(middle.v[0]-error))
