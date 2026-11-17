"""Vector outward bounds for long repetition/SPC terminal rates."""
from core import *
from fractions import Fraction as F
from math import comb
from functools import lru_cache
from interval_modes import signed_mul,signed_sum
from compound_terminals import spc_type_table
import stable_intervals as iv


def clip(a):return np.maximum(0,a[0]),np.minimum(1,np.maximum(0,a[1]))
def h(a):
    out=iv.entropy_interval(clip(a));return np.maximum(0,out[0]),np.maximum(0,out[1])
def positive(a):return np.maximum(0,a[0]),np.maximum(0,a[1])
def stack(items,axis=-1):return tuple(np.stack([v[i] for v in items],axis) for i in (0,1))
def column(a,i):return a[0][...,i],a[1][...,i]
def scale(a,s):return signed_mul(a,iv.rational(F(s)))
def powers(a,n,signed=False):
    out=[(np.ones_like(a[0]),np.ones_like(a[1]))]
    for _ in range(n):out.append(signed_mul(out[-1],a) if signed else iv.mul(out[-1],a))
    return out


def sqrt_bound(a):
    a=positive(a)
    def normalized(x):
        m,e=np.frexp(x);odd=(e%2)!=0
        return np.where(odd,m*2,m),np.where(odd,e-1,e)//2
    ml,el=normalized(a[0]);mh,eh=normalized(a[1])
    lo=np.maximum(0,iv.down(np.sqrt(ml)));hi=iv.up(np.sqrt(mh))
    # Verify the bracket by outward squaring. This does not assume a
    # particular vendor's sqrt is correctly rounded.
    for _ in range(10):
        bad=(lo>0)&(iv.up(lo*lo)>ml)
        if not bad.any():break
        lo=np.where(bad,np.maximum(0,iv.down(lo)),lo)
    assert np.all((lo==0)|(iv.up(lo*lo)<=ml))
    for _ in range(10):
        bad=(mh>0)&(iv.down(hi*hi)<mh)
        if not bad.any():break
        hi=np.where(bad,iv.up(hi),hi)
    assert np.all((mh==0)|(iv.down(hi*hi)>=mh))
    return np.ldexp(lo,el),np.where(mh==0,0.,np.ldexp(hi,eh))


class LongBounds:
    def __init__(self,p,nmax=128):
        self.p=p;self.nmax=nmax;self.a=clip(iv.add(column(p,0),column(p,3)));self.b=clip(iv.add(column(p,1),column(p,2)))
        self.ap=powers(self.a,nmax);self.bp=powers(self.b,nmax)
        self.up=powers(iv.sub(column(p,0),column(p,3)),nmax,True)
        self.vp=powers(iv.sub(column(p,1),column(p,2)),nmax,True)
        self.dp=powers(iv.sub((np.ones(len(p[0])),np.ones(len(p[0]))),scale(self.b,2)),nmax,True)
        hp=h(p);self.H=iv.positive_sum(hp)
        self.Hp=iv.sub(iv.add(column(hp,0),column(hp,3)),iv.add(column(hp,1),column(hp,2)))
        self.phase=[]
        for pair,mass in (((0,3),self.a),((1,2),self.b)):
            numerator=positive(iv.sub(iv.add(column(hp,pair[0]),column(hp,pair[1])),h(mass)))
            good=mass[0]>0;safe=(np.where(good,mass[0],1.),np.where(good,mass[1],1.))
            hh=clip(iv.div(numerator,safe));self.phase.append((np.where(good,hh[0],0.),np.where(good,hh[1],1.)))
        o0=scale(sqrt_bound(iv.mul(column(p,0),column(p,3))),2)
        o1=scale(sqrt_bound(iv.mul(column(p,1),column(p,2))),2)
        self.op=powers(positive(iv.add(o0,o1)),nmax)
        self.om=powers(iv.sub(o0,o1),nmax,True)
        self.physical_powers=None

    def repetition(self,n,clip_net=True):
        ws=range(n//2+1)
        aa=stack([iv.mul(self.ap[n-w],self.bp[w]) for w in ws]);bb=stack([iv.mul(self.ap[w],self.bp[n-w]) for w in ws])
        cc=stack([signed_mul(self.up[n-w],self.vp[w]) for w in ws]);dd=stack([signed_mul(self.up[w],self.vp[n-w]) for w in ws])
        mass=clip(iv.add(aa,bb))
        raw=stack([clip(scale(iv.add(aa,cc),F(1,2))),clip(scale(iv.add(bb,dd),F(1,2))),
                   clip(scale(iv.sub(bb,dd),F(1,2))),clip(scale(iv.sub(aa,cc),F(1,2)))])
        gain=iv.add(iv.sub(mass,iv.positive_sum(h(raw))),h(mass))
        mult=[iv.rational(F(comb(n,w),n*(2 if 2*w==n else 1))) for w in ws]
        weights=tuple(np.array([v[i] for v in mult]) for i in (0,1))
        no=iv.positive_sum(iv.mul(positive(gain),weights))
        h0,h1=self.phase
        ph0=stack([positive(iv.add(scale(h0,n-w),scale(h1,w))) for w in ws])
        ph1=stack([positive(iv.add(scale(h0,w),scale(h1,n-w))) for w in ws])
        physical=iv.sub(iv.add(iv.add(h(aa),h(bb)),iv.add(iv.mul(aa,ph0),iv.mul(bb,ph1))),h(mass))
        gross=iv.positive_sum(iv.mul(positive(iv.sub(scale(mass,n),physical)),weights))
        cost=iv.positive_sum(iv.mul(h(mass),weights))
        finish=positive if clip_net else lambda v:v
        return no,finish(iv.sub(gross,iv.rational(F(n-1,n)))),finish(iv.sub(gross,cost))

    def spc(self,n,clip_net=True):
        ps=[];gross=[];nos=[]
        for sign in (1,-1):
            mass=clip(scale(iv.add((np.ones(len(self.p[0])),np.ones(len(self.p[0]))),scale(self.dp[n],sign)),F(1,2)))
            raw_h=scale(iv.add(self.H,scale(signed_mul(self.dp[n-1],self.Hp),sign)),F(n,2))
            gross.append(positive(iv.add(iv.sub(scale(mass,n),raw_h),h(mass))))
            overlap=positive(scale(iv.add(self.op[n],scale(self.om[n],sign)),F(1,2)))
            loss=(np.minimum(mass[0],overlap[0]),np.minimum(mass[1],overlap[1]))
            upper=positive(iv.add(iv.add(iv.sub(scale(mass,n-1),raw_h),h(mass)),loss))
            nos.append(upper);ps.append(mass)
        gross=scale(iv.add(*gross),F(1,n));cost=scale(iv.add(*[h(x) for x in ps]),F(1,n))
        finish=positive if clip_net else lambda v:v
        return scale(iv.add(*nos),F(1,n)),finish(iv.sub(gross,iv.rational(F(1,n)))),finish(iv.sub(gross,cost))

    def spc_exact(self,n,indices):
        """Complete type enumeration, only for states that survive the bound."""
        p=(self.p[0][indices],self.p[1][indices]);types,parities,coeff=spc_type_table(n)
        pp=[powers(column(p,i),n) for i in range(4)];ent=[(np.zeros(len(indices)),np.zeros(len(indices))) for _ in (0,1)]
        for start in range(0,len(types),512):
            stop=min(start+512,len(types));typ=types[start:stop]
            a=(np.ones((len(indices),len(typ))),np.ones((len(indices),len(typ))));b=a
            for i in range(4):
                a=iv.mul(a,stack([pp[i][t] for t in typ[:,i]]));b=iv.mul(b,stack([pp[3-i][t] for t in typ[:,i]]))
            weighted=iv.mul(h(clip(iv.add(a,b))),(coeff[0][start:stop],coeff[1][start:stop]))
            for s in (0,1):
                mask=parities[start:stop]==s
                ent[s]=iv.add(ent[s],iv.positive_sum((weighted[0][:,mask],weighted[1][:,mask])))
        dp=(self.dp[n][0][indices],self.dp[n][1][indices]);gains=[]
        for s,sign in enumerate((1,-1)):
            mass=clip(scale(iv.add((np.ones(len(indices)),np.ones(len(indices))),scale(dp,sign)),F(1,2)))
            gains.append(positive(iv.add(iv.sub(scale(mass,n-1),ent[s]),h(mass))))
        return positive(scale(iv.add(*gains),F(1,n)))


def classical_frame_boxes(p,quantum=1e-14):
    """Merge close known-state boxes by taking their containing hull.

    No equality of close states is assumed. Each exact state, after a
    classical entropy-preserving relabeling, remains inside its group box.
    """
    mid=(p[0]+p[1])/2;size=len(mid);rows=np.arange(size)
    a=np.tile([0,3],(size,1));b=np.tile([1,2],(size,1))
    aa=mid[:,0]<mid[:,3];bb=mid[:,1]<mid[:,2];a[aa]=a[aa,::-1];b[bb]=b[bb,::-1]
    av=mid[rows,a[:,0]];bv=mid[rows,b[:,0]]
    swap=(bv>av)|((bv==av)&(mid[rows,b[:,1]]>mid[rows,a[:,1]]))
    high=np.where(swap[:,None],b,a);low=np.where(swap[:,None],a,b)
    perm=np.column_stack((high[:,0],low[:,0],low[:,1],high[:,1]))
    pp=tuple(np.take_along_axis(v,perm,axis=1) for v in p)
    tag=np.rint(((pp[0]+pp[1])/2)/quantum).astype(np.int64)
    _,inverse=np.unique(tag,axis=0,return_inverse=True);count=int(inverse.max())+1
    lower=np.full((count,4),np.inf);upper=np.full((count,4),-np.inf)
    np.minimum.at(lower,inverse,pp[0]);np.maximum.at(upper,inverse,pp[1])
    return (lower,upper),inverse


def best_bounds(p,kernels,physical,nmax=128,target=None,compress=True):
    if compress and len(p[0])>1:
        boxes,inverse=classical_frame_boxes(p)
        if len(boxes[0])<len(p[0]):
            goal=None
            if target is not None:
                goal=np.full(len(boxes[0]),np.inf);np.minimum.at(goal,inverse,target)
            lo,hi=best_bounds(boxes,kernels,physical,nmax,goal,False)
            return lo[inverse],hi[inverse]
    from interval_modes import best_short_bounds
    from certify_bb84_recursive import watanabe_bound
    lo,hi=best_short_bounds(p,kernels,physical);wl,wh=watanabe_bound(p);lo=np.maximum(lo,wl);hi=np.maximum(hi,wh)
    if nmax<8:return lo,hi
    long=LongBounds(p,nmax);spc_candidates=[]
    for n in range(8,nmax+1):
        for rl,rh in long.repetition(n):lo=np.maximum(lo,rl);hi=np.maximum(hi,rh)
        spc,plain,compressed=long.spc(n);spc_candidates.append(spc)
        for rl,rh in (plain,compressed):lo=np.maximum(lo,rl);hi=np.maximum(hi,rh)
    threshold=lo if target is None else np.maximum(lo,target)
    from spc_interval_dynamic import all_bounds
    needed=np.column_stack([spc[1]>threshold for spc in spc_candidates])
    lower,upper=all_bounds(p,np.arange(8,nmax+1),needed)
    for i,spc in enumerate(spc_candidates):
        lo=np.maximum(lo,lower[:,i]);hi=np.maximum(hi,np.where(needed[:,i],upper[:,i],spc[1]))
    return lo,hi
