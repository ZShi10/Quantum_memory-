"""Scalar compiled forms of existing outward repetition/ambiguity formulas.

No code, syndrome, or length is removed. Every elementary interval operation
retains directed nextafter enlargement, and positive sums retain the original
gamma bound for their full summand count. Installation is explicit.
"""
import math
from fractions import Fraction as F
from math import comb,factorial
import numpy as np
from numba import njit
from fast_entropy import _endpoint,U,TINY,LN2LO,LN2HI

@njit(cache=True,fastmath=False)
def down(x):return np.nextafter(x,-np.inf)
@njit(cache=True,fastmath=False)
def up(x):return np.nextafter(x,np.inf)
@njit(cache=True,fastmath=False)
def add(a,b):return down(a[0]+b[0]),up(a[1]+b[1])
@njit(cache=True,fastmath=False)
def sub(a,b):return down(a[0]-b[1]),up(a[1]-b[0])
@njit(cache=True,fastmath=False)
def mul(a,b):return max(0.,down(a[0]*b[0])),up(a[1]*b[1])
@njit(cache=True,fastmath=False)
def smul(a,b):
    x0=a[0]*b[0];x1=a[0]*b[1];x2=a[1]*b[0];x3=a[1]*b[1]
    return down(min(x0,x1,x2,x3)),up(max(x0,x1,x2,x3))
@njit(cache=True,fastmath=False)
def div(a,b):return max(0.,down(a[0]/b[1])),up(a[1]/b[0])
@njit(cache=True,fastmath=False)
def positive(a):return max(0.,a[0]),max(0.,a[1])
@njit(cache=True,fastmath=False)
def clip(a):return max(0.,a[0]),min(1.,max(0.,a[1]))
@njit(cache=True,fastmath=False)
def h(a):
    lo,hi=clip(a);ll,lh=_endpoint(lo);hl,hh=_endpoint(hi)
    low=min(ll,hl);high=max(lh,hh)
    if lo<=.3678794411714424 and hi>=.3678794411714422:high=max(high,.5307378454230432)
    return max(0.,low),max(0.,high)
@njit(cache=True,fastmath=False)
def sum_adjust(lo,hi,n):
    g=up((n+1)*U/(1-(n+1)*U))
    return max(0.,down(lo/(1+g)-n*TINY)),up(hi/(1-g)+n*TINY)
@njit(cache=True,fastmath=False)
def interval(a,j,i):return a[0,j,i],a[1,j,i]

@njit(cache=True,fastmath=False)
def repetitions(n,ap,bp,upow,vpow,phases,weights,plain_cost,clip_net):
    size=ap.shape[2];out=np.empty((3,2,size));count=n//2+1
    for i in range(size):
        nl=0.;nh=0.;gl=0.;gh=0.;cl=0.;ch=0.
        h0=(phases[0,0,i],phases[0,1,i]);h1=(phases[1,0,i],phases[1,1,i])
        for w in range(count):
            aa=mul(interval(ap,n-w,i),interval(bp,w,i));bb=mul(interval(ap,w,i),interval(bp,n-w,i))
            cc=smul(interval(upow,n-w,i),interval(vpow,w,i));dd=smul(interval(upow,w,i),interval(vpow,n-w,i))
            mass=clip(add(aa,bb));hm=h(mass)
            raw=(clip(smul(add(aa,cc),(.5,.5))),clip(smul(add(bb,dd),(.5,.5))),
                 clip(smul(sub(bb,dd),(.5,.5))),clip(smul(sub(aa,cc),(.5,.5))))
            el=0.;eh=0.
            for value in raw:
                l,r=h(value);el+=l;eh+=r
            gain=positive(add(sub(mass,sum_adjust(el,eh,4)),hm))
            weight=weights[0,w],weights[1,w]
            l,r=mul(gain,weight);nl+=l;nh+=r
            ph0=positive(add(smul(h0,(float(n-w),float(n-w))),smul(h1,(float(w),float(w)))))
            ph1=positive(add(smul(h0,(float(w),float(w))),smul(h1,(float(n-w),float(n-w)))))
            physical=sub(add(add(h(aa),h(bb)),add(mul(aa,ph0),mul(bb,ph1))),hm)
            gross=positive(sub(smul(mass,(float(n),float(n))),physical))
            l,r=mul(gross,weight);gl+=l;gh+=r
            l,r=mul(hm,weight);cl+=l;ch+=r
        no=sum_adjust(nl,nh,count);gross=sum_adjust(gl,gh,count);cost=sum_adjust(cl,ch,count)
        plain=sub(gross,plain_cost);compressed=sub(gross,cost)
        if clip_net:plain=positive(plain);compressed=positive(compressed)
        for mode,value in enumerate((no,plain,compressed)):
            out[mode,0,i]=value[0];out[mode,1,i]=value[1]
    return out

def repetition(self,n,clip_net=True):
    import stable_intervals as iv
    if not hasattr(self,'_compiled_powers'):
        self._compiled_powers=[np.stack([np.stack([p[k] for p in values]) for k in (0,1)])
                               for values in (self.ap,self.bp,self.up,self.vp)]
        self._compiled_phases=np.asarray(self.phase)
    w=[iv.rational(F(comb(n,j),n*(2 if 2*j==n else 1))) for j in range(n//2+1)]
    weights=np.asarray([[v[k] for v in w] for k in (0,1)],float)
    cost=iv.rational(F(n-1,n));out=repetitions(n,*self._compiled_powers,self._compiled_phases,weights,(float(cost[0]),float(cost[1])),clip_net)
    return tuple((out[j,0],out[j,1]) for j in range(3))

FACTORIALS=np.array([1/factorial(j) for j in range(22)])
G3=up(3*U/(1-3*U));G128=up(128*U/(1-128*U));G256=up(256*U/(1-256*U));G512=up(512*U/(1-512*U))
@njit(cache=True,fastmath=False)
def exp2_scalar(x):
    safe=max(-1100.,min(0.,x));k=math.floor(safe);f=safe-k
    fl=max(0.,down(f));fh=min(1.,up(f));al=max(0.,down(fl*LN2LO));ah=up(fh*LN2HI)
    vl=FACTORIALS[20];vh=FACTORIALS[20]
    for j in range(19,-1,-1):vl=FACTORIALS[j]+al*vl;vh=FACTORIALS[j]+ah*vh
    lo=down(vl/(1+G256));hi=up(vh/(1-G256))
    tail=up(3*ah**21/51090942171709440000./(1-G128)+128*TINY);hi=up(hi+tail)
    lo=max(0.,down(math.ldexp(lo,k)));hi=up(math.ldexp(hi,k))
    return (0.,TINY) if x<-1100 else (lo,hi)
@njit(cache=True,fastmath=False)
def log1p_ratio(t):
    den=2+t;r=t/den;rr=r*r;value=1/31
    for k in range(14,-1,-1):value=1/(2*k+1)+rr*value
    value=2/den*value;rh=up(r/(1-G3))
    tail=up(2/down(den)*rh**32/(33*(1-rh*rh))/(1-G128)+128*TINY)
    return max(0.,down(value/(1+G512))),up(value/(1-G512)+tail)
@njit(cache=True,fastmath=False)
def ambiguity(values):
    low=np.empty(len(values));high=np.empty(len(values))
    for i in range(len(values)):
        L=values[i]
        if math.isinf(L):low[i]=0.;high[i]=0.;continue
        if L>1024:low[i]=0.;high[i]=exp2_scalar(-min(L,1e6)/4)[1];continue
        safe=min(1024.,max(0.,L));t=exp2_scalar(-safe);root=exp2_scalar(-safe/2)
        ratio=log1p_ratio(t[1])[0],log1p_ratio(t[0])[1]
        first=div(mul(add((1.,1.),t),ratio),(LN2LO,LN2HI))
        value=mul(mul(root,add(first,(safe,safe))),(.5,.5))
        low[i],high[i]=value
    return low,high
def ambiguity_point(L):
    L=np.asarray(L,float);out=ambiguity(L.ravel());return tuple(v.reshape(L.shape) for v in out)

def install():
    import long_code_bounds,spc_interval_dynamic
    long_code_bounds.LongBounds.repetition=repetition
    spc_interval_dynamic.ambiguity_point=ambiguity_point

def validate():
    from pathlib import Path
    import decimal,hashlib,json,time
    from full_clifford_model import HERE,bootstrap
    core,engine,model,ops=bootstrap(HERE/'diagnostics/fast_long_menu')
    from fast_entropy import install as entropy_install
    entropy_install()
    import long_code_bounds as long,spc_interval_dynamic as spc
    rng=np.random.default_rng(221026)
    p=np.vstack((rng.dirichlet([.3,.3,.3,.3],45),np.eye(4),[.25]*4,[.5,.5,0,0],[.52,.24,.24,0]))
    box=np.maximum(0,np.nextafter(p-1e-12,-np.inf)),np.minimum(1,np.nextafter(p+1e-12,np.inf))
    instance=long.LongBounds(box,128);count=0;times={}
    for n in (2,3,4,7,8,16,31,64,93,128):
        before=instance.repetition(n,False);after=repetition(instance,n,False)
        for a,b in zip(before,after):
            assert np.all(a[0]<=b[1]) and np.all(b[0]<=a[1]),n
        count+=len(p)*3
    values=np.unique(np.r_[0.,np.linspace(0,1024,1025),rng.uniform(0,20000,1000),np.inf])
    before=spc.ambiguity_point(values);after=ambiguity_point(values)
    assert np.all(before[0]<=after[1]) and np.all(after[0]<=before[1])
    # ln(1+2^-L) loses hundreds of decimal places near L=1024 if the
    # independent oracle itself uses only 100 digits.
    decimal.getcontext().prec=600;D=decimal.Decimal;ln2=D(2).ln();decimals=0
    for j in range(0,1025,7):
        L=D(float(values[j]));t=(-L*ln2).exp();root=(-L*ln2/2).exp()
        exact=root*((1+t)*(1+t).ln()/t/ln2+L)/2
        assert D(float(after[0][j]))<=exact<=D(float(after[1][j])),j
        decimals+=1
    # Compare all three repetition modes to an independent Decimal expression.
    for row in p[:12]:
        pp=[D(float(x)) for x in row];a=pp[0]+pp[3];b=pp[1]+pp[2];u=pp[0]-pp[3];v=pp[1]-pp[2]
        def ent(x):return -x*x.ln()/ln2 if x>0 else D(0)
        h0=(ent(pp[0])+ent(pp[3])-ent(a))/a if a else D(0)
        h1=(ent(pp[1])+ent(pp[2])-ent(b))/b if b else D(0)
        for n in (3,8,31,128):
            lo=np.maximum(0,np.nextafter(row[None,:],-np.inf));hi=np.nextafter(row[None,:],np.inf)
            enclosed=repetition(long.LongBounds((lo,hi),n),n,False)
            no=D(0);gross=D(0);cost=D(0)
            for w in range(n//2+1):
                aa=a**(n-w)*b**w;bb=a**w*b**(n-w);cc=u**(n-w)*v**w;dd=u**w*v**(n-w)
                mass=aa+bb;weight=D(comb(n,w))/D(n*(2 if 2*w==n else 1))
                raw=((aa+cc)/2,(bb+dd)/2,(bb-dd)/2,(aa-cc)/2)
                no+=max(D(0),mass-sum(map(ent,raw))+ent(mass))*weight
                physical=ent(aa)+ent(bb)+aa*((n-w)*h0+w*h1)+bb*(w*h0+(n-w)*h1)-ent(mass)
                gross+=max(D(0),n*mass-physical)*weight;cost+=ent(mass)*weight
            for val,bounds in zip((no,gross-D(n-1)/D(n),gross-cost),enclosed):
                assert D(float(bounds[0][0]))<=val<=D(float(bounds[1][0])),(n,val,bounds)
                decimals+=1
    sample=np.tile(p[:16],(16,1));bounds=np.maximum(0,np.nextafter(sample,-np.inf)),np.nextafter(sample,np.inf)
    test=long.LongBounds(bounds,128)
    for name,fn in [('original',lambda n:test.repetition(n)),('compiled',lambda n:repetition(test,n))]:
        start=time.monotonic()
        for n in range(8,129):fn(n)
        times[name]=time.monotonic()-start
    report=dict(status='passed',repetition_mode_cases=count,ambiguity_values=len(values),decimal_cases=decimals,
        oracle_decimal_digits=600,repetition_seconds=times,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Same directed elementary arithmetic and full-count gamma sums. Exact code formulas and complete length menu unchanged. No fastmath.')
    (HERE/'diagnostics/fast_long_menu/result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
if __name__=='__main__':validate()
