"""Compiled scalar form of the existing outward atanh-series arithmetic.

No libm logarithm and no fast-math. The same finite series, gamma bounds,
analytic remainder, and directed adjacent-float enlargement are retained.
Installation is explicit and recorded by the caller; frozen modules stay intact.
"""
import math,time
import numpy as np
from numba import njit

U=2.**-53;TINY=np.nextafter(0.,1.)
LN2LO=np.nextafter(float('0.693147180559945309417232121458176568'),-np.inf)
LN2HI=np.nextafter(float('0.693147180559945309417232121458176568'),np.inf)
G2=np.nextafter(2*U/(1-2*U),np.inf)
G128=np.nextafter(128*U/(1-128*U),np.inf)
G256=np.nextafter(256*U/(1-256*U),np.inf)

@njit(cache=True,fastmath=False)
def _log(x):
    m,e=math.frexp(x);m=m*2;e=e-1
    r=(m-1)/(m+1);rh=np.nextafter(r/(1-G2),np.inf)
    rr=r*r;value=1/31
    for k in range(14,-1,-1):value=1/(2*k+1)+rr*value
    value=2*r*value
    sl=max(0.,np.nextafter(value/(1+G256)-128*TINY,-np.inf))
    tail=np.nextafter((2*rh**33/(33*(1-rh*rh)))/(1-G128),np.inf)
    sh=np.nextafter(value/(1-G256)+tail+128*TINY,np.inf)
    return (np.nextafter(e+np.nextafter(sl/LN2HI,-np.inf),-np.inf),
            np.nextafter(e+np.nextafter(sh/LN2LO,np.inf),np.inf))

@njit(cache=True,fastmath=False)
def _endpoint(x):
    if x==0:return 0.,0.
    a,b=_log(x)
    return np.nextafter(-x*b,-np.inf),np.nextafter(-x*a,np.inf)

@njit(cache=True,fastmath=False)
def _logs(values):
    lower=np.empty(len(values));upper=np.empty(len(values))
    for i in range(len(values)):lower[i],upper[i]=_log(values[i])
    return lower,upper

@njit(cache=True,fastmath=False)
def _endpoints(values):
    lower=np.empty(len(values));upper=np.empty(len(values))
    for i in range(len(values)):lower[i],upper[i]=_endpoint(values[i])
    return lower,upper

@njit(cache=True,fastmath=False)
def _entropies(low,high):
    lower=np.empty(len(low));upper=np.empty(len(low))
    for i in range(len(low)):
        lo=low[i];hi=high[i];finite=math.isfinite(hi)
        ll,lh=_endpoint(lo);hl,hh=_endpoint(hi if finite else 1.)
        lower[i]=min(ll,hl) if finite else -np.inf;upper[i]=max(lh,hh)
        if lo<=0.3678794411714424 and hi>=0.3678794411714422:
            upper[i]=max(upper[i],0.5307378454230432)
    return lower,upper

def log2_positive(x):
    x=np.asarray(x,dtype=float);assert np.all(x>0)
    out=_logs(x.ravel());return tuple(v.reshape(x.shape) for v in out)

def entropy_exact_float(x):
    x=np.asarray(x,dtype=float);out=_endpoints(x.ravel())
    return tuple(v.reshape(x.shape) for v in out)

def entropy_interval(x):
    lo,hi=np.broadcast_arrays(*x)
    assert np.all(lo>=0) and np.all(hi>=lo) and not np.isnan(hi).any()
    out=_entropies(lo.ravel(),hi.ravel());return tuple(v.reshape(lo.shape) for v in out)

def install():
    import stable_intervals as iv
    iv.log2_positive=log2_positive;iv.entropy_exact_float=entropy_exact_float
    iv.entropy_interval=entropy_interval

def validate():
    from pathlib import Path
    import hashlib,json,decimal
    from full_clifford_model import HERE,bootstrap
    core,engine,model,ops=bootstrap(HERE/'diagnostics/fast_entropy')
    import stable_intervals as iv
    rng=np.random.default_rng(20261022)
    values=np.unique(np.r_[rng.random(12000),np.exp2(rng.uniform(-1074,0,12000)),
        np.nextafter(0.,1.),np.nextafter(1.,0.),1.,np.nextafter(1.,2.),.3678794411714423])
    oldlog=iv.log2_positive(values);newlog=log2_positive(values)
    assert np.all(oldlog[0]<=newlog[1]) and np.all(newlog[0]<=oldlog[1])
    low=np.maximum(0,np.nextafter(values,-np.inf));high=np.nextafter(values,np.inf)
    original=iv.entropy_interval((low,high));fast=entropy_interval((low,high))
    assert np.all(original[0]<=fast[1]) and np.all(fast[0]<=original[1])
    decimal.getcontext().prec=100;D=decimal.Decimal;ln2=D(2).ln()
    chosen=np.unique(np.r_[np.arange(0,len(values),max(1,len(values)//600)),np.arange(min(100,len(values))),len(values)-1])
    for i in chosen:
        exact=D(float(values[i])).ln()/ln2
        assert D(float(newlog[0][i]))<=exact<=D(float(newlog[1][i]))
        for endpoint in (low[i],high[i]):
            h=-D(float(endpoint))*D(float(endpoint)).ln()/ln2 if endpoint else D(0)
            assert D(float(fast[0][i]))<=h<=D(float(fast[1][i]))
    input_=np.tile(values,8);box=(input_,np.nextafter(input_,np.inf));timings={}
    for label,fun in [('numpy',iv.entropy_interval),('compiled',entropy_interval)]:
        start=time.monotonic()
        for _ in range(3):fun(box)
        timings[label]=(time.monotonic()-start)/3
    from long_code_bounds import best_bounds
    from bb84_fixed import physical_kernel
    ks=core.load_codes();physical={k.entry['id']:physical_kernel(k) for k in ks}
    states=np.array([[.7,.1,.1,.1],[.52,.16,.16,.16],[.61,.18,.03,.18],
                     [.81,.04,.12,.03],[.99,.003,.004,.003]])
    enclosure=(np.maximum(0,np.nextafter(states,-np.inf)),np.nextafter(states,np.inf))
    before=best_bounds(enclosure,ks,physical,128);install()
    after=best_bounds(enclosure,ks,physical,128)
    assert np.all(before[0]<=after[1]) and np.all(after[0]<=before[1])
    out=dict(status='passed',probabilities=len(values),decimal_cases=len(chosen),timings=timings,
        whole_menu_states=len(states),short_code_count=len(ks),endpoint_max=128,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Same analytic atanh-series and outward binary64 error proof; JIT has fastmath=False. No assertion that two implementations have bit-identical endpoints.')
    path=HERE/'diagnostics/fast_entropy/result.json';path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out),flush=True)

if __name__=='__main__':validate()
