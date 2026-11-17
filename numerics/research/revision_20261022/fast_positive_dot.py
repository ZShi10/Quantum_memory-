"""Sparse nonnegative polynomial contractions with the same gamma bound.

The exact kernel coefficient matrices contain many zeros. A scalar CSR sum
evaluates the same polynomial, with no fast-math and no cancellation. The
original gamma_(2*n+2) and n*minimum-subnormal error allowance are retained,
using the full inner dimension n rather than the smaller nonzero count.
"""
from collections import OrderedDict
import numpy as np
from numba import njit

_cache=OrderedDict();_original=None

@njit(cache=True,fastmath=False)
def contract(low,high,indices,weights,offsets,g,tiny,n):
    outlow=np.empty((len(low),len(offsets)-1));outhigh=np.empty_like(outlow)
    for row in range(len(low)):
        for col in range(len(offsets)-1):
            lo=0.;hi=0.
            for term in range(offsets[col],offsets[col+1]):
                index=indices[term];weight=weights[term]
                lo=lo+low[row,index]*weight;hi=hi+high[row,index]*weight
            outlow[row,col]=max(0.,np.nextafter(lo/(1+g)-n*tiny,-np.inf))
            outhigh[row,col]=np.nextafter(hi/(1-g)+n*tiny,np.inf)
    return outlow,outhigh

def positive_dot(a,b):
    import stable_intervals as iv
    source=np.asarray(b);shape=source.shape
    if source.ndim!=2 or source.size<128:return _original(a,b)
    key=(source.__array_interface__['data'][0],shape,source.strides,source.dtype.str)
    # Pointer identity alone is insufficient: callers may mutate a matrix.
    # Retain and compare its coefficients before reusing a CSR representation.
    if key not in _cache or not np.array_equal(source,_cache[key][1]):
        matrix=np.asarray(source,dtype=float);assert np.all(matrix>=0)
        nonzero=np.count_nonzero(matrix)
        if nonzero>.24*matrix.size:record=(source,matrix.copy(),None)
        else:
            indices=[];values=[];offsets=[0]
            for col in range(shape[1]):
                ids=np.flatnonzero(matrix[:,col]);indices.extend(ids);values.extend(matrix[ids,col]);offsets.append(len(indices))
            record=(source,matrix.copy(),(np.array(indices,np.int32),np.array(values,float),np.array(offsets,np.int64)))
        _cache[key]=record
        if len(_cache)>2048:_cache.popitem(last=False)
    else:_cache.move_to_end(key)
    _,_,csr=_cache[key]
    if csr is None:return _original(a,b)
    low,high=np.broadcast_arrays(*a);assert low.shape[-1]==shape[0]
    n=shape[0];out=contract(low.reshape(-1,n),high.reshape(-1,n),*csr,iv.gamma(2*n+2),iv.TINY,n)
    return tuple(v.reshape(low.shape[:-1]+(shape[1],)) for v in out)

def install():
    global _original
    import stable_intervals as iv
    if iv.positive_dot is positive_dot:return
    _original=iv.positive_dot;iv.positive_dot=positive_dot

def validate():
    from pathlib import Path
    import decimal,hashlib,json,time
    from full_clifford_model import HERE,bootstrap
    core,engine,model,ops=bootstrap(HERE/'diagnostics/fast_positive_dot')
    import stable_intervals as iv
    from fast_entropy import install as install_entropy
    from fast_short_modes import install as install_modes
    install_entropy();install_modes();original=iv.positive_dot
    rng=np.random.default_rng(20261022);decimal.getcontext().prec=110;D=decimal.Decimal
    before=[];cases=[]
    for n,width in [(5,30),(21,80),(165,160),(330,360)]:
        matrix=rng.integers(0,16,(n,width)).astype(float);matrix[rng.random(matrix.shape)<.96]=0
        point=np.exp2(rng.uniform(-1000,0,(3,n)));lo=np.maximum(0,np.nextafter(point,-np.inf));hi=np.nextafter(point,np.inf)
        cases.append(((lo,hi),matrix));before.append(original((lo,hi),matrix))
    install();checked=0
    for (interval,matrix),prior in zip(cases,before):
        result=positive_dot(interval,matrix)
        assert np.all(prior[0]<=result[1]) and np.all(result[0]<=prior[1])
        for row in range(3):
            for col in np.linspace(0,matrix.shape[1]-1,15,dtype=int):
                for side in (0,1):
                    exact=sum((D(float(interval[side][row,k]))*D(float(matrix[k,col])) for k in range(matrix.shape[0])),D(0))
                    assert D(float(result[0][row,col]))<=exact<=D(float(result[1][row,col]));checked+=1
    matrix=cases[-1][1];matrix[0,0]+=7
    changed=positive_dot(cases[-1][0],matrix);reference=original(cases[-1][0],matrix)
    assert np.all(changed[0]<=reference[1]) and np.all(reference[0]<=changed[1])
    # A real complete-menu comparison checks shapes, coefficient ordering,
    # sparse histogram contractions, and paid/unmasked paths together.
    from long_code_bounds import best_bounds
    from bb84_fixed import physical_kernel
    ks=core.load_codes();physical={k.entry['id']:physical_kernel(k) for k in ks}
    p=rng.dirichlet([.6,.6,.6,.6],24);box=(np.maximum(0,np.nextafter(p,-np.inf)),np.nextafter(p,np.inf))
    iv.positive_dot=original;start=time.monotonic();old=best_bounds(box,ks,physical,16);oldtime=time.monotonic()-start
    iv.positive_dot=positive_dot;start=time.monotonic();new=best_bounds(box,ks,physical,16);newtime=time.monotonic()-start
    assert np.all(old[0]<=new[1]) and np.all(new[0]<=old[1])
    report=dict(status='passed',decimal_dot_cases=checked,whole_menu_states=len(p),codes=len(ks),
        old_seconds=oldtime,new_seconds=newtime,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Same nonnegative polynomial and full-dimension gamma error bound, CSR summation with fastmath=False. No rounding-based equality of different statistics.')
    path=HERE/'diagnostics/fast_positive_dot/result.json';path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':validate()
