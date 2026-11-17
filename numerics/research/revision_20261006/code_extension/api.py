"""Drop-in kernels and signed Du-mode values for the complete n<=8 menu.

Only this extension's data directory is written. Previous source/data files
are imported or read unchanged. Values are diagnostics/optimization inputs;
use signed_bounds for an outward enclosure of a fixed input law.
"""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys,json
from functools import lru_cache
from collections import Counter
import numpy as np
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
for folder in ('research/joint_optimization','research/completion','research/revision_20260922'):
    sys.path.insert(0,str(PROJECT/folder))
from rates import CodeKernel
from algebra import span,parity,compositions
from ad_modes import OTPKernel,MODES
DATA=HERE/'data'


def entries(max_n=8,all_codes=False):
    doc=json.loads((DATA/'codes_complete_n8.json').read_text())
    assert doc['status']=='enumeration_complete'
    return [e for e in doc['entries'] if e['n']<=max_n and (all_codes or e['indecomposable'])]


@lru_cache(None)
def physical_index(n):
    """Direct physical enumeration, vectorized but with exact integer counts."""
    types=compositions(n);look=np.zeros((n+1,n+1,n+1),dtype=int)
    for i,(identity,x,y,z) in enumerate(types):look[x,y,z]=i
    xs=np.repeat(np.arange(1<<n,dtype=np.int64),1<<n)
    zs=np.tile(np.arange(1<<n,dtype=np.int64),1<<n)
    weights=np.array([v.bit_count() for v in range(1<<n)])
    y=weights[xs&zs];x=weights[xs]-y;z=weights[zs]-y
    return xs,zs,look[x,y,z]


def build_kernel(e):
    """Same polynomial tensor contract as frozen rates.CodeKernel."""
    n,k=e['n'],e['k'];types=compositions(n);nt=len(types)
    xs,zs,typ=physical_index(n);phase=np.zeros(len(zs),dtype=np.int64)
    parity_lut=np.array([v.bit_count()&1 for v in range(1<<n)])
    for j,row in enumerate(e['rows']):phase|=parity_lut[zs&row]<<j
    flat=np.zeros((1<<(n+k))*nt,dtype=np.int32)
    np.add.at(flat,((xs<<k)|phase)*nt+typ,1)
    counts=flat.reshape(-1,nt)
    words=span(e['rows']);syndromes={}
    for x in range(1<<n):syndromes.setdefault(min(x^w for w in words),[]).append(x)
    unique={};histograms=[]
    for rep,xx in sorted(syndromes.items()):
        hist=Counter()
        for x in xx:
            rows=counts[x<<k:(x+1)<<k]
            for row in rows[np.any(rows,axis=1)]:
                key=tuple(map(int,row));index=unique.setdefault(key,len(unique));hist[index]+=1
        histograms.append((rep,hist))
    classes={};representatives={}
    for rep,hist in histograms:
        key=tuple(sorted(hist.items()));classes[key]=classes.get(key,0)+1
        representatives.setdefault(key,[]).append(rep)
    obj=CodeKernel.__new__(CodeKernel);obj.entry=e;obj.n=n;obj.k=k
    obj.types=np.asarray(types);obj.polys=np.array(list(unique),dtype=float)
    obj.hist=np.zeros((len(classes),len(unique)))
    obj.multiplicity=np.array(list(classes.values()));obj.syndrome_classes=[]
    for j,key in enumerate(classes):
        for i,mult in key:obj.hist[j,i]=mult
        obj.syndrome_classes.append(representatives[key])
    obj.marginal=obj.hist@obj.polys
    return obj


def _save_kernel(path,k):
    path.parent.mkdir(exist_ok=True)
    with path.open('wb') as f:
        np.savez_compressed(f,**{name:getattr(k,name) for name in
                           ('types','polys','hist','multiplicity','marginal')},
                            syndrome_classes=json.dumps(k.syndrome_classes))


def _load_kernel(path,e):
    k=CodeKernel.__new__(CodeKernel);k.entry=e;k.n=e['n'];k.k=e['k']
    with np.load(path,allow_pickle=False) as arrays:
        for name in ('types','polys','hist','multiplicity','marginal'):setattr(k,name,arrays[name])
        k.syndrome_classes=json.loads(str(arrays['syndrome_classes']))
    return k


def load_codes(max_n=8,all_codes=False):
    result=[]
    for e in entries(max_n,all_codes):
        path=DATA/'kernels'/f"{e['id']}.npz"
        old=PROJECT/'research/resource_extension/data/kernels'/f"{e['id']}.npz"
        if path.exists():k=_load_kernel(path,e)
        elif e['n']<=7 and old.exists():k=_load_kernel(old,e)
        else:
            k=build_kernel(e);_save_kernel(path,k)
        result.append(k)
    return result


@lru_cache(None)
def kernel_map():return {k.entry['id']:k for k in load_codes()}


@lru_cache(None)
def physical_map():
    from bb84_fixed import physical_kernel
    return {key:physical_kernel(k) for key,k in kernel_map().items()}


def actions():
    return [dict(kind='linear',code=k.entry['id'],n=k.n,mode=mode)
            for k in kernel_map().values() for mode in MODES]


def signed_values(p,menu=None):
    """All-three-mode coefficients; never clip OTP net costs for BB84."""
    if menu is None:menu=actions()
    p=np.asarray(p,dtype=float).reshape(-1,4);cache={};columns=[]
    for a in menu:
        key=a['code'];k=kernel_map()[key]
        if key not in cache:
            gain,cost=OTPKernel(k.entry).components(p);gross=np.maximum(gain,0).sum(-1)
            cache[key]=np.column_stack((k.evaluate(p),gross-(k.n-k.k)/k.n,gross-cost))
        columns.append(cache[key][:,MODES.index(a['mode'])])
    return np.column_stack(columns)


def signed_bounds(p,menu=None):
    """Outward coefficients using the existing independently audited engine."""
    from interval_modes import components
    import stable_intervals as iv
    from fractions import Fraction
    if menu is None:menu=actions()
    powers={n:iv.monomial_bounds(p,np.asarray(compositions(n))) for n in {a['n'] for a in menu}}
    cache={};lo=[];hi=[]
    for a in menu:
        key=a['code'];k=kernel_map()[key]
        if key not in cache:
            gain,_=components(powers[k.n],k)
            no=iv.positive_sum((np.maximum(gain[0],0),np.maximum(gain[1],0)))
            gain,cost=components(powers[k.n],physical_map()[key])
            gross=iv.positive_sum((np.maximum(gain[0],0),np.maximum(gain[1],0)))
            plain=iv.sub(gross,iv.rational(Fraction(k.n-k.k,k.n)))
            comp=iv.sub(gross,cost);cache[key]=(no,plain,comp)
        lower,upper=cache[key][MODES.index(a['mode'])];lo.append(lower);hi.append(upper)
    return np.column_stack(lo),np.column_stack(hi)


if __name__=='__main__':
    import time
    start=time.time();ks=load_codes()
    print(json.dumps(dict(kernels=len(ks),n8=sum(k.n==8 for k in ks),seconds=time.time()-start)),flush=True)
