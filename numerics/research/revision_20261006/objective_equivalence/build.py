"""Noise-independent polynomial equivalence, never sampled-rate clustering.

Equal normalized-source U and D polynomials, up to one common integer
factor and a bit-sector-preserving Bell frame, imply identical state/yield
and all registered AD/B objectives. Retain an actually executable minimum-M
representative. This does not quotient arbitrary asymmetric BB84 sources
when mode='equal_xz'.
"""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from resource_model.model import RAW,Raw,EVENTS,AXIS_PERMUTATIONS,terminal_memory,tree_id
from resource_model.catalog import structural_catalog
from functools import lru_cache
from itertools import permutations
from math import comb,gcd,factorial
import argparse,hashlib,json,time
import numpy as np

FRAMES=tuple(p for p in permutations(range(4)) if set(p[i] for i in (0,3)) in ({0,3},{1,2}))


@lru_cache(None)
def compositions(n,k):
    if k==1:return ((n,),)
    return tuple((a,)+rest for a in range(n+1) for rest in compositions(n-a,k-1))


@lru_cache(None)
def product_indices(n,m,k):
    left,right=compositions(n,k),compositions(m,k)
    index={v:i for i,v in enumerate(compositions(n+m,k))}
    return np.array([index[tuple(a+b for a,b in zip(v,w))] for v in left for w in right],int)


def multiply(a,n,b,m,k):
    out=np.zeros(len(compositions(n+m,k)),dtype=np.int64)
    np.add.at(out,product_indices(n,m,k),np.multiply.outer(a,b).reshape(-1))
    return out


class PolynomialEvaluator:
    def __init__(self,mode='equal_xz'):
        if mode not in ('equal_xz','general'):raise ValueError(mode)
        self.mode=mode;self.variables=3 if mode=='equal_xz' else 4
        basis=compositions(1,self.variables);lookup={v:i for i,v in enumerate(basis)}
        raw=np.zeros((4,self.variables),np.int64)
        destinations=(0,1,2,1) if mode=='equal_xz' else (0,1,2,3)
        for row,col in enumerate(destinations):
            exp=tuple(int(j==col) for j in range(self.variables));raw[row,lookup[exp]]=1
        mass=raw.sum(axis=0)
        self.cache={RAW:(1,raw,mass,mass)}

    def evaluate(self,tree):
        if tree in self.cache:return self.cache[tree]
        nl,a,pa,da=self.evaluate(tree.left);nr,b,pb,db=self.evaluate(tree.right)
        n=nl+nr;k=self.variables
        raw=np.zeros((4,len(compositions(n,k))),np.int64)
        for i,j,out in EVENTS[tree.decoder]:raw[out]+=multiply(a[i],nl,b[j],nr,k)
        mass=raw.sum(axis=0)
        den=multiply(da,nl,pb,nr,k)+multiply(db,nr,pa,nl,k)
        assert np.min(raw)>=0 and np.min(den)>=0
        result=(n,raw,mass,den);self.cache[tree]=result;return result


@lru_cache(None)
def normalized_monomials(max_n,variables):
    return tuple(v for d in range(max_n+1) for v in compositions(d,variables))


@lru_cache(None)
def substitution_matrix(n,max_n,mode):
    """Integer affine substitution: I=1-2b+z, X=Z=b-z,Y=z, or I=1-x-y-z."""
    k=3 if mode=='equal_xz' else 4
    outk=2 if mode=='equal_xz' else 3
    outputs=normalized_monomials(max_n,outk);lookup={v:i for i,v in enumerate(outputs)}
    transform=np.zeros((len(compositions(n,k)),len(outputs)),np.int64)
    for row,powers in enumerate(compositions(n,k)):
        if mode=='equal_xz':
            ni,nb,ny=powers
            # (1-2b+z)^ni * (b-z)^nb * z^ny.
            for constant,bpower,zpower in compositions(ni,3):
                c=factorial(ni)//(factorial(constant)*factorial(bpower)*factorial(zpower))*(-2)**bpower
                for zextra in range(nb+1):
                    exponent=(bpower+nb-zextra,zpower+zextra+ny)
                    transform[row,lookup[exponent]]+=c*comb(nb,zextra)*(-1)**zextra
        else:
            ni,nx,ny,nz=powers
            for constant,xp,yp,zp in compositions(ni,4):
                c=factorial(ni)//(factorial(constant)*factorial(xp)*factorial(yp)*factorial(zp))*(-1)**(xp+yp+zp)
                transform[row,lookup[(nx+xp,ny+yp,nz+zp)]]+=c
    return transform


def canonical_key(raw,den):
    divisor=int(np.gcd.reduce(np.r_[raw.reshape(-1),den]))
    assert divisor>0
    raw=raw//divisor;den=den//divisor
    variants=[raw[list(p)].tobytes() for p in FRAMES]
    key,frame=min((v,i) for i,v in enumerate(variants))
    return den.tobytes()+key,frame,divisor


def build(mode='equal_xz',max_leaves=5,max_pump_checks=7):
    start=time.time();items=structural_catalog(max_leaves,max_pump_checks)
    maximum=max(max_leaves,max_pump_checks+1)
    coefficient_bound=maximum*16**maximum
    assert coefficient_bound < 2**63, 'Requested polynomial degree exceeds the proved int64 bound'
    engine=PolynomialEvaluator(mode)
    keys=[];frames=[];divisors=[];memories=[];ids=[];axes=('Z','X','Y')
    for tree,_ in items:
        n,raw,mass,den=engine.evaluate(tree)
        transform=substitution_matrix(n,maximum,mode)
        normalized_raw=raw@transform;normalized_den=den@transform
        for axis in axes:
            oriented=normalized_raw[list(AXIS_PERMUTATIONS[axis])]
            key,frame,divisor=canonical_key(oriented,normalized_den)
            keys.append(key);frames.append(frame);divisors.append(divisor)
            memories.append(terminal_memory(tree,axis));ids.append(tree_id(tree))
    result={}
    for permission,allowed in (('real',(0,1)),('full',(0,1,2))):
        original_indices=[i for i in range(len(keys)) if i%3 in allowed]
        best={}
        for local,i in enumerate(original_indices):
            if keys[i] not in best or (memories[i],local)<best[keys[i]][0]:
                best[keys[i]]=((memories[i],local),local)
        mapping=[best[keys[i]][1] for i in original_indices]
        reps=sorted(set(mapping))
        assert all(memories[original_indices[r]]<=memories[original_indices[i]] for i,r in enumerate(mapping))
        result[permission]=dict(record_count=len(mapping),representative_count=len(reps),
                representative_indices=reps,representative_for_record=mapping,
                canonical_frame_indices=[frames[i] for i in original_indices],
                common_integer_divisors=[divisors[i] for i in original_indices],
                memory_for_record=[memories[i] for i in original_indices])
    resource_sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((HERE.parent/'resource_model').glob('*.py'))}
    summary=dict(schema_version=1,status='complete',source_family=mode,
                 source_parameterization='(1-2b+z,b-z,z,b-z)' if mode=='equal_xz' else '(1-x-y-z,x,y,z)',
                 max_leaves=max_leaves,max_pump_checks=max_pump_checks,
                 maximum_absolute_coefficient_bound=coefficient_bound,
                 raw_D_leaf='sum_of_Bell_probabilities',integer_arithmetic='int64;finite-degree coefficient bounds independently checked',
                 equivalence='exact_polynomial_U_and_D_after_normalization;common_integer_scale;bit_sector_preserving_frames',
                 frames=FRAMES,permissions=result,resource_source_sha256=resource_sources,
                 coefficients_checked=4*sum(len(v[1][0]) for v in engine.cache.values()),
                 seconds=time.time()-start)
    path=HERE/f'{mode}_map.json'
    encoded=json.dumps(summary,separators=(',',':')).encode();path.write_bytes(encoded)
    manifest={k:v for k,v in summary.items() if k!='permissions'}
    manifest['permissions']={k:{name:v[name] for name in ('record_count','representative_count')} for k,v in result.items()}
    manifest['map_sha256']=hashlib.sha256(encoded).hexdigest();manifest['map_file']=path.name
    (HERE/f'{mode}_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='resource_source_sha256'},indent=2))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('equal_xz','general'),default='equal_xz')
    p.add_argument('--max-leaves',type=int,default=5);p.add_argument('--max-pump-checks',type=int,default=7)
    a=p.parse_args();build(a.mode,a.max_leaves,a.max_pump_checks)
