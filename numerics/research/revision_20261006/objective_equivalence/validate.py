"""Independent Python-integer symbolic proof of EVERY declared alias.

Unlike build.py, substitute the normalized source first, operate on sparse
signed polynomials throughout, and do not homogenize. This checks exact
coefficients, not rates sampled at finitely many source laws.
"""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from resource_model.model import RAW,Raw,EVENTS,AXIS_PERMUTATIONS,terminal_memory,evaluate_exact
from resource_model.catalog import structural_catalog
from objective_equivalence.build import normalized_monomials,FRAMES
from fractions import Fraction as F
from functools import lru_cache
import argparse,hashlib,json,time
import numpy as np


def add(*polys):
    out={}
    for p in polys:
        for key,value in p.items():out[key]=out.get(key,0)+value
    return {k:v for k,v in out.items() if v}


def multiply(a,b):
    out={}
    for left,x in a.items():
        for right,y in b.items():
            key=tuple(i+j for i,j in zip(left,right));out[key]=out.get(key,0)+x*y
    return {k:v for k,v in out.items() if v}


class IndependentPolynomials:
    def __init__(self,mode):
        if mode=='equal_xz':
            one={(0,0):1};x={(1,0):1,(0,1):-1};y={(0,1):1}
            initial=({(0,0):1,(1,0):-2,(0,1):1},x,y,x)
        else:
            one={(0,0,0):1}
            initial=({(0,0,0):1,(1,0,0):-1,(0,1,0):-1,(0,0,1):-1},
                     {(1,0,0):1},{(0,1,0):1},{(0,0,1):1})
        self.cache={RAW:(initial,one,one)}

    def evaluate(self,tree):
        if tree not in self.cache:
            a,pa,da=self.evaluate(tree.left);b,pb,db=self.evaluate(tree.right)
            raw=[{} for _ in range(4)]
            for i,j,k in EVENTS[tree.decoder]:raw[k]=add(raw[k],multiply(a[i],b[j]))
            self.cache[tree]=(raw,add(*raw),add(multiply(da,pb),multiply(db,pa)))
        return self.cache[tree]


def numeric(poly,variables):
    return sum(F(coefficient)*np.prod([v**e for v,e in zip(variables,powers)]) for powers,coefficient in poly.items())


def validate(mode='equal_xz'):
    start=time.time();path=HERE/f'{mode}_map.json';encoded=path.read_bytes();data=json.loads(encoded)
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((HERE.parent/'resource_model').glob('*.py'))}
    assert source_hashes==data['resource_source_sha256'], 'Resource source changed'
    trees=[t for t,_ in structural_catalog(data['max_leaves'],data['max_pump_checks'])]
    degree=max(data['max_leaves'],data['max_pump_checks']+1)
    monomials=normalized_monomials(degree,2 if mode=='equal_xz' else 3)
    engine=IndependentPolynomials(mode)
    polynomial_checks=0;memory_checks=0;max_coefficient=0
    for permission,axes in (('real',('Z','X')),('full',('Z','X','Y'))):
        rows=data['permissions'][permission];repkeys={}
        def signature(index):
            nonlocal max_coefficient
            tree=trees[index//len(axes)];axis=axes[index%len(axes)]
            raw,mass,den=engine.evaluate(tree)
            frame=FRAMES[rows['canonical_frame_indices'][index]]
            permutation=AXIS_PERMUTATIONS[axis]
            order=[permutation[j] for j in frame]
            divisor=rows['common_integer_divisors'][index]
            coefficients=tuple(den.get(m,0) for m in monomials)+tuple(raw[j].get(m,0) for j in order for m in monomials)
            assert all(c%divisor==0 for c in coefficients)
            max_coefficient=max(max_coefficient,max(map(abs,coefficients)))
            return tuple(c//divisor for c in coefficients)
        for index in rows['representative_indices']:repkeys[index]=signature(index)
        for index,rep in enumerate(rows['representative_for_record']):
            assert signature(index)==repkeys[rep], (permission,index,rep)
            tree=trees[index//len(axes)];rep_tree=trees[rep//len(axes)]
            m=terminal_memory(tree,axes[index%len(axes)])
            mr=terminal_memory(rep_tree,axes[rep%len(axes)])
            assert mr<=m and m==rows['memory_for_record'][index]
            polynomial_checks+=1;memory_checks+=1
    # An independent rational evaluator computes the normalized Bell law and
    # expected retry cost, without any polynomial homogenization/substitution.
    rational_checks=0
    source_points=([(F(1,5),F(1,20)),(F(6,25),F(1,100)),(F(3,20),F(3,20))]
                   if mode=='equal_xz' else [(F(1,10),F(1,20),F(3,20)),(F(1,5),F(0),F(1,4))])
    for tree in trees[::127]:
        raw,mass,den=engine.evaluate(tree)
        for point in source_points:
            p=((1-2*point[0]+point[1],point[0]-point[1],point[1],point[0]-point[1])
               if mode=='equal_xz' else (1-sum(point),*point))
            expected=evaluate_exact(tree,p)
            pp=numeric(mass,point);dd=numeric(den,point)
            assert tuple(numeric(v,point)/pp for v in raw)==expected['state']
            assert dd/pp==expected['expected_raw_cost']
            rational_checks+=1
    assert max_coefficient<=degree*16**degree<2**63
    out=dict(status='passed',source_family=mode,map_sha256=hashlib.sha256(encoded).hexdigest(),
             exact_independent_polynomial_alias_checks=polynomial_checks,
             minimum_executable_memory_checks=memory_checks,
             rational_state_and_retry_cost_checks=rational_checks,
             largest_absolute_coefficient=max_coefficient,
             proved_absolute_coefficient_bound=degree*16**degree,
             integer_implementation='Independent Python arbitrary-precision signed sparse polynomials; substitute first, no homogeneous vectors',
             numerical_sampling_is_not_equivalence_proof=True,seconds=time.time()-start)
    (HERE/f'{mode}_validation.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2));return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('equal_xz','general'),default='equal_xz')
    validate(p.parse_args().mode)
