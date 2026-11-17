"""Complete binary-code coordinate classes by puncturing induction.

Every code punctured in its last coordinate either keeps dimension k, in
which case that coordinate is a linear functional of its puncture, or loses
one dimension, in which case it is the punctured code direct-summed with F2.
All functionals are tried; row operations and coordinate permutations alone
are removed by an exact canonical column multiset. No rate pruning occurs.
"""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys,json,time,hashlib
from itertools import permutations
import numpy as np

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
sys.path.insert(0,str(PROJECT/'research/joint_optimization'))
from algebra import rref,span,parity,gl_actions,code_components
DATA=HERE/'data'
DATA.mkdir(exist_ok=True)
FROZEN=PROJECT/'research/resource_extension/data/codes.json'


def save(path,obj):
    temp=path.with_suffix('.writing.json')
    temp.write_text(json.dumps(obj,indent=2),encoding='utf-8')
    temp.replace(path)


def canonical(n,rows):
    """Exact coordinate-equivalence invariant, including zero columns."""
    rows=tuple(rref(rows,n)); k=len(rows); basis=rows
    if k>n//2:
        basis=rref([x for x in range(1<<n) if all(parity(x&r)==0 for r in rows)],n)
    columns=np.array([sum(((r>>j)&1)<<i for i,r in enumerate(basis)) for j in range(n)])
    variants=np.sort(gl_actions(len(basis))[:,columns],axis=1)
    # Lexicographic minimum without building millions of Python tuples.
    for j in range(n):
        v=variants[:,j].min();variants=variants[variants[:,j]==v]
        if len(variants)==1:break
    return (n,k,tuple(map(int,variants[0])))


def entry(n,rows):
    rows=list(rref(rows,n));words=span(rows);k=len(rows)
    out=dict(n=n,k=k,d=min((x.bit_count() for x in words if x),default=0),rows=rows)
    out['components']=code_components(n,rows)
    out['indecomposable']=len(out['components'])==1
    out['endpoint']=(k==1 and out['d']==n) or (k==n-1 and out['d']==2) or n==1
    return out


def extend(n,previous,resume=True):
    checkpoint=DATA/f'enumeration_n{n}_checkpoint.json'
    signature=hashlib.sha256(json.dumps(previous,sort_keys=True).encode()).hexdigest()
    start=time.time();seen={};begin=0;tried=0
    if resume and checkpoint.exists():
        old=json.loads(checkpoint.read_text())
        assert old['parent_sha256']==signature
        begin=old['processed_parents'];tried=old['extensions_tried']
        for e in old['entries']:seen[canonical(n,e['rows'])]=e
    for index in range(begin,len(previous)):
        parent=previous[index];rows=parent['rows'];k=len(rows)
        candidates=[[row|(((mask>>i)&1)<<(n-1)) for i,row in enumerate(rows)]
                    for mask in range(1<<k)]
        candidates.append(rows+[1<<(n-1)])
        for candidate in candidates:
            key=canonical(n,candidate);tried+=1
            if key not in seen:seen[key]=entry(n,candidate)
        save(checkpoint,dict(status='enumerating',n=n,parent_sha256=signature,
                            processed_parents=index+1,total_parents=len(previous),
                            extensions_tried=tried,entries=list(seen.values())))
        if n==8 and (index+1)%10==0:
            print('n8 parents',index+1,'/',len(previous),'classes including zero',len(seen),
                  'seconds',round(time.time()-start,1),flush=True)
    result=[seen[key] for key in sorted(seen)]
    save(checkpoint,dict(status='complete',n=n,parent_sha256=signature,
                        processed_parents=len(previous),total_parents=len(previous),
                        extensions_tried=tried,entries=result,seconds=time.time()-start))
    return result,tried


def main():
    frozen=json.loads(FROZEN.read_text());old=[e for e in frozen['entries'] if e['n']<=7]
    previous=[dict(n=0,k=0,rows=[])]
    checks=[];total=[]
    for n in range(1,9):
        previous,tried=extend(n,previous)
        nonzero=[e for e in previous if e['k']]
        record=dict(n=n,nonzero_coordinate_classes=len(nonzero),extensions_tried=tried,
                    dimensions={str(k):sum(e['k']==k for e in nonzero) for k in range(1,n+1)})
        if n<=7:
            expected=[e for e in old if e['n']==n]
            actual_keys={canonical(n,e['rows']) for e in nonzero}
            expected_keys={canonical(n,e['rows']) for e in expected}
            assert actual_keys==expected_keys,(n,len(actual_keys),len(expected_keys))
            assert len(nonzero)==frozen['orbit_counts'][str(n)]
            record['frozen_signatures_exactly_match']=True
            total.extend(expected)
        else:
            for i,e in enumerate(nonzero):
                e['id']=f'N8_{i:04d}';e['construction']='complete puncture induction'
                total.append(e)
        checks.append(record)
        print(record,flush=True)
    out=dict(status='enumeration_complete',scope='All nonzero binary linear codes n<=8 modulo coordinate permutations; zero columns included.',
             proof='Puncture in the last coordinate. Equal rank gives every linear functional on the punctured representative; rank loss gives C direct sum F2. Exact GL column-multiset canonicalization, using duality when smaller, removes only coordinate equivalence.',
             maximum_length=8,frozen_source_sha256=hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
             induction_checks=checks,entries=total,
             total_coordinate_classes=len(total),indecomposable_classes=sum(e['indecomposable'] for e in total))
    save(DATA/'codes_complete_n8.json',out)
    print('complete',out['total_coordinate_classes'],out['indecomposable_classes'],flush=True)


if __name__=='__main__':main()
