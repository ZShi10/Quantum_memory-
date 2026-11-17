"""Complete m=5 terminal Clifford instruments, quotienting IID input permutations.

Real bases are the maximal totally singular subspaces of the Pauli quadratic
form. Full bases are all maximal symplectic isotropic subspaces. Generation
uses symmetric graph matrices and every local-H mask; completeness is also
checked against their exact finite-geometry cardinalities. No input local
Clifford frame is removed from a BB84-compatible family.
"""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import sys,json,time,hashlib,argparse
from itertools import combinations,permutations
import numpy as np
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3]
sys.path.insert(0,str(PROJECT/'research/joint_optimization'))
sys.path.insert(0,str(PROJECT/'research/completion'))
from algebra import rref,span,symp,quadratic,compositions,physical_types
from streaming_clifford import compile_measurement
from compile_graph_library import exact_replay
from verify_streaming import dense_check
DATA=HERE/'data';DATA.mkdir(parents=True,exist_ok=True)


def save(path,obj):
    temp=path.with_suffix('.writing.json');temp.write_text(json.dumps(obj,separators=(',',':')),encoding='utf-8');temp.replace(path)


def bases(m,permission):
    path=DATA/f'bases_{permission}_{m}.json'
    if path.exists():return [tuple(x) for x in json.loads(path.read_text())['bases']]
    start=time.time();edges=list(combinations(range(m),2));seen=set();mask=(1<<m)-1
    diagonals=1 if permission=='real' else 1<<m
    for graph in range(1<<len(edges)):
        base=[1<<j for j in range(m)]
        for j,(a,b) in enumerate(edges):
            if graph>>j&1:base[a]^=1<<(m+b);base[b]^=1<<(m+a)
        for diagonal in range(diagonals):
            rows=[row^(((diagonal>>j)&1)<<(m+j)) for j,row in enumerate(base)]
            for hm in range(1<<m):
                transformed=[]
                for row in rows:
                    x=row&mask;z=row>>m;v=(x^z)&hm
                    transformed.append((x^v)|((z^v)<<m))
                seen.add(rref(transformed,2*m))
        if m==5 and permission=='full' and (graph+1)%128==0:
            print('full graph generation',graph+1,'/1024 bases',len(seen),'seconds',round(time.time()-start,1),flush=True)
    expected=(2 if permission=='real' else 1)
    for j in range(1,m if permission=='real' else m+1):expected*=2**j+1
    assert len(seen)==expected,(permission,m,len(seen),expected)
    result=sorted(seen)
    assert all(all(symp(a,b,m)==0 for a in rows for b in rows) for rows in result)
    if permission=='real':assert all(all(quadratic(a,m)==0 for a in rows) for rows in result)
    save(path,dict(status='complete',m=m,permission=permission,bases=result,expected_count=expected,seconds=time.time()-start))
    return result


def swap_wire(p,a,b,m):
    x=((p>>a)^(p>>b))&1;z=((p>>(m+a))^(p>>(m+b)))&1
    return p^(x<<a)^(x<<b)^(z<<(m+a))^(z<<(m+b))


def permutation_orbits(allbases,m):
    remaining=set(allbases);representatives=[];orbit_sizes=[]
    while remaining:
        seed=min(remaining);orbit={seed};queue=[seed]
        while queue:
            rows=queue.pop()
            for a in range(m-1):
                nxt=rref([swap_wire(p,a,a+1,m) for p in rows],2*m)
                if nxt not in orbit:orbit.add(nxt);queue.append(nxt)
        assert orbit<=remaining
        remaining.difference_update(orbit);representatives.append(seed);orbit_sizes.append(len(orbit))
    assert sum(orbit_sizes)==len(allbases)
    return representatives,orbit_sizes


def prepare(m):
    count=1<<(2*m);sy=np.empty((count,count),dtype=np.uint8)
    for p in range(count):sy[p]=[symp(p,g,m) for g in range(count)]
    types=compositions(m);lookup={v:i for i,v in enumerate(types)}
    type_index=np.array([lookup[v] for v in physical_types(m)])
    return sy,type_index,types


def tensor_for(rows,v,m,sy,type_index,nt):
    j=(v&-v).bit_length()-1
    checks=[rows[i]^(rows[j] if v>>i&1 else 0) for i in range(m) if i!=j]
    lz=rows[j]
    logical_candidates=[p for p in range(1<<(2*m)) if all(int(sy[p,g])==((v>>i)&1) for i,g in enumerate(rows))]
    real=all(quadratic(g,m)==0 for g in rows)
    lx=next((p for p in logical_candidates if not real or quadratic(p,m)==0),None)
    assert lx is not None and symp(lx,lz,m)==1
    x=sy[:,lz].astype(int);z=sy[:,lx].astype(int)
    syndrome=np.zeros(len(x),dtype=int)
    for i,g in enumerate(checks):syndrome|=sy[:,g].astype(int)<<i
    label=np.array([[0,3],[1,2]])[x,z]
    tensor=np.bincount((syndrome*4+label)*nt+type_index,minlength=(1<<(m-1))*4*nt).reshape(-1,4,nt).astype(np.uint16)
    return tensor,checks,lx,lz,j


FRAMES=[p for p in permutations(range(4)) if set(p[i] for i in (0,3)) in ({0,3},{1,2})]


def tensor_key(tensor):
    # Frame changes are only affine bit/phase corrections and Z->Z+X:
    # key bit sectors are never exchanged with phase sectors.
    return b''.join(sorted(min(row[list(frame)].tobytes() for frame in FRAMES) for row in tensor))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--permission',choices=['real','full'],default='real');parser.add_argument('--m',type=int,default=5);args=parser.parse_args()
    m=args.m;mode=args.permission;start=time.time();allbases=bases(m,mode);reps,sizes=permutation_orbits(allbases,m)
    print('Basis S_m orbits',mode,m,len(reps),'from',len(allbases),flush=True)
    sy,types_index,types=prepare(m);nt=len(types)
    path=DATA/f'instruments_{mode}_m{m}.json';arrays=DATA/f'tensors_{mode}_m{m}.npz'
    checkpoint=DATA/f'checkpoint_{mode}_m{m}.json'
    compilers={};seen={};records=[];tensors=[];first=0;dense_error=0.;raw=0
    # Each completed orbit is checkpointed. Separate tensor checkpoint
    # avoids allocating a large text serialization of integer arrays.
    if checkpoint.exists() and arrays.exists():
        old=json.loads(checkpoint.read_text());first=old['processed_basis_orbits'];compilers=old['compilers'];records=old['entries']
        dense_error=old['max_dense_error'];raw=old['raw_choices_evaluated']
        with np.load(arrays) as saved:tensors=list(saved['tensors'])
        for i,tensor in enumerate(tensors):seen[tensor_key(tensor)]=i
    for basisindex in range(first,len(reps)):
        rows=reps[basisindex];key='_'.join(map(str,rows));real=all(quadratic(g,m)==0 for g in rows)
        compiled=compile_measurement(rows,m,real=real);exact_replay(compiled,m)
        dense_error=max(dense_error,dense_check(compiled,m));compilers[key]=compiled
        for v in range(1,1<<m):
            tensor,checks,lx,lz,j=tensor_for(rows,v,m,sy,types_index,nt);canonical=tensor_key(tensor);raw+=1
            record=dict(m=m,M=compiled['M'],real=mode=='real',permission=mode,basis_real=real,
                        checks=checks,logical_X=lx,logical_Z=lz,
                        measurement_key=key,logical_dual_vector=v,key_generator_index=j,
                        key_generator_mask=1<<j,public_generator_masks=[(1<<i)|(((v>>i)&1)<<j) for i in range(m) if i!=j],
                        basis_orbit_size=sizes[basisindex],multiplicity=1,
                        terminal_implementation='exact_streaming',zero_flag_index=0)
            if canonical in seen:
                index=seen[canonical];old=records[index];multiplicity=old['multiplicity']+1
                if (record['M'],len(compiled['gates']))<(old['M'],len(compilers[old['measurement_key']]['gates'])):
                    records[index]=record;tensors[index]=tensor
                records[index]['multiplicity']=multiplicity
            else:seen[canonical]=len(records);records.append(record);tensors.append(tensor)
        if (basisindex+1)%25==0 or basisindex+1==len(reps):
            with arrays.open('wb') as f:np.savez_compressed(f,tensors=np.asarray(tensors,dtype=np.uint16))
            save(checkpoint,dict(status='enumerating',processed_basis_orbits=basisindex+1,total_basis_orbits=len(reps),
                                 compilers=compilers,entries=records,max_dense_error=dense_error,raw_choices_evaluated=raw))
            print(mode,'orbit',basisindex+1,'/',len(reps),'distinct instruments',len(records),'seconds',round(time.time()-start,1),flush=True)
    for i,e in enumerate(records):e['id']=f'BB{m}{mode[0].upper()}_{i:05d}';e['tensor_index']=i
    out=dict(status='enumeration_complete',m=m,permission=mode,complete_basis_count=len(allbases),basis_input_permutation_orbits=len(reps),
             input_permutation_orbit_sizes=sizes,raw_key_check_choices=raw,labeled_key_check_choices=len(allbases)*((1<<m)-1),distinct_instruments=len(records),entries=records,compilers=compilers,
             max_dense_error=dense_error,minimum_memories=sorted({e['M'] for e in records}),
             scope='All m-input complete rank-one Clifford measurements with one key bit and m-1 public relative flags, modulo identical-input permutations and bit-sector-preserving conditional Pauli frames. No arbitrary input Clifford frames removed. Full four-variable Bell tensors.',
             tensor_file=arrays.name,tensor_sha256=hashlib.sha256(arrays.read_bytes()).hexdigest(),seconds=time.time()-start)
    save(path,out);print('COMPLETE',mode,m,len(records),'M',out['minimum_memories'],'error',dense_error,flush=True)


if __name__=='__main__':main()
