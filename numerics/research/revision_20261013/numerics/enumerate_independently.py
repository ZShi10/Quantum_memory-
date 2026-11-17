"""Independent finite enumerations, with no imported rate-based pruning.

1. All labelled graphs through N=7 under local complementations/permutations.
2. All real maximal stabilizer bases through N=5, generated as graph+local H.
3. All nonzero binary subspaces through n=8 under coordinate permutations.
4. Explicit sequential schedules for every supplied graph representative N<=8.

N=8 graph-catalog completeness remains an inherited mathematical input.
"""
from pathlib import Path
from collections import Counter, deque
from itertools import combinations
import hashlib
import json
import sys
import time
import numpy as np
from numba import njit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reproduction/controls/code'))
sys.path.insert(0,str(ROOT/'reproduction/extensions/code'))
from stopping import graph_data, rank
from compile_winner import compile_order, multiply, gate

@njit(cache=True)
def graph_orbits(n,edges,swaps):
    total=1<<len(edges)
    labels=np.full(total,255,dtype=np.uint8)
    queue=np.empty(total,dtype=np.int32)
    edgeindex=np.full((n,n),-1,dtype=np.int32)
    for e in range(len(edges)):
        i,j=edges[e];edgeindex[i,j]=e;edgeindex[j,i]=e
    counts=[]
    for seed in range(total):
        if labels[seed]!=255:continue
        oid=len(counts);head=0;tail=1;queue[0]=seed;labels[seed]=oid
        while head<tail:
            g=queue[head];head+=1
            adj=np.zeros(n,dtype=np.int32)
            for e in range(len(edges)):
                if (g>>e)&1:
                    i,j=edges[e];adj[i]|=1<<j;adj[j]|=1<<i
            for v in range(n):
                nxt=g
                for i in range(n):
                    if (adj[v]>>i)&1:
                        for j in range(i+1,n):
                            if (adj[v]>>j)&1:nxt^=1<<edgeindex[i,j]
                if labels[nxt]==255:labels[nxt]=oid;queue[tail]=nxt;tail+=1
            for s in range(n-1):
                nxt=0
                for e in range(len(edges)):
                    if (g>>e)&1:nxt|=1<<swaps[s,e]
                if labels[nxt]==255:labels[nxt]=oid;queue[tail]=nxt;tail+=1
        counts.append(tail)
    return labels,counts

def rref(rows,width):
    rows=list(rows);k=0
    for j in range(width):
        pivot=next((i for i in range(k,len(rows)) if rows[i]>>j&1),None)
        if pivot is None:continue
        rows[k],rows[pivot]=rows[pivot],rows[k]
        for i in range(len(rows)):
            if i!=k and rows[i]>>j&1:rows[i]^=rows[k]
        k+=1
    return tuple(rows[:k])

@njit(cache=True)
def code_neighbors(key,n,k):
    answer=np.zeros(n-1,dtype=np.int64)
    mask=(1<<n)-1
    for j in range(n-1):
        rows=np.empty(k,dtype=np.int64)
        for i in range(k):
            row=(key>>(n*i))&mask
            if ((row>>j)^(row>>(j+1)))&1:row^=(1<<j)|(1<<(j+1))
            rows[i]=row
        pivot=0
        for col in range(n):
            ix=-1
            for i in range(pivot,k):
                if (rows[i]>>col)&1:ix=i;break
            if ix<0:continue
            tmp=rows[pivot];rows[pivot]=rows[ix];rows[ix]=tmp
            for i in range(k):
                if i!=pivot and (rows[i]>>col)&1:rows[i]^=rows[pivot]
            pivot+=1
        packed=0
        for i in range(k):packed|=rows[i]<<(n*i)
        answer[j]=packed
    return answer

def code_seeds(n,k):
    for pivots in combinations(range(n),k):
        base=sum((1<<p)<<(n*i) for i,p in enumerate(pivots))
        free=[(1<<j)<<(n*i) for i,p in enumerate(pivots) for j in range(p+1,n) if j not in pivots]
        key=base;previous=0
        for val in range(1<<len(free)):
            gray=val^(val>>1)
            if val:key^=free[(gray^previous).bit_length()-1]
            previous=gray
            yield key

def gauss_binomial(n,k):
    num=den=1
    for i in range(k):num*=2**(n-i)-1;den*=2**(k-i)-1
    return num//den

def run_codes(catalog):
    records=[];global_ids={};total_classes=0
    for n in range(1,9):
        nc=0;nl=0;dimensions={}
        for k in range(1,n+1):
            seen={};sizes=[]
            for seed in code_seeds(n,k):
                if seed in seen:continue
                oid=len(sizes);queue=[seed];seen[seed]=oid
                for key in queue:
                    if k==n:continue
                    for neighbor in code_neighbors(key,n,k):
                        nxt=int(neighbor)
                        if nxt not in seen:seen[nxt]=oid;queue.append(nxt)
                sizes.append(len(queue))
            assert len(seen)==gauss_binomial(n,k)
            supplied=[e for e in catalog['entries'] if e['n']==n and e['k']==k]
            covered=[]
            for e in supplied:
                rows=rref(e['rows'],n)
                key=sum(row<<(n*i) for i,row in enumerate(rows))
                covered.append(seen[key])
                global_ids[e['id']]=dict(n=n,k=k,orbit=seen[key],labeled_multiplicity=sizes[seen[key]])
            assert len(covered)==len(sizes) and len(set(covered))==len(sizes),(n,k,len(covered),len(sizes))
            nc+=len(sizes);nl+=len(seen)
            dimensions[str(k)]=dict(labeled=len(seen),orbits=len(sizes))
        records.append(dict(n=n,nonzero_subspaces=nl,coordinate_classes=nc,dimensions=dimensions))
        total_classes+=nc
        print('Binary codes',n,'subspaces',nl,'classes',nc,flush=True)
    assert total_classes==612
    return dict(total_classes=total_classes,by_length=records,catalog_members=global_ids)

def run_real_bases():
    out=[]
    for n in range(1,6):
        edges=list(combinations(range(n),2));bases=set()
        for graph in range(1<<len(edges)):
            adj=[0]*n
            for e,(i,j) in enumerate(edges):
                if graph>>e&1:adj[i]|=1<<j;adj[j]|=1<<i
            original=[(1<<i)|(adj[i]<<n) for i in range(n)]
            for hs in range(1<<n):
                rows=[]
                for row in original:
                    x=row&((1<<n)-1);z=row>>n;change=(x^z)&hs
                    rows.append((x^change)|((z^change)<<n))
                b=rref(rows,2*n)
                assert len(b)==n
                assert all(((v&((1<<n)-1))&(v>>n)).bit_count()%2==0 for v in b)
                bases.add(b)
        expected=2
        for j in range(1,n):expected*=2**j+1
        assert len(bases)==expected
        # Store canonical basis list, not only the claimed count.
        (ROOT/'data'/f'real_bases_N{n}.json').write_text(json.dumps(sorted(bases))+'\n',encoding='utf-8')
        out.append(dict(N=n,complete_real_bases=len(bases),expected_orthogonal_space_count=expected,
                        raw_key_check_splits=len(bases)*(2**n-1)))
        print('Real stabilizer bases',n,len(bases),flush=True)
    return out

def run_graphs(catalog,nmax=7):
    coverage=[]
    for n in range(1,nmax+1):
        edge_list=list(combinations(range(n),2));edge_index={e:i for i,e in enumerate(edge_list)}
        edges=np.array(edge_list,dtype=np.int32).reshape((-1,2))
        swaps=[]
        for j in range(n-1):
            perm=list(range(n));perm[j],perm[j+1]=perm[j+1],perm[j]
            swaps.append([edge_index[tuple(sorted((perm[a],perm[b])))] for a,b in edge_list])
        swaps=np.array(swaps,dtype=np.int32).reshape((n-1,len(edge_list)))
        labels,counts=graph_orbits(n,edges,swaps)
        supplied=[g for g in catalog.values() if g['N']==n]
        ids=[int(labels[sum(1<<edge_index[tuple(sorted(e))] for e in g['edges'])]) for g in supplied]
        assert len(ids)==len(counts) and len(set(ids))==len(counts)
        coverage.append(dict(N=n,labeled_graphs=len(labels),orbits=len(counts),orbit_sizes=counts,
                             supplied_representatives_cover_every_orbit_once=True))
        print('Graph LC/permutation orbits',n,len(labels),len(counts),flush=True)
    return coverage

def schedules(catalog):
    hist={};out={}
    for name,g in catalog.items():
        n=g['N'];adj=[0]*n
        for i,j in g['edges']:adj[i]|=1<<j;adj[j]|=1<<i
        _,_,cuts,minimum=graph_data(adj);S=1<<n
        rows=[(1<<i)|(adj[i]<<n) for i in range(n)]
        for a in range(S):
            outside=(S-1)^a;restrict=outside|(outside<<n)
            stabilizer_E=a.bit_count()-n+rank([r&restrict for r in rows])
            assert stabilizer_E==cuts[a]
        dp=[n]*S;parent=[None]*S;dp[0]=0
        for a in range(1,S):
            best=min((max(dp[a^(1<<v)],cuts[a]),v) for v in range(n) if a>>v&1)
            dp[a],parent[a]=best
        a=S-1;order=[]
        while a:
            v=parent[a];order.append(v);a^=1<<v
        order.reverse();compiled=compile_order(adj,order)
        assert compiled['peak_memory']==minimum==g['original_min_memory']
        # Symbolic signed conjugation of every original-generator combination.
        for measurement in compiled['measurements']:
            target=(0,0,0,0)
            for i in range(n):
                if measurement['original_generator_mask']>>i&1:
                    target=multiply(target,(1<<i,adj[i],0,1<<i))
            for operation in compiled['gates']:target=gate(target,operation)
            assert target[:3]==(0,1<<measurement['wire'],measurement['offset'])
        hist.setdefault(str(n),Counter())[str(minimum)]+=1
        out[name]=dict(N=n,edges=g['edges'],minimum_memory=minimum,**compiled)
    (ROOT/'data/measurement_schedules.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    return {n:dict(v) for n,v in hist.items()}

def main():
    start=time.monotonic();(ROOT/'data').mkdir(exist_ok=True)
    cat=json.loads((ROOT/'reproduction/controls/data/graph_representatives.json').read_text())['graphs']
    codes_path=ROOT/'data/codes_complete_n8.json'
    codes=json.loads(codes_path.read_text())
    result=dict(status='running',graph_catalog_sha256=hashlib.sha256(json.dumps(cat,sort_keys=True).encode()).hexdigest())
    result['graph_coverage']=run_graphs(cat)
    result['real_bases']=run_real_bases()
    result['memory_histogram']=schedules(cat)
    result['binary_codes']=run_codes(codes)
    result.update(status='passed',elapsed_seconds=time.monotonic()-start,
        N8_graph_completeness='Inherited LC/permutation representative catalog; all cut ranks and explicit schedules independently checked. Independent exhaustive labelled graph orbit closure is completed through N=7.',
        boundary='Complete labelled measurement bases and specified one-private-bit key/check splits; not all quantum-output instruments or all memory-constrained QKD.')
    (ROOT/'validation/enumeration.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('Independent enumeration and sequential schedule checks PASSED',round(time.monotonic()-start,1),'s',flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graphs-through',type=int,choices=[8])
    args=parser.parse_args()
    if args.graphs_through:
        cat=json.loads((ROOT/'reproduction/controls/data/graph_representatives.json').read_text())['graphs']
        started=time.monotonic()
        result=dict(status='passed',coverage=run_graphs(cat,args.graphs_through),
            elapsed_seconds=time.monotonic()-started,
            scope='Every labelled simple graph through N=8, quotiented by local complementations and adjacent-vertex swaps; each supplied LC/permutation orbit is present exactly once.')
        (ROOT/'validation/graph_catalog_full.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    else:main()
