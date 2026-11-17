"""Scalable ED candidates from bounded-width stabilizer measurements.

These are explicitly constructed families, not an enumeration of all m>4
Cliffords. Werner invariance makes local input Pauli-axis permutations free
for these statistics. Bell logical axes and syndrome partitions still matter.
"""
from pathlib import Path
from itertools import combinations
from collections import deque
import json,random,argparse,time
import numpy as np

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data'

def gf2_rank(rows):
    pivots={}
    for x in rows:
        while x:
            j=x.bit_length()-1
            if j in pivots:x^=pivots[j]
            else:pivots[j]=x;break
    return len(pivots)

def graph_widths(m,edges):
    adjacency=[0]*m
    for a,b in edges:adjacency[a]|=1<<b;adjacency[b]|=1<<a
    N=1<<m;allbits=N-1
    frontier=[sum(bool(adjacency[v]&(allbits^s)) for v in range(m) if s>>v&1) for s in range(N)]
    ranks=[gf2_rank([adjacency[v]&(allbits^s) for v in range(m) if s>>v&1]) for s in range(N)]
    def optimize(cost):
        dp=[m+1]*N;parents=[None]*N;dp[0]=0
        for s in range(N-1):
            value=max(dp[s],cost[s])
            for v in range(m):
                if s>>v&1:continue
                t=s|(1<<v)
                if value<dp[t]:dp[t]=value;parents[t]=(s,v)
        s=N-1;order=[]
        while s:s,v=parents[s];order.append(v)
        return dp[-1],list(reversed(order))
    memory,order=optimize([v+1 for v in frontier])
    rank,_=optimize(ranks)
    return dict(M=memory,schmidt_memory_lower=rank+1,input_order=order,linear_rank_width=rank)

def families(m,seed=19427):
    rng=random.Random(seed+m);graphs={}
    def add(name,edges):
        edges=tuple(sorted({tuple(sorted(e)) for e in edges}))
        graphs.setdefault(edges,name)
    add('path',[(i,i+1) for i in range(m-1)])
    add('cycle',[(i,(i+1)%m) for i in range(m)])
    add('star',[(0,i) for i in range(1,m)])
    add('complete',combinations(range(m),2))
    for a in range(2,m//2+1):add(f'bipartite_{a}',[(i,j) for i in range(a) for j in range(a,m)])
    if m%2==0:
        h=m//2;add('ladder',[(i,i+1) for i in range(h-1)]+[(i+h,i+h+1) for i in range(h-1)]+[(i,i+h) for i in range(h)])
    for j in range(12):
        density=(.25,.4,.6)[j%3]
        add(f'random_{j}',[e for e in combinations(range(m),2) if rng.random()<density])
    return [(name,list(edges)) for edges,name in graphs.items()]

def graph_error_arrays(m,edges):
    ids=np.arange(1<<(2*m),dtype=np.int64);mask=(1<<m)-1
    x=ids&mask;z=ids>>m
    pop=np.array([i.bit_count() for i in range(1<<m)])
    weight=pop[x|z]
    bit=z.copy()
    for a,b in edges:
        bit^=((x>>a)&1)<<b;bit^=((x>>b)&1)<<a
    return x,bit,weight,pop

def graph_instruments(m,edges,vs=None):
    x,bit,weight,pop=graph_error_arrays(m,edges)
    vs=range(1,1<<m) if vs is None else vs
    for v in vs:
        pivot=v.bit_length()-1
        logical_x=(bit>>pivot)&1
        representative=bit^(logical_x*v)
        syndrome=(representative&((1<<pivot)-1))|((representative>>(pivot+1))<<pivot)
        logical_z=pop[x&v]&1
        label=np.where(logical_z,3-logical_x,logical_x)
        flat=(syndrome*4+label)*(m+1)+weight
        counts=np.bincount(flat,minlength=(1<<(m-1))*4*(m+1)).reshape(1<<(m-1),4,m+1)
        yield int(v),counts

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--max-m',type=int,default=8)
    args=parser.parse_args();entries=[];seen={};start=time.time()
    for m in range(5,args.max_m+1):
        rng=random.Random(7209+m)
        for name,edges in families(m):
            resource=graph_widths(m,edges)
            if resource['M']>4:continue
            # Above six inputs, retain several support patterns and a
            # deterministic random sample. No completeness claim is made.
            vs=None
            if m>6:
                vs=sorted(set([1<<j for j in range(m)]+[(1<<m)-1]+[(1<<j)-1 for j in range(2,m)]+rng.sample(range(1,1<<m),24)))
            for v,counts in graph_instruments(m,edges,vs):
                signature=counts[0].tobytes()+b''.join(sorted(row.tobytes() for row in counts[1:]))
                key=(m,signature)
                record=dict(m=m,graph=name,edges=edges,logical_dual_vector=v,**resource,werner_weights=counts.tolist(),real=True)
                if key not in seen or record['M']<seen[key]['M']:seen[key]=record
            print('graph family',m,name,'M',resource['M'],'lower',resource['schmidt_memory_lower'],'unique',len(seen),'seconds',round(time.time()-start,1),flush=True)
    for i,e in enumerate(seen.values()):e['id']=f'G{i:05d}';entries.append(e)
    result=dict(scope='Constructed graph families with m=5..8. All nonzero logical dual vectors for m<=6; selected vectors above. Exact Werner weight statistics; no exhaustive Clifford claim.',entries=entries,seconds=time.time()-start)
    (DATA/'graph_family_blocks.json').write_text(json.dumps(result,separators=(',',':')))
    print('Saved',len(entries),'constructed graph instruments',flush=True)

if __name__=='__main__':main()
