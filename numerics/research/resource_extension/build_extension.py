"""Selected m<=14 graph measurements, budget M<=8, and longer AD codes."""
from common import *
import json,random,time
from itertools import permutations
import numpy as np
from algebra import code_library,code_key,code_components,rref,span,compositions
from graph_families import graph_widths,families
from graph_character import enumerator


def build_graphs():
    started=time.time();entries=[];seen={}
    for m in range(5,15):
        rng=random.Random(22092026+m)
        for name,edges in families(m):
            # Keep structured graphs and six reproducible random graphs.
            if name.startswith('random_') and int(name[7:])>=6:continue
            resource=graph_widths(m,edges)
            if resource['M']>8:continue
            vectors=sorted(set([1<<j for j in range(m)]+[(1<<m)-1]+[(1<<j)-1 for j in range(2,m)]+rng.sample(range(1,1<<m),min(10,(1<<m)-1))))
            rotations=[None]
            # The additional real local-H patterns matter for BB84.
            if m<=10:
                rotations += [[(0,3,2,1) if j%2 else (0,1,2,3) for j in range(m)],
                              [(0,3,2,1) if j<m//2 else (0,1,2,3) for j in range(m)]]
            for pattern,rot in enumerate(rotations):
                for v in vectors:
                    counts=enumerator(m,edges,v,rot)
                    sig=(m,counts.tobytes())
                    record=dict(m=m,graph=name,edges=edges,logical_dual_vector=v,local_H_pattern=pattern,
                                input_rotations=rot,**resource,characters=counts.tolist(),real=True,
                                ED_acceptance='Zero relative syndrome only; all failed complete blocks cost m raw pairs.')
                    if sig not in seen or resource['M']<seen[sig]['M']:seen[sig]=record
            print('graph',m,name,'M',resource['M'],'unique',len(seen),'seconds',round(time.time()-started,1),flush=True)
    for i,record in enumerate(seen.values()):record['id']=f'XG{i:05d}';entries.append(record)
    result=dict(scope='Constructed graph families, m=5..14, M implementation upper bound <=8; sampled logical duals and real local-H patterns. Not exhaustive.',
                entries=entries,seconds=time.time()-started)
    (DATA/'graphs.json').write_text(json.dumps(result,separators=(',',':')))


def build_codes():
    started=time.time();entries,counts,orbits=code_library(7)
    for e in entries:e['id']=f'XC{entries.index(e):04d}'
    # A reproducible n=8 selection covers intermediate dimensions. Avoid
    # enumerating GL(4,2) for every systematic matrix; row/coordinate
    # duplicates only cost computation and cannot invalidate a maximum.
    rng=random.Random(9202026);seen=set();extra=[]
    for k in range(2,7):
        for j in range(8):
            rows=rref([(1<<i)|(rng.randrange(1<<(8-k))<<k) for i in range(k)],8)
            if rows in seen or len(code_components(8,rows))>1:continue
            seen.add(rows);words=span(rows)
            extra.append(dict(n=8,k=k,d=min(v.bit_count() for v in words if v),rows=list(rows),
                              endpoint=False,indecomposable=True,components=[list(range(8))],construction='seeded systematic binary code'))
    # Extended Hamming [8,4,4], constructed as first-order affine functions.
    rows=[255]+[sum(((j>>i)&1)<<j for j in range(8)) for i in range(3)]
    rows=list(rref(rows,8));extra.append(dict(n=8,k=4,d=4,rows=rows,endpoint=False,indecomposable=True,
                                           components=[list(range(8))],construction='affine evaluation on F_2^3'))
    for e in extra:e['id']=f'XC{len(entries):04d}';entries.append(e)
    result=dict(scope='All coordinate-equivalence classes through n=7, selected intermediate-dimension n=8 codes; long repetition/SPC evaluated separately.',
                labeled_counts=counts,orbit_counts=orbits,entries=entries,seconds=time.time()-started)
    (DATA/'codes.json').write_text(json.dumps(result,indent=2))
    print('Codes',len(entries),'orbit counts',orbits,'seconds',round(time.time()-started,1),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--part',choices=['graphs','codes'],required=True);args=parser.parse_args()
    build_graphs() if args.part=='graphs' else build_codes()
