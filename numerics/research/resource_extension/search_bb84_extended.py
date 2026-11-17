"""Pilot-selectable BB84 protocols with larger graph blocks and n<=8 codes.

Five source-parameter anchors screen a candidate pool. Subsequent continuous
minimization certifies policies inside that pool; no exhaustive m>4 claim.
"""
from common import *
import json,time,argparse
from itertools import permutations
import numpy as np
from algebra import compositions
from bernstein import monomial_coefficients,evaluate
from search_bb84 import load_libraries,raw_polynomials
from search_pilot_bb84 import Oracle
from kernels import load


def make_oracle(entries,raws,kernels):
    obj=Oracle.__new__(Oracle);obj.entries=entries;obj.raws=raws;obj.kernels=kernels
    obj.starts=np.cumsum([0]+[len(r) for r in raws]);obj.ms=np.concatenate([np.full(len(r),e['m']) for r,e in zip(raws,entries)])
    obj.groups={m:[i for i,e in enumerate(entries) if e['m']==m] for m in range(1,15)}
    obj.degree_arrays={m:np.concatenate([raws[i] for i in ids]) for m,ids in obj.groups.items() if ids}
    obj.degree_positions={m:np.concatenate([np.arange(obj.starts[i],obj.starts[i+1]) for i in ids]) for m,ids in obj.groups.items() if ids}
    obj.cache={};return obj


def candidates(eb,ep):
    entries,_,_=load_libraries(eb==ep);raws=[raw_polynomials(e,eb,ep) for e in entries]
    for e in entries:e['source']='complete_short'
    graphs=json.loads((DATA/'graphs.json').read_text())['entries'];s=eb+ep;t=min(eb,ep)
    chars=np.array([[1,1],[1-2*ep]*2,[1-2*s,1-2*s+4*t],[1-2*eb]*2])
    perms=[(0,1,2,3),(0,2,1,3),(0,1,3,2)] if eb==ep else [(0,)+p for p in permutations((1,2,3))]
    for m in range(5,15):
        group=[e for e in graphs if e['m']==m];matrix=np.array([e['characters'] for e in group])
        for perm in perms:
            poly=monomial_coefficients(chars[list(perm)],np.asarray(compositions(m)))
            values=(matrix@poly)/(2**(m+1));assert values.min()>-2e-12;values=np.maximum(values,0.)
            real=perm in ((0,1,2,3),(0,3,2,1))
            for e,v in zip(group,values):
                ps=v.sum(0);active=ps>1e-18
                if not active.any() or np.max(v[:,active]/ps[active])<=.5+1e-13:continue
                entries.append(dict(id=e['id']+'_'+''.join(map(str,perm)),m=m,M=e['M'],real=real,
                                    source='extended_graph',graph_id=e['id'],axis_permutation=list(perm)))
                raws.append(v[None,:,:])
    return entries,raws


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--eb',type=float,required=True);parser.add_argument('--ep',type=float);args=parser.parse_args()
    eb=args.eb;ep=eb if args.ep is None else args.ep;start=time.time();kernels=load()
    entries,raws=candidates(eb,ep);original_count=len(entries);full=make_oracle(entries,raws,kernels);keep=set()
    for t in (0.,.25,.5,.75,1.):
        rates,_=full.point(t)
        for mode in ('real','full'):
            for M in range(1,9):
                ids=np.array([i for i,e in enumerate(entries) if e['M']<=M and (mode=='full' or e['real'])])
                keep.update(map(int,ids[np.argsort(rates[ids])[-3:]]))
        print('BB84 screen',eb,ep,t,'retained',len(keep),'seconds',round(time.time()-start,1),flush=True)
    oldpath=OLD/'data'/f'bb84_pilot_{eb:.5f}_{ep:.5f}.json'
    if oldpath.exists():
        old=json.loads(oldpath.read_text());previous={c['ED'] for r in old['results'].values() for c in r['policy']}
        keep.update(i for i,e in enumerate(entries) if e['id'] in previous)
    ids=sorted(keep);oracle=make_oracle([entries[i] for i in ids],[raws[i] for i in ids],kernels)
    del full,raws,entries
    results={}
    for mode in ('real','full'):
        for M in range(1 if mode=='full' else 2,9):
            r=oracle.certify(mode,M,atol=1e-8,max_cells=2500);results[f'{mode}_M{M}']=r
            print('BB84 extended',eb,ep,mode,M,r['lower'],r['upper'],'cells',r['cells'],'seconds',round(time.time()-start,1),flush=True)
    sourceids={c['ED'] for r in results.values() for c in r['policy']}
    output=dict(eb=eb,ep=ep,scope='Pilot-identifiable BB84: same compatible Bell family, no twirl. m<=14 graph candidates, M implementation bound <=8, all code orbits n<=7 plus selected n=8. Five-anchor screening followed by continuous minimization on the retained production pool. Not exhaustive for large m. Real M1 not plotted as state-informed achievable.',
                candidates_before_screening=original_count,screened_candidates=len(ids),
                results=results,protocol_library={e['id']:e for e in oracle.entries if e['id'] in sourceids},seconds=time.time()-start)
    (DATA/f'bb84_{eb:.5f}_{ep:.5f}.json').write_text(json.dumps(output,separators=(',',':')))


if __name__=='__main__':main()
