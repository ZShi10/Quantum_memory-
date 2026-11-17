"""Complete fixed-Clifford terminal statistics for Werner input, m<=8.

Local Clifford invariance of the IID Werner input permits graph LC-isomorph
representatives. Every nonzero dual v of the full outcome space is kept.
Public ED branches are not screened by noise or guessed keyability.
"""
from common import *
from itertools import permutations
from collections import defaultdict
from math import gcd
import json,time
import numpy as np
from streaming_clifford import compile_measurement,graph_generators
from graph_families import graph_instruments
from search_werner import collapse_werner


FRAMES=[p for p in permutations(range(4)) if set(p[i] for i in (0,3)) in ({0,3},{1,2})]


class States:
    def __init__(self,m):self.m=m;self.raw_cache={};self.lookup={};self.weights=[]
    def intern(self,weights):
        rawkey=weights.astype(np.int32).tobytes()
        if rawkey in self.raw_cache:return self.raw_cache[rawkey]
        factor=int(np.gcd.reduce(weights.reshape(-1)));assert factor>0
        w=np.asarray(weights//factor,dtype=np.int32)
        variants=[w[list(p)] for p in FRAMES];key,at=min((v.tobytes(),j) for j,v in enumerate(variants))
        if key not in self.lookup:self.lookup[key]=len(self.weights);self.weights.append(variants[at])
        result=(self.lookup[key],factor,FRAMES[at]);self.raw_cache[rawkey]=result;return result
    def block(self,counts):
        branches=[];sums=defaultdict(int)
        for syndrome,w in enumerate(counts):
            state,factor,frame=self.intern(w);sums[state]+=factor
            branches.append(dict(syndrome=syndrome,state=state,factor=factor,bell_permutation=frame))
        return tuple(sorted(sums.items())),branches


def main():
    classes=json.loads((DATA/'lc_graph_classes.json').read_text())['results'];allentries=[];graphs={};coverage={};start=time.time()
    oldentries=json.loads((OLD/'data/blocks.json').read_text())['entries']
    for m in range(1,9):
        state=States(m);seen={};generated=0
        for gi,adj in enumerate(classes[str(m)]['representatives']):
            edges=[(a,b) for a in range(m) for b in range(a+1,m) if adj[a]>>b&1]
            gid=f'LC{m}_{gi:03d}';c=compile_measurement(graph_generators(m,edges),m,real=True)
            graphs[gid]=dict(m=m,edges=edges,compilation=c)
            for v,counts in graph_instruments(m,edges):
                key,branches=state.block(counts);generated+=1
                record=dict(id=f'{gid}_v{v}',m=m,M=c['M'],graph=gid,logical_dual_vector=v,
                            grouped_states=list(key),branches=branches)
                if key not in seen or c['M']<seen[key]['M']:seen[key]=record
        missing=[]
        if m<=4:
            for e in oldentries:
                if e['m']!=m:continue
                key,_=state.block(collapse_werner(e))
                if key not in seen:missing.append(e['id'])
            assert not missing,(m,missing[:8])
        entries=list(seen.values());allentries+=entries
        np.savez_compressed(DATA/f'werner_states_m{m}.npz',weights=np.stack(state.weights))
        coverage[m]=dict(LC_classes=len(classes[str(m)]['representatives']),all_duals=generated,
                        distinct_instruments=len(entries),distinct_branch_states=len(state.weights),
                        old_small_complete_covered=m<=4)
        print('Complete Werner',m,coverage[m],'seconds',round(time.time()-start,1),flush=True)
        (DATA/'werner_complete_library.json').write_text(json.dumps(dict(coverage=coverage,entries=allentries,graphs=graphs),separators=(',',':')))
    result=dict(status='passed',coverage=coverage,entries=allentries,graphs=graphs,
                scope='All fixed Clifford one-bit-output terminal measurements on m<=8 IID Werner pairs; all ED branches and all nonzero logical duals. LC invariance and exact outcome-frame statistics are used for reduction. All such statistics have real H/CNOT realizations. Does not cover adaptive quantum branching, multiple quantum output pairs, or arbitrary-depth recurrence.')
    (DATA/'werner_complete_library.json').write_text(json.dumps(result,separators=(',',':')))
    print('Completed instruments',len(allentries),'seconds',round(time.time()-start,1),flush=True)


if __name__=='__main__':main()
