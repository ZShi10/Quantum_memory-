"""Complete m<=4, n<=7 single-AD-round pilot class, with exact memory.

No ED is screened on a finite set of anchor states. All compatible z are
covered by a predeclared finite decision rule. The floating optimizer's
upper enclosure is numerical; certify_complete_bb84.py verifies the entire
achievable rule independently with exact polynomials and directed rounding.
"""
from common import *
import argparse,json,time
from itertools import permutations
from functools import reduce
from math import gcd
import numpy as np
from libraries import complete_short_entries,code_kernels
from search_pilot_bb84 import Oracle
from search_bb84 import raw_polynomials
from bernstein import evaluate
from rates import best_codes
from algebra import compositions


class SharedBranchOracle(Oracle):
    def point(self,t):
        if t in self.cache:return self.cache[t]
        value=np.empty((len(self.ms),4))
        for m,array in self.degree_arrays.items():value[self.degree_positions[m]]=evaluate(array,t)
        ps=value.sum(1);p=np.divide(value,ps[:,None],out=np.full_like(value,.25),where=ps[:,None]>0)
        unique=p[self.branch_representatives];unique=np.take_along_axis(unique,self.branch_permutations,axis=1)
        best=best_codes(unique,self.kernels);codes=best['code'][self.branch_groups]
        contributions=ps/self.ms*best['rate'][self.branch_groups]
        rates=np.add.reduceat(contributions,self.starts[:-1]);result=(rates,codes);self.cache[t]=result;return result


def make_oracle(eb,ep,entries,kernels):
    o=SharedBranchOracle.__new__(SharedBranchOracle);o.entries=entries;o.kernels=kernels
    o.raws=[raw_polynomials(e,eb,ep) for e in entries]
    o.starts=np.cumsum([0]+[len(r) for r in o.raws])
    o.ms=np.concatenate([np.full(len(raw),e['m']) for raw,e in zip(o.raws,entries)])
    o.groups={m:[i for i,e in enumerate(entries) if e['m']==m] for m in range(1,5)}
    o.degree_arrays={m:np.concatenate([o.raws[i] for i in ids]) for m,ids in o.groups.items() if ids}
    o.degree_positions={m:np.concatenate([np.arange(o.starts[i],o.starts[i+1]) for i in ids]) for m,ids in o.groups.items() if ids}
    frames=[p for p in permutations(range(4)) if set(p[i] for i in (0,3)) in ({0,3},{1,2})]
    seen={};groups=[];representatives=[];perms=[]
    for e in entries:
        tensor=np.asarray(e['tensor'],dtype=np.int32)
        if eb==ep:
            types=compositions(e['m']);pairs=sorted({(v[0],v[2]) for v in types});lookup={v:i for i,v in enumerate(pairs)}
            collapsed=np.zeros(tensor.shape[:2]+(len(pairs),),dtype=np.int32)
            for j,typ in enumerate(types):collapsed[:,:,lookup[typ[0],typ[2]]]+=tensor[:,:,j]
            tensor=collapsed
        for row in tensor:
            g=reduce(gcd,map(int,row.ravel()));primitive=row//max(1,g)
            frame=min(frames,key=lambda perm:primitive[list(perm)].tobytes())
            key=(e['m'],primitive[list(frame)].tobytes())
            if key not in seen:
                seen[key]=len(seen);representatives.append(len(groups));perms.append(frame)
            groups.append(seen[key])
    o.branch_groups=np.array(groups);o.branch_representatives=np.array(representatives);o.branch_permutations=np.array(perms)
    o.cache={};return o


def main():
    p=argparse.ArgumentParser();p.add_argument('--points',nargs='*');p.add_argument('--atol',type=float,default=2e-8)
    p.add_argument('--max-cells',type=int,default=3000);a=p.parse_args()
    pairs=[tuple(map(float,s.split(','))) for s in a.points] if a.points else [(q,q) for q in [.10,.15,.18,.20,.21,.22,.23,.24,.245]]+[(.10,.20),(.15,.25),(.20,.25)]
    libraries={};kernels=code_kernels();started=time.time()
    for eb,ep in pairs:
        path=DATA/f'bb84_complete_{eb:.5f}_{ep:.5f}.json'
        saved=json.loads(path.read_text()) if path.exists() else dict(eb=eb,ep=ep,results={})
        if all(k in saved['results'] for k in ('real_M2','full_M1','full_M2')):continue
        symmetric=eb==ep
        if symmetric not in libraries:libraries[symmetric]=complete_short_entries(symmetric)
        entries=libraries[symmetric];o=make_oracle(eb,ep,entries,kernels)
        for mode,M in [('real',2),('full',1),('full',2)]:
            key=f'{mode}_M{M}'
            if key in saved['results']:continue
            result=o.certify(mode,M,atol=a.atol,max_cells=a.max_cells)
            saved['results'][key]=result
            saved.update(scope='All fixed Clifford terminal one-bit instruments m<=4 with exact streaming memory, all binary nonzero linear codes n<=7 via direct-sum domination, a single AD round, all public ED/AD branches. Pilot identifies z. No quantum candidate shortlist.',
                         entries=len(entries),kernels=len(kernels),distinct_branch_polynomials=len(o.branch_representatives),physical_points_are_discrete=True,
                         z_domain='t=z/min(eb,ep) in [0,1]',atol=a.atol,seconds=time.time()-started)
            path.write_text(json.dumps(saved,separators=(',',':')))
            print('Complete BB84',eb,ep,key,result['lower'],result['upper'],'cells',result['cells'],
                  'converged',result['converged'],'point evaluations',len(o.cache),'seconds',round(time.time()-started,1),flush=True)


if __name__=='__main__':main()
