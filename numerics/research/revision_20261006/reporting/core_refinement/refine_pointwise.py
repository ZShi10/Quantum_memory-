"""Add outward test-state bounds for unresolved complete BB84 instruments.

Preserve every original instrument and its valid upper. The new whole-class
bound is the budget maximum after taking each instrument's minimum of its
old and newly computed upper bounds. No lower or optimum is fabricated.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, sys, time
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
REV=HERE.parents[1]
PROJECT=REV.parents[1]
LEGACY=PROJECT/'research/revision_20260922'


def save(path, obj):
    temp=path.with_suffix('.writing.json')
    temp.write_text(json.dumps(obj,indent=2),encoding='utf-8')
    temp.replace(path)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--absolute',type=float,default=1e-11)
    parser.add_argument('--relative',type=float,default=1e-4)
    parser.add_argument('--local-tolerance',type=float,default=1e-13)
    parser.add_argument('--test-t',nargs='+',default=['.125','.15','.175','.19','.2','.21','.225','.24','.25','.275','.3','.375','.5'])
    args=parser.parse_args()
    base=json.loads(args.base.read_text())
    assert base['complete'] and base['permissions']=='real'
    assert base['fixed_block_max']==5 and base['general_code_max']==8 and base['long_max']==128 and base['B_max']==16
    for name,expected in base['inputs'].items():assert sha(PROJECT/name)==expected,name
    bounds={row['ED']:dict(row) for row in base['entries']}
    def goal(row):
        lo=base['lower'][row['M']]
        return lo+max(args.absolute,args.relative*lo)
    selected=[row['ED'] for row in base['entries'] if row['upper']>goal(row)]
    sources={str(Path(__file__).relative_to(PROJECT)):sha(Path(__file__)),str(args.base.resolve().relative_to(PROJECT)):sha(args.base)}
    # The original result already commits to its engine/provider dependencies.
    sources.update(base['inputs'])
    for helper in ('certify_bb84_recursive.py','core.py','compound_terminals.py','interval_jets.py'):
        path=LEGACY/helper
        if path.exists():sources[str(path.relative_to(PROJECT))]=sha(path)
    identity=dict(base_fingerprint=base['job_fingerprint'],b=base['b'],p=base['p'],test_t=args.test_t,
                  absolute=args.absolute,relative=args.relative,local_tolerance=args.local_tolerance,
                  selected=selected,inputs=sources)
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    job=HERE/'runs'/fingerprint;job.mkdir(parents=True,exist_ok=True)
    save(job/'manifest.json',identity)
    if (job/'result.json').exists():
        print((job/'result.json').read_text(),flush=True)
        return
    sys.path.insert(0,str(LEGACY))
    import core
    core.ROOT=job;core.DATA=job/'data';core.DATA.mkdir(exist_ok=True)
    codes=load('refinement_codes',REV/'code_extension/api.py')
    blocks=load('refinement_blocks',REV/'code_extension/bb84_m5/api.py')
    import numpy as np
    from fractions import Fraction as F
    from algebra import compositions
    from certify_bb84_recursive import source,step
    from long_code_bounds import best_bounds,classical_frame_boxes
    from bb84_fixed import physical_kernel
    import stable_intervals as iv
    entries={e['id']:e for e in blocks.load_entries('real',base['b']==base['p'])}
    assert set(entries)==set(bounds)
    kernels=codes.load_codes();physical={k.entry['id']:physical_kernel(k) for k in kernels}
    test=[F(t) for t in args.test_t]
    assert all(0<=t<=1 for t in test)
    initial=source(base['b'],base['p'],[(t,t) for t in test])
    start=time.time();refinements={}
    for identifier in selected:
        cached=job/(identifier+'.json')
        if cached.exists():
            record=json.loads(cached.read_text());refinements[identifier]=record
            bounds[identifier]['upper']=min(bounds[identifier]['upper'],record['upper'])
            continue
        entry=entries[identifier]
        power=iv.monomial_bounds(initial,np.asarray(compositions(entry['m'])))
        tensor=np.asarray(entry['tensor'])
        flat=iv.positive_dot(power,tensor.reshape(-1,tensor.shape[-1]).T)
        raw=tuple(v.reshape(len(test),-1,4).transpose(1,0,2) for v in flat)
        mass=iv.positive_sum(raw)
        weight=iv.div(mass,iv.point(entry['m']))
        good=mass[0]>0
        denominator=(np.where(good,mass[0],1.)[...,None],np.where(good,mass[1],1.)[...,None])
        state=iv.div(raw,denominator)
        state=tuple(np.where(good[...,None],v,0. if i==0 else 1.).reshape(-1,4) for i,v in enumerate(state))
        factor=(np.ones(len(state[0])),np.ones(len(state[0])))
        best=np.zeros(len(state[0]));flagcount=len(tensor);rounds=[]
        def aggregate(values):
            terms=iv.up(weight[1]*values.reshape(good.shape))
            # Sum along flags with the same proven gamma envelope as the base audit.
            return iv.up(terms.sum(0)/(1-iv.gamma(2*flagcount)))
        for B in range(17):
            concurrence=np.minimum(1,np.maximum(0,iv.up(2*state[1].max(-1)-1)))
            er=iv.up(factor[1]*iv.up(concurrence*concurrence))
            potential=aggregate(np.maximum(best,er))
            if float(potential.min())<=goal(bounds[identifier]):
                upper=float(potential.min())
                rounds.append(dict(B=B,upper=upper,stage='future_tail_dominated'))
                break
            ids=np.flatnonzero((factor[1]>0)&(er>best)&(weight[1].reshape(-1)>0))
            groups=0
            if len(ids):
                importance=iv.up(weight[1].reshape(-1)[ids]*flagcount*factor[1][ids])
                localtol=iv.down(args.local_tolerance/np.maximum(importance,1e-300))
                # No incumbent threshold here: sharpen only the two residual objects.
                target=np.maximum(best[ids]/np.maximum(factor[1][ids],iv.TINY),localtol)
                boxes,inverse=classical_frame_boxes((state[0][ids],state[1][ids]),quantum=1e-15)
                groups=len(boxes[0]);goals=np.full(groups,np.inf);np.minimum.at(goals,inverse,target)
                upper_box=np.zeros(groups)
                for first in range(0,groups,64):
                    stop=min(first+64,groups)
                    _,v=best_bounds((boxes[0][first:stop],boxes[1][first:stop]),kernels,physical,128,goals[first:stop],compress=False)
                    upper_box[first:stop]=v
                    save(job/'progress.json',dict(ED=identifier,B=B,groups_done=stop,groups_total=groups,seconds=time.time()-start))
                best[ids]=np.maximum(best[ids],iv.up(factor[1][ids]*upper_box[inverse]))
            actual=aggregate(best)
            upper=float(actual.min())
            rounds.append(dict(B=B,upper_so_far=upper,states=len(ids),groups=groups,test_values=list(map(float,actual))))
            print('REFINE',identifier,'B',B,'known-action upper-so-far',upper,'groups',groups,flush=True)
            if B==16:break
            state,s=step(state,state,'Z');factor=iv.scale(iv.mul(factor,s),F(1,2))
        record=dict(ED=identifier,m=entry['m'],M=entry['M'],upper=upper,rounds=rounds,
                    test_t=args.test_t,seconds=time.time()-start)
        save(cached,record);refinements[identifier]=record
        bounds[identifier]['upper']=min(bounds[identifier]['upper'],upper)
    results={}
    for M in range(1,9):
        upper=max(row['upper'] for row in bounds.values() if row['M']<=M)
        lower=base['lower'][M]
        results['M'+str(M)]=dict(upper=upper,lower_witness=lower,
             gap=upper-lower,dominated_within_tolerance=upper<=lower+max(args.absolute,args.relative*lower))
    result=dict(**identity,job_fingerprint=fingerprint,complete=True,
                union_gap_closed=all(row['dominated_within_tolerance'] for row in results.values()),
                permissions='real',fixed_block_max=5,general_code_max=8,long_max=128,B_max=16,
                results=results,entries=list(bounds.values()),refinements=refinements,
                scope='Outward point-test refinement and minimum with every original complete-block entry upper; complete 265 instrument class retained. No new achievable lower inferred.',
                seconds=time.time()-start)
    save(job/'result.json',result)
    print('RESULT',str(job/'result.json'),json.dumps(results),flush=True)


if __name__=='__main__':main()
