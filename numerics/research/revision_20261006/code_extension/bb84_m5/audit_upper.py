"""Outward upper relaxation of the complete m<=5 block + n<=8 AD union.

At each test correlation, every public flag may choose its own B prefix and
AD action. This is an upper relaxation of any fixed, hidden-state-independent
policy, never an achievable policy. Existing continuous-correlation lower
witnesses prune candidates only; their source files are hashed in the job.
"""
from pathlib import Path
import os,sys,json,time,hashlib,argparse,importlib.util
sys.dont_write_bytecode=True
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3]
LEGACY=PROJECT/'research/revision_20260922'


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def save(path,obj):
    tmp=path.with_suffix('.writing.json');tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8');tmp.replace(path)


def lower_sources(b,p,permission,explicit=()):
    name=f'bb84_joint_extension_{permission}_{b:.5f}_{p:.5f}_n128.json'
    paths=[LEGACY/'data'/name]
    paths+=list((PROJECT/'research/revision_20261006/bb84_search/runs').glob(f'*/data/{name}'))
    paths+=[Path(v) for v in explicit]
    lower=[0.]*9;ts={'0','1'};sources={}
    for path in sorted(set(paths)):
        if not path.exists():continue
        source=json.loads(path.read_text())
        if abs(float(source.get('b',b))-b)>1e-13 or abs(float(source.get('p',p))-p)>1e-13:continue
        accepted=False
        for key,row in source.get('results',{}).items():
            if not key.startswith('M') or not isinstance(row,dict):continue
            budget=int(key[1:]);value=float(row.get('lower',0.));cert=row.get('certificate')
            if value<=0 or not cert or float(cert.get('lower',-1))<value-1e-18:continue
            assert 1<=budget<=8 and row.get('ED',{}).get('memory_upper_bound',budget)<=budget
            for k in range(budget,9):lower[k]=max(lower[k],value)
            if budget<=3 and 0<=float(cert.get('argmin',-1))<=1:ts.add(str(cert['argmin']))
            accepted=True
        if accepted:sources[str(path.relative_to(PROJECT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return lower,sorted(ts,key=float),sources


def run(args,b):
    p=float(args.p) if args.p is not None else b
    lower,ts,provenance=lower_sources(b,p,args.permissions,args.lower_file)
    # The two endpoints alone are always legal test states. Additional
    # witness minima sharpen the relaxation without changing its validity.
    ts=sorted(set(ts+args.test_t),key=float)
    inputs=[Path(__file__),HERE/'api.py',HERE/'build_library.py',HERE.parent/'api.py',
            HERE.parent/'data/codes_complete_n8.json',
            HERE/f'data/instruments_{args.permissions}_m5.json',
            HERE/f'data/tensors_{args.permissions}_m5.npz',
            LEGACY/'long_code_bounds.py',LEGACY/'interval_modes.py',
            LEGACY/'stable_intervals.py']
    hashes={str(v.relative_to(PROJECT)):hashlib.sha256(v.read_bytes()).hexdigest() for v in inputs}
    identity=dict(b=b,p=p,permissions=args.permissions,B_max=args.K,long_max=args.nmax,
                  general_code_max=8,fixed_block_max=5,test_parameters=ts,
                  tolerance=args.tolerance,global_frame_box_quantum=args.quantum,
                  lower=lower,lower_sources=provenance,inputs=hashes)
    tag=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    job=HERE/'upper_runs'/tag;job.mkdir(parents=True,exist_ok=True)
    save(job/'manifest.json',identity)
    if (job/'result.json').exists():
        result=json.loads((job/'result.json').read_text())
        if result.get('complete'):print('ALREADY COMPLETE',job,flush=True);return result
    sys.path.insert(0,str(LEGACY));import core
    core.ROOT=job;core.DATA=job/'data';core.DATA.mkdir(exist_ok=True)
    codes=load_module('bb5_complete_codes',HERE.parent/'api.py')
    blocks=load_module('bb5_complete_blocks',HERE/'api.py')
    import numpy as np
    from fractions import Fraction as F
    from algebra import compositions
    from certify_bb84_recursive import source,step
    from long_code_bounds import best_bounds,classical_frame_boxes
    from bb84_fixed import physical_kernel
    import stable_intervals as iv
    start=time.time();entries=blocks.load_entries(args.permissions,b==p)
    lower=np.asarray(lower);M=np.array([e['M'] for e in entries]);m=np.array([e['m'] for e in entries])
    kernels=codes.load_codes();physical={k.entry['id']:physical_kernel(k) for k in kernels}
    initial=source(b,p,[(F(t),F(t)) for t in ts]);powers={};raw=[];owner=[]
    for i,e in enumerate(entries):
        if e['m'] not in powers:powers[e['m']]=iv.monomial_bounds(initial,np.asarray(compositions(e['m'])))
        tensor=np.asarray(e['tensor']);value=iv.positive_dot(powers[e['m']],tensor.reshape(-1,tensor.shape[-1]).T)
        raw.append(tuple(v.reshape(len(ts),-1,4).transpose(1,0,2) for v in value));owner.extend([i]*len(tensor))
    raw=tuple(np.concatenate([r[k] for r in raw]) for k in (0,1));owner=np.asarray(owner)
    mass=iv.positive_sum(raw);weight=iv.div(mass,iv.point(m[owner,None]));good=mass[0]>0
    denominator=(np.where(good,mass[0],1.)[...,None],np.where(good,mass[1],1.)[...,None])
    state=iv.div(raw,denominator)
    state=tuple(np.where(good[...,None],v,0. if i==0 else 1.) for i,v in enumerate(state))
    shape=mass[0].shape;state=tuple(v.reshape(-1,4) for v in state)
    factor=(np.ones(len(state[0])),np.ones(len(state[0])));best=np.zeros(len(state[0]))
    active=np.ones(len(entries),bool);upper=np.full(len(entries),np.inf)
    flagcount=int(max(np.bincount(owner)));rounds=[];begin=0
    checkpoint=job/'checkpoint.npz'
    if checkpoint.exists():
        with np.load(checkpoint) as data:
            begin=int(data['next_B']);state=(data['state_lo'],data['state_hi']);factor=(data['factor_lo'],data['factor_hi'])
            best=data['best'];active=data['active'];upper=data['upper']
        rounds=json.loads((job/'rounds.json').read_text())
    def aggregate(values):
        terms=iv.up(weight[1]*values.reshape(shape));total=np.zeros((len(entries),len(ts)))
        np.add.at(total,owner,terms)
        return iv.up(total/(1-iv.gamma(2*flagcount))).min(-1)
    for B in range(begin,args.K+1):
        concurrence=np.minimum(1,np.maximum(0,iv.up(2*state[1].max(-1)-1)))
        er=iv.up(factor[1]*iv.up(concurrence*concurrence));potential=aggregate(np.maximum(best,er))
        upper[active]=potential[active];active &= potential>lower[M]+args.tolerance
        ids=np.flatnonzero(np.repeat(active[owner],len(ts)))
        ids=ids[(factor[1][ids]>0)&(er[ids]>best[ids])&(weight[1].reshape(-1)[ids]>0)]
        grouped=0;benchmark={}
        if len(ids):
            importance=iv.up(weight[1].reshape(-1)[ids]*flagcount*factor[1][ids])
            localtol=iv.down(args.tolerance/np.maximum(importance,1e-300));selected_owner=owner[ids//len(ts)]
            allocated=iv.down(lower[M[selected_owner]]/np.maximum(importance,1e-300))
            target=np.maximum(best[ids]/np.maximum(factor[1][ids],iv.TINY),np.maximum(localtol,allocated))
            boxes,inverse=classical_frame_boxes((state[0][ids],state[1][ids]),quantum=args.quantum)
            grouped=len(boxes[0]);goals=np.full(grouped,np.inf);np.minimum.at(goals,inverse,target)
            bounds=np.zeros(grouped)
            if B==0:
                for quantum in (1e-14,1e-12):
                    other,_=classical_frame_boxes((state[0][ids],state[1][ids]),quantum=quantum)
                    benchmark[str(quantum)]=len(other[0])
                print('Global containing boxes',b,'states',len(ids),'groups',grouped,'comparison',benchmark,flush=True)
            for first in range(0,grouped,96):
                stop=min(first+96,grouped)
                _,bound=best_bounds((boxes[0][first:stop],boxes[1][first:stop]),kernels,physical,args.nmax,goals[first:stop],compress=False)
                bounds[first:stop]=bound
                save(job/'progress.json',dict(b=b,p=p,B=B,grouped_states_done=stop,grouped_states_total=grouped,
                                              ungrouped_states=len(ids),active=int(active.sum()),seconds=time.time()-start))
            best[ids]=np.maximum(best[ids],iv.up(factor[1][ids]*bounds[inverse]))
        assert np.isfinite(best).all()
        if B==args.K:upper[active]=aggregate(best)[active]
        rounds.append(dict(B=B,active_instruments=int(active.sum()),conditional_states=len(ids),
                           containing_boxes=grouped,compression_benchmark=benchmark))
        complete=B==args.K or not active.any()
        result=dict(**identity,job_fingerprint=tag,complete=complete,rounds=rounds,
                    results={f'M{k}':dict(upper=float(upper[M<=k].max()),lower_witness=float(lower[k]),
                        dominated_within_tolerance=bool(upper[M<=k].max()<=lower[k]+args.tolerance)) for k in range(1,9)},
                    entries=[dict(ED=e['id'],M=e['M'],m=e['m'],upper=float(u)) for e,u in zip(entries,upper)],
                    seconds=time.time()-start,scope='Whole m<=5 flagged block class and n<=8 general-code plus long endpoint menu; per-state per-flag action/B choices upper-relax one fixed compound policy. Complete source/model scope in manifest. No block lower policy or global-memory optimum is inferred.')
        save(job/'result.json',result);save(job/'rounds.json',rounds)
        print('BLOCK UPPER',b,B,result['results'],'active',int(active.sum()),'seconds',round(time.time()-start,1),flush=True)
        if complete:return result
        state,s=step(state,state,'Z');factor=iv.scale(iv.mul(factor,s),F(1,2))
        with checkpoint.open('wb') as f:np.savez_compressed(f,next_B=B+1,state_lo=state[0],state_hi=state[1],factor_lo=factor[0],factor_hi=factor[1],best=best,active=active,upper=upper)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--q',type=float,nargs='+',required=True);parser.add_argument('--p',type=float)
    parser.add_argument('--permissions',choices=['real','full'],default='real');parser.add_argument('--nmax',type=int,default=128)
    parser.add_argument('--K',type=int,default=16);parser.add_argument('--tolerance',type=float,default=1e-9)
    parser.add_argument('--quantum',type=float,default=1e-12)
    parser.add_argument('--lower-file',action='append',default=[]);parser.add_argument('--test-t',action='append',default=[])
    args=parser.parse_args()
    for q in args.q:run(args,q)
