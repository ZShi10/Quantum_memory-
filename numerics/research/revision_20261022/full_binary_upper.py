"""Outward upper for every full-intermediate binary construction and AD action.

Legal known-source test states upper-bound the fixed-policy BB84 minimax.
Every node/axis/B is either evaluated or excluded by a distillable-key upper.
An unresolved numerical gap is explicitly retained, never called optimal.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse,hashlib,json,math,time
import numpy as np
from full_clifford_model import HERE,AXES,bootstrap
from full_binary_search import read,save
from binary_evaluate import outward

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--q',required=True);ap.add_argument('--t',nargs='*')
    ap.add_argument('--absolute',type=float,default=2e-9);ap.add_argument('--relative',type=float,default=1e-4)
    ap.add_argument('--batch',type=int,default=96);args=ap.parse_args()
    qstr=str(float(args.q));q=F(qstr);root=HERE/'bb84'/qstr
    lower_source=root/'feasible.json';feasible=read(lower_source);assert feasible['complete']
    lower=np.array([0.]+[float(feasible['results'][f'M{m}']['lower']) for m in range(1,9)])
    tolerance=np.maximum(args.absolute,args.relative*lower)
    test={F(0),F(1,32),F(1,16),F(1,8),F(1,4),F(1)}
    for row in feasible['results'].values():
        c=row.get('certificate',{}).get('argmin')
        if c is not None:test.add(F(str(c)))
    if args.t is not None:test={F(t) for t in args.t}
    assert all(0<=t<=1 for t in test);test=sorted(test)
    identity=dict(Q=qstr,test_t=list(map(str,test)),lower_source_sha256=hashlib.sha256(lower_source.read_bytes()).hexdigest(),
      catalog_sha256=read(HERE/'data/full_binary_catalog.json')['sha256'],
      evaluator_sha256=hashlib.sha256((HERE/'binary_evaluate.py').read_bytes()).hexdigest(),
      driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),absolute=args.absolute,relative=args.relative)
    tag=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    job=root/'upper'/tag;job.mkdir(parents=True,exist_ok=True)
    if (job/'result.json').exists() and read(job/'result.json').get('complete'):
        print('COMPLETE',job);return
    save(job/'manifest.json',identity)
    (job/'lower_input.json').write_bytes(lower_source.read_bytes())
    core,engine,model,ops=bootstrap(job)
    import stable_intervals as iv
    from bb84_fixed import physical_kernel
    from long_code_bounds import best_bounds
    from recurrence_interval_library import step
    kernels=core.load_codes();physical={k.entry['id']:physical_kernel(k) for k in kernels}
    nodes=np.load(HERE/'data/full_binary_catalog.npz')['nodes']
    events=np.array([[(i//4,i%4,k) for i,k in enumerate(r['mapping']) if k>=0] for r in ops],np.int32)
    source=[tuple(iv.rational(v) for v in (1-2*q+q*t,q-q*t,q*t,q-q*t)) for t in test]
    sl=np.array([[v[0] for v in row] for row in source]);sh=np.array([[v[1] for v in row] for row in source])
    start=time.monotonic();pl,ph,yl,yh=outward(nodes,events,sl,sh)
    c=np.minimum(1,np.maximum(0,iv.up(2*ph.max(-1)-1)))
    er=np.minimum(yh,iv.up(yh*iv.up(c*c))).min(-1)
    upper=np.zeros(9);residual=[];history=[]
    for ai,axis in enumerate(AXES):
        memory=nodes[:,5+ai];threshold=iv.up(lower[memory]+tolerance[memory])
        pruned=er<=threshold
        for m in range(1,9):
            subset=pruned&(memory<=m)
            if subset.any():upper[m]=max(upper[m],float(er[subset].max()))
        active=np.flatnonzero(~pruned)
        print('FULL BINARY UPPER',qstr,axis,'active',len(active),'of',len(nodes),flush=True)
        perm=list(model.AXIS_PERMUTATIONS[axis])
        p=(pl[active][...,perm].copy(),ph[active][...,perm].copy())
        y=(yl[active].copy(),yh[active].copy());mm=memory[active]
        cap=er[active].copy();initial_active=len(active)
        for B in range(17):
            c=np.minimum(1,np.maximum(0,iv.up(2*p[1].max(-1)-1)))
            quantum=np.minimum(y[1],iv.up(y[1]*iv.up(c*c))).min(-1)
            cap=np.minimum(cap,quantum)
            keep=cap>iv.up(lower[mm]+tolerance[mm])
            for m in range(1,9):
                subset=(~keep)&(mm<=m)
                if subset.any():upper[m]=max(upper[m],float(cap[subset].max()))
            active=active[keep];mm=mm[keep];cap=cap[keep]
            p=tuple(v[keep] for v in p);y=tuple(v[keep] for v in y)
            count=len(active)
            for offset in range(0,count,args.batch):
                end=min(count,offset+args.batch);cache=job/f'{axis}_B{B:02d}_{offset:07d}.json'
                ids=active[offset:end];mem=mm[offset:end]
                if cache.exists():
                    saved=read(cache);assert saved['nodes']==ids.tolist();bound=np.array(saved['upper'])
                else:
                    n=end-offset;nt=len(test)
                    state=tuple(v[offset:end].reshape(-1,4) for v in p)
                    yy=y[1][offset:end]
                    goal=iv.up(lower[mem]+tolerance[mem])
                    target=np.maximum(0,iv.down(goal[:,None]/np.maximum(yy,iv.TINY))).ravel()
                    _,rates=best_bounds(state,kernels,physical,128,target)
                    values=iv.up(yy*rates.reshape(n,nt))
                    bound=np.minimum(cap[offset:end],values.min(-1))
                    save(cache,dict(nodes=ids.tolist(),memory=mem.tolist(),B=B,axis=axis,
                         upper=bound.tolist(),known_state_rate_upper=values.tolist()))
                for m in range(1,9):
                    subset=mem<=m
                    if subset.any():upper[m]=max(upper[m],float(bound[subset].max()))
                for idx,m,u in zip(ids,mem,bound):
                    if u>lower[m]+tolerance[m]:residual.append(dict(node=int(idx),axis=axis,B=B,memory=int(m),upper=float(u)))
                save(job/'status.json',dict(Q=qstr,axis=axis,B=B,processed=end,active=count,
                     upper=upper.tolist(),residual=len(residual),seconds=time.monotonic()-start))
            history.append(dict(axis=axis,B=B,active=count,axis_initial_active=initial_active))
            print('FULL BINARY ROUND',qstr,axis,B,count,'upper',upper.tolist(),flush=True)
            if not count:break
            if B<16:
                p,s=step(p,p,'Z');y=iv.scale(iv.mul(y,s),F(1,2))
        del p,y
    results={f'M{m}':dict(lower=float(lower[m]),upper=float(upper[m]),
             gap=max(0.,float(upper[m]-lower[m])),tolerance=float(tolerance[m]),
             converged=upper[m]-lower[m]<=tolerance[m]) for m in range(1,9)}
    result=dict(complete=True,Q=qstr,identity=identity,results=results,residual=residual,history=history,
        all_gaps_closed=all(v['converged'] for v in results.values()),nodes=len(nodes),
        terminal_records=3*len(nodes),seconds=time.monotonic()-start,
        scope='All full-Clifford zero-check binary constructions in the exact catalog; complete classical menu at every allowed B<=16. Legal test-state relaxation; not a pointwise optimum or an upper for all QKD.')
    save(job/'result.json',result);save(root/'upper_latest.json',dict(path=str(job/'result.json'),sha256=hashlib.sha256((job/'result.json').read_bytes()).hexdigest()))
    print(json.dumps(dict(Q=qstr,results=results,all_gaps_closed=result['all_gaps_closed'])),flush=True)
if __name__=='__main__':main()
