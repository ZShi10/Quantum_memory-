"""Whole binary-family upper with enclosing state groups and lazy duals.

Groups are interval hulls, not assumed physical equivalences. One legal
hidden-source point per row already upper-bounds its robust fixed-policy
rate. All future B prefixes are excluded only by a quantum key upper.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from full_clifford_model import HERE,AXES,bootstrap
from full_binary_search import read,save
from binary_evaluate import outward

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--q',required=True)
    ap.add_argument('--absolute',type=float,default=2e-9);ap.add_argument('--relative',type=float,default=1e-4)
    ap.add_argument('--batch',type=int,default=128);ap.add_argument('--catalog',default='full_binary_catalog')
    ap.add_argument('--axes',default='XYZ');ap.add_argument('--lower');ap.add_argument('--label',default='full')
    args=ap.parse_args();qstr=str(float(args.q));q=F(qstr);root=HERE/'bb84'/qstr
    lower_source=Path(args.lower) if args.lower else root/'feasible.json'
    feasible=read(lower_source);assert feasible['complete']
    lower=np.array([0.]+[float(feasible['results'][f'M{m}']['lower']) for m in range(1,9)])
    tolerance=np.maximum(args.absolute,args.relative*lower)
    tests={F(0),F(1,32),F(1,8),F(1)}
    for row in feasible['results'].values():
        c=row.get('certificate',{}).get('argmin')
        if c is not None:tests.add(F(str(c)))
    tests=sorted(tests);catalog=HERE/'data'/args.catalog
    identity=dict(Q=qstr,label=args.label,axes=args.axes,test_t=list(map(str,tests)),
      lower_source_sha256=hashlib.sha256(lower_source.read_bytes()).hexdigest(),
      catalog_sha256=hashlib.sha256(catalog.with_suffix('.npz').read_bytes()).hexdigest(),
      evaluator_sha256=hashlib.sha256((HERE/'binary_evaluate.py').read_bytes()).hexdigest(),
      driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),absolute=args.absolute,relative=args.relative)
    tag=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    job=root/f'upper_grouped_{args.label}'/tag;job.mkdir(parents=True,exist_ok=True)
    if (job/'result.json').exists() and read(job/'result.json').get('complete'):
        print('COMPLETE',job);return
    save(job/'manifest.json',identity);(job/'lower_input.json').write_bytes(lower_source.read_bytes())
    core,engine,model,ops=bootstrap(job)
    import stable_intervals as iv
    from bb84_fixed import physical_kernel
    from long_code_bounds import best_bounds,classical_frame_boxes,h
    from recurrence_interval_library import step
    kernels=core.load_codes();physical={k.entry['id']:physical_kernel(k) for k in kernels}
    nodes=np.load(catalog.with_suffix('.npz'))['nodes']
    events=np.array([[(i//4,i%4,k) for i,k in enumerate(r['mapping']) if k>=0] for r in ops],np.int32)
    source=[tuple(iv.rational(v) for v in (1-2*q+q*t,q-q*t,q*t,q-q*t)) for t in tests]
    sl=np.array([[v[0] for v in row] for row in source]);sh=np.array([[v[1] for v in row] for row in source])
    start=time.monotonic();p0,p1,y0,y1=outward(nodes,events,sl,sh)

    def er_bound(p,y):
        # E_R of a two-qubit Bell law is max(0,1-h2(p_max)); monotone
        # for p_max>=1/2. The supplied largest upper endpoint is sufficient.
        mx=np.clip(p[1].max(-1),.5,1.)
        complement=iv.sub((np.ones_like(mx),np.ones_like(mx)),(mx,mx))
        entropy=iv.add(h((mx,mx)),h(complement))
        return iv.up(y[1]*np.maximum(0,iv.up(1-entropy[0])))

    er0=er_bound((p0,p1),(y0,y1));cap0=er0.min(-1)
    upper=np.zeros(9);history=[];residual=[]
    ids=[];axes=[];mems=[];caps=[];low=[];high=[];ylow=[];yhigh=[]
    for axis in args.axes:
        ai=AXES.index(axis);mm=nodes[:,5+ai];keep=cap0>iv.up(lower[mm]+tolerance[mm])
        for m in range(1,9):
            subset=(~keep)&(mm<=m)
            if subset.any():upper[m]=max(upper[m],float(cap0[subset].max()))
        index=np.flatnonzero(keep);perm=list(model.AXIS_PERMUTATIONS[axis])
        ids.append(index);axes.append(np.full(len(index),ai));mems.append(mm[index]);caps.append(cap0[index])
        low.append(p0[index][...,perm]);high.append(p1[index][...,perm]);ylow.append(y0[index]);yhigh.append(y1[index])
    ids=np.concatenate(ids);axes=np.concatenate(axes);memory=np.concatenate(mems);cap=np.concatenate(caps)
    p=(np.concatenate(low),np.concatenate(high));y=(np.concatenate(ylow),np.concatenate(yhigh))
    del p0,p1,y0,y1,low,high,ylow,yhigh
    print('GROUPED UPPER START',qstr,args.label,'active',len(ids),flush=True)
    for B in range(17):
        quantum=er_bound(p,y);cap=np.minimum(cap,quantum.min(-1))
        keep=cap>iv.up(lower[memory]+tolerance[memory])
        for m in range(1,9):
            subset=(~keep)&(memory<=m)
            if subset.any():upper[m]=max(upper[m],float(cap[subset].max()))
        ids=ids[keep];axes=axes[keep];memory=memory[keep];cap=cap[keep]
        p=tuple(v[keep] for v in p);y=tuple(v[keep] for v in y);quantum=quantum[keep]
        if not len(ids):
            history.append(dict(B=B,active=0,groups=0));break
        anchor=quantum.argmin(-1);rows=np.arange(len(ids));scale=y[1][rows,anchor]
        selected=tuple(v[rows,anchor] for v in p)
        boxes,inverse=classical_frame_boxes(selected,quantum=1e-13)
        target=np.maximum(0,iv.down(iv.up(lower[memory]+tolerance[memory])/np.maximum(scale,iv.TINY)))
        group_target=np.full(len(boxes[0]),np.inf);np.minimum.at(group_target,inverse,target)
        bound_by_group=np.zeros(len(boxes[0]));group_path=job/f'B{B:02d}_groups.npz'
        if not group_path.exists():
            np.savez_compressed(group_path,lower=boxes[0],upper=boxes[1],inverse=inverse,
                target=group_target,node=ids,axis=axes,memory=memory,anchor=anchor,scale_upper=scale,quantum_upper=cap)
        print('GROUPED ROUND',qstr,args.label,B,'rows',len(ids),'groups',len(boxes[0]),flush=True)
        for offset in range(0,len(boxes[0]),args.batch):
            end=min(len(boxes[0]),offset+args.batch);cache=job/f'B{B:02d}_{offset:07d}.json'
            if cache.exists():
                saved=read(cache);assert saved['end']==end;bound_by_group[offset:end]=saved['upper']
            else:
                _,bb=best_bounds(tuple(v[offset:end] for v in boxes),kernels,physical,128,group_target[offset:end])
                bound_by_group[offset:end]=bb
                save(cache,dict(offset=offset,end=end,upper=bb.tolist()))
            save(job/'status.json',dict(Q=qstr,label=args.label,B=B,processed=end,groups=len(boxes[0]),
                active=len(ids),seconds=time.monotonic()-start))
        bound=np.minimum(cap,iv.up(scale*bound_by_group[inverse]))
        for m in range(1,9):
            subset=memory<=m
            if subset.any():upper[m]=max(upper[m],float(bound[subset].max()))
        unresolved=np.flatnonzero(bound>iv.up(lower[memory]+tolerance[memory]))
        residual.extend(dict(node=int(ids[i]),axis=AXES[axes[i]],B=B,memory=int(memory[i]),
            upper=float(bound[i])) for i in unresolved)
        np.savez_compressed(job/f'B{B:02d}_bounds.npz',node=ids,axis=axes,memory=memory,upper=bound)
        history.append(dict(B=B,active=len(ids),groups=len(boxes[0]),unresolved=len(unresolved),upper=upper.tolist()))
        print('GROUPED ROUND DONE',qstr,args.label,B,'upper',upper.tolist(),'unresolved',len(unresolved),flush=True)
        if B<16:
            p,s=step(p,p,'Z');y=iv.scale(iv.mul(y,s),F(1,2))
    results={f'M{m}':dict(lower=float(lower[m]),upper=float(upper[m]),gap=max(0.,float(upper[m]-lower[m])),
        tolerance=float(tolerance[m]),converged=upper[m]-lower[m]<=tolerance[m]) for m in range(1,9)}
    result=dict(complete=True,Q=qstr,identity=identity,results=results,residual=residual,history=history,
        all_gaps_closed=all(v['converged'] for v in results.values()),nodes=len(nodes),
        terminal_records=len(args.axes)*len(nodes),seconds=time.monotonic()-start,
        scope='Every declared binary node, permitted final axis and B<=16; complete classical menu. Enclosing state groups and legal known-source upper relaxations. No unrestricted memory-only optimum claim.')
    save(job/'result.json',result);save(root/f'upper_grouped_{args.label}_latest.json',dict(path=str(job/'result.json'),sha256=hashlib.sha256((job/'result.json').read_bytes()).hexdigest()))
    print(json.dumps(dict(Q=qstr,label=args.label,results=results,all_gaps_closed=result['all_gaps_closed'])),flush=True)
if __name__=='__main__':main()
