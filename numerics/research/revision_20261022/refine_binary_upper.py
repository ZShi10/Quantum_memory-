"""Sharpen every unresolved binary row by outward legal-source tests/duals."""
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from full_clifford_model import HERE,AXES,bootstrap
from full_binary_search import read,save
from binary_evaluate import outward

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);ap.add_argument('--lower')
    ap.add_argument('--solve',action='store_true');ap.add_argument('--batch',type=int,default=128)
    ap.add_argument('--dual-only',action='store_true');ap.add_argument('--cache-source')
    ap.add_argument('--seed',action='append',default=[])
    args=ap.parse_args();basepath=Path(args.base);pointer=read(basepath)
    if 'path' in pointer and 'results'not in pointer:basepath=Path(pointer['path']);base=read(basepath)
    else:base=pointer
    assert base['complete'];q=float(base['Q']);ident=base['identity'];label=ident.get('label','full')
    catalog_name='real_binary_catalog' if label.startswith('real') else 'full_binary_catalog'
    lowerpath=Path(args.lower) if args.lower else basepath.parent/'lower_input.json'
    feasible=read(lowerpath);assert feasible['complete'];low={m:float(feasible['results'][m]['lower']) for m in feasible['results']}
    spec=dict(base=str(basepath),base_sha256=hashlib.sha256(basepath.read_bytes()).hexdigest(),
        lower_sha256=hashlib.sha256(lowerpath.read_bytes()).hexdigest(),solve=args.solve,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),dual_only=args.dual_only,
        seed_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args.seed})
    tag=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:20]
    job=HERE/'bb84'/str(q)/f'refined_{label}'/tag;job.mkdir(parents=True,exist_ok=True)
    if (job/'result.json').exists() and read(job/'result.json').get('complete'):
        print('ALREADY REFINED',job);return
    save(job/'spec.json',spec)
    if args.cache_source:
        import shutil
        cached=Path(args.cache_source);oldspec=read(cached/'spec.json')
        assert oldspec['base_sha256']==spec['base_sha256'] and oldspec['lower_sha256']==spec['lower_sha256']
        # The legal test-source set and known-source stage are unchanged.
        # Only this file's subsequent dual/lower-proposal stage is extended.
        for p in cached.glob('B??_*.json'):
            target=job/p.name
            if not target.exists():shutil.copyfile(p,target)
        save(job/'known_source_cache_provenance.json',dict(source=str(cached),base_sha256=spec['base_sha256'],
            lower_sha256=spec['lower_sha256'],files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in cached.glob('B??_*.json')}))
    core,engine,model,ops=bootstrap(job)
    import stable_intervals as iv
    from long_code_bounds import best_bounds,classical_frame_boxes
    from bb84_fixed import physical_kernel
    from recurrence_interval_library import step
    from compound_menu_bounds import dual_upper
    nodes=np.load(HERE/'data'/f'{catalog_name}.npz')['nodes']
    @lru_cache(None)
    def tree(idx):
        if idx==0:return model.RAW
        op,l,r=map(int,nodes[idx,:3]);return model.Node(ops[op]['name'],tree(l),tree(r))
    def record(idx,axis):return model.structural_record(tree(int(idx)),axis,'enlarged_axes')
    tests={F(t) for t in ('0','1/256','1/64','1/32','1/16','3/32','1/8','3/16','1/4','5/16','3/8','1/2','3/4','1')}
    for r in feasible['results'].values():
        t=r.get('certificate',{}).get('argmin')
        if t is not None:tests.add(F(str(t)))
    tests=sorted(tests);qq=F(str(q));source=[tuple(iv.rational(v) for v in (1-2*qq+qq*t,qq-qq*t,qq*t,qq-qq*t)) for t in tests]
    sl=np.array([[v[0] for v in row] for row in source]);sh=np.array([[v[1] for v in row] for row in source])
    events=np.array([[(i//4,i%4,k) for i,k in enumerate(r['mapping']) if k>=0] for r in ops],np.int32)
    residual=base['residual'];wanted={int(r['node']) for r in residual}|{0};stack=list(wanted)
    while stack:
        i=stack.pop()
        if i:
            for c in nodes[i,1:3]:
                if int(c)not in wanted:wanted.add(int(c));stack.append(int(c))
    ids=sorted(wanted);index={idx:i for i,idx in enumerate(ids)};small=nodes[ids].copy()
    for i in range(1,len(small)):
        for j in (1,2):small[i,j]=index[int(small[i,j])]
    p0,p1,y0,y1=outward(small,events,sl,sh)
    kernels=core.load_codes();physical={k.entry['id']:physical_kernel(k) for k in kernels}
    start=time.monotonic();rows={};allfeasible=dict(feasible['results']);dual_seeds=[]
    for r in allfeasible.values():
        if r.get('sampled_dual'):dual_seeds.append(r)
    for B in sorted({r['B'] for r in residual}):
        rr=[r for r in residual if r['B']==B];selected=np.array([index[r['node']] for r in rr])
        pp=(p0[selected].copy(),p1[selected].copy());yy=(y0[selected].copy(),y1[selected].copy())
        for i,r in enumerate(rr):
            perm=list(model.AXIS_PERMUTATIONS[r['axis']])
            for v in pp:v[i]=v[i][:,perm]
        for _ in range(B):pp,s=step(pp,pp,'Z');yy=iv.scale(iv.mul(yy,s),F(1,2))
        boxes,inverse=classical_frame_boxes(tuple(v.reshape(-1,4) for v in pp),quantum=1e-13)
        goals=np.array([low[f'M{r["memory"]}']+base['results'][f'M{r["memory"]}']['tolerance'] for r in rr])
        target=np.maximum(0,iv.down(iv.up(goals[:,None])/np.maximum(yy[1],iv.TINY))).ravel()
        gt=np.full(len(boxes[0]),np.inf);np.minimum.at(gt,inverse,target)
        bounds=np.zeros(len(boxes[0]));print('REFINE GROUPS',q,label,B,len(rr),len(bounds),flush=True)
        for offset in range(0,len(bounds),args.batch):
            end=min(len(bounds),offset+args.batch);path=job/f'B{B:02d}_{offset:07d}.json'
            if path.exists():bb=read(path)['upper']
            else:
                _,bb=best_bounds(tuple(v[offset:end] for v in boxes),kernels,physical,128,gt[offset:end])
                save(path,dict(offset=offset,end=end,upper=bb.tolist()))
            bounds[offset:end]=bb
            save(job/'status.json',dict(stage='known_source_tests',B=B,processed=end,total=len(bounds),seconds=time.monotonic()-start))
        per=iv.up(yy[1]*bounds[inverse].reshape(len(rr),len(tests)))
        for i,r in enumerate(rr):
            key=f'{r["node"]}_{r["axis"]}_B{B}';u=min(r['upper'],float(per[i].min()))
            rows[key]=dict(r,upper=u,base_upper=r['upper'],test_t=list(map(str,tests)),test_rate_upper=per[i].tolist(),duals=[])
    # Additional independently certified feasible policies do not change the
    # preceding known-source tests or any stored upper. Real-node seeds are
    # admitted to the full-node family, never in the reverse direction.
    for seed in args.seed:
        document=read(Path(seed));assert float(document['Q'])==q and document['complete']
        assert label=='full' or document.get('label')==label
        for name,candidate in document['results'].items():
            assert candidate['ED']['memory_upper_bound']<=int(name[1:])
            assert candidate['lower']<=max(0.,candidate['certificate']['lower'])+1e-18
            if candidate['lower']>low[name]:low[name]=candidate['lower'];allfeasible[name]=candidate
    # A feasible dual measure can be proposed by a floating LP or copied from
    # a different row; its coefficients must be evaluated afresh for this row.
    ordered=sorted(rows,key=lambda k:rows[k]['upper']-low[f'M{rows[k]["memory"]}'],reverse=True)
    for key in ordered:
        r=rows[key];m=r['memory'];goal=low[f'M{m}']+base['results'][f'M{m}']['tolerance']
        if r['upper']<=goal:continue
        ed=record(r['node'],r['axis']);path=job/f'row_{key}.json'
        if path.exists():
            saved=read(path);r.update(saved['bound']);candidate=saved.get('candidate')
        elif args.solve:
            # A sampled dual that is already below the certified global
            # incumbent need not trigger another expensive feasible-policy
            # continuum minimization. It is still independently re-evaluated
            # over EVERY classical coefficient below; no floating exclusion
            # is accepted as a rigorous bound.
            solve_tolerance=max(2e-9,base['results'][f'M{m}']['tolerance']/4)
            if args.dual_only:
                from dual_only_lp import solve as solve_dual
                candidate=solve_dual(engine,q,q,ed,r['B'],nmax=128,tolerance=solve_tolerance,
                    max_iterations=32,incumbent=low[f'M{m}'])
            else:
                candidate=engine.solve(q,q,ed,r['B'],nmax=128,tolerance=solve_tolerance,
                    max_iterations=32,incumbent=low[f'M{m}'])
            candidate.update(ED=ed,B=r['B'])
            if candidate.get('sampled_dual'):
                cert=dual_upper(q,q,candidate,128);r['duals'].append(cert);r['upper']=min(r['upper'],cert['upper'])
            save(path,dict(bound=r,candidate=candidate))
        else:candidate=None
        if candidate and candidate.get('certificate',{}).get('converged'):
            for budget in range(m,9):
                name=f'M{budget}'
                if candidate['lower']>low[name]:low[name]=candidate['lower'];allfeasible[name]=candidate
        print('REFINED ROW',q,label,key,r['upper'],low[f'M{m}'],flush=True)
        save(job/'status.json',dict(stage='duals',key=key,seconds=time.monotonic()-start))
    results={}
    for m in range(1,9):
        name=f'M{m}';basecap=float(iv.up(base['results'][name]['lower']+base['results'][name]['tolerance']))
        upper=min(base['results'][name]['upper'],max([basecap]+[r['upper'] for r in rows.values() if r['memory']<=m]))
        assert upper+1e-13>=low[name],(name,upper,low[name])
        tolerance=max(2e-9,1e-4*low[name])
        endpoint=float(iv.up(low[name]+tolerance))
        results[name]=dict(lower=low[name],upper=upper,gap=max(0.,upper-low[name]),tolerance=tolerance,
            upper_endpoint_target=endpoint,converged=bool(upper<=endpoint))
    result=dict(complete=True,Q=q,identity=spec,results=results,refinements=rows,seconds=time.monotonic()-start,
        all_gaps_closed=all(r['converged'] for r in results.values()),
        scope='All base candidates retained. Every new upper is an outward known-source or rational dual full-menu bound. No floating result used as a certificate.')
    save(job/'feasible.json',dict(complete=True,Q=q,results=allfeasible,source_spec=spec))
    save(job/'result.json',result);save(HERE/'bb84'/str(q)/f'refined_{label}_latest.json',dict(path=str(job/'result.json'),sha256=hashlib.sha256((job/'result.json').read_bytes()).hexdigest()))
    print('REFINE DONE',q,label,results,flush=True)
if __name__=='__main__':main()
