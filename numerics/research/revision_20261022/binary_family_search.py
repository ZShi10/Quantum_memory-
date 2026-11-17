"""Matched real/full binary-family discovery and continuum feasible policies."""
from functools import lru_cache
import argparse,hashlib,json,time
import numpy as np
from full_clifford_model import HERE,AXES,bootstrap
from binary_evaluate import floating
from full_binary_search import read,save

def setup(q,catalog_name='full_binary_catalog',label='full'):
    root=HERE/'bb84'/str(float(q));job=root/label
    core,engine,model,ops=bootstrap(job)
    path=HERE/'data'/f'{catalog_name}.npz';nodes=np.load(path)['nodes']
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    events=np.array([[(i//4,i%4,k) for i,k in enumerate(r['mapping']) if k>=0] for r in ops],np.int32)
    @lru_cache(None)
    def tree(idx):
        if idx==0:return model.RAW
        op,l,r=map(int,nodes[idx,:3]);return model.Node(ops[op]['name'],tree(l),tree(r))
    def record(idx,axis):
        ed=model.structural_record(tree(int(idx)),axis,'enlarged_axes' if axis=='Y' else 'real')
        ed.update(binary_node_id=int(idx),catalog_sha256=digest,catalog_name=catalog_name)
        assert ed['memory_upper_bound']==int(nodes[idx,5+AXES.index(axis)])
        return ed
    return job,core,engine,model,ops,nodes,events,record

def run(q,catalog_name,label,axes,optimize=True):
    job,core,engine,model,ops,nodes,events,record=setup(q,catalog_name,label)
    path=job/'feasible.json'
    if path.exists() and read(path).get('complete'):
        print('ALREADY COMPLETE',q,label,flush=True);return
    from literature_rates import watanabe
    start=time.monotonic();ts=np.array([0.,1/256,1/64,1/32,1/16,1/8,3/16,1/4,3/8,1/2,3/4,1.])
    z=q*ts;source=np.stack((1-2*q+z,q-z,z,q-z),-1)
    state,y=floating(nodes,events,source)
    er=(y*np.maximum(0,2*state.max(-1)-1)**2).min(-1)
    incumbent=np.zeros(9);candidates=[];batch=2048
    for axis in axes:
        ai=AXES.index(axis);perm=list(model.AXIS_PERMUTATIONS[axis]);memory=nodes[:,5+ai]
        ids=np.flatnonzero(er>incumbent[memory]+1e-13);axis_best=np.zeros(9)
        for offset in range(0,len(ids),batch):
            chosen=ids[offset:offset+batch];p=state[chosen][...,perm].copy();yy=y[chosen].copy();mem=memory[chosen]
            for B in range(17):
                rates=(yy*watanabe(p)).min(-1)
                for m in range(1,9):
                    valid=np.flatnonzero(mem<=m)
                    if not len(valid):continue
                    top=valid[np.argsort(rates[valid])[-3:]]
                    for i in top:
                        value=float(rates[i])
                        if value>max(1e-13,axis_best[m]*.97):candidates.append((value,int(chosen[i]),axis,B,int(mem[i])))
                    axis_best[m]=max(axis_best[m],float(rates[top[-1]]))
                p,s=core.b_step(p);yy=yy*s/2
            if offset%(batch*20)==0:save(job/'status.json',dict(stage='discovery',axis=axis,processed=offset,total=len(ids),seconds=time.monotonic()-start))
        incumbent=np.maximum(incumbent,axis_best)
        print('BINARY DISCOVERY',q,label,axis,incumbent.tolist(),flush=True)
    candidates=sorted(set(candidates),reverse=True);selected=set()
    for m in range(1,9):selected.update(r for r in candidates if r[4]<=m and r[0]>=incumbent[m]*.97)
    selected=sorted(selected,reverse=True)
    save(job/'discovery.json',dict(complete=True,Q=q,label=label,axes=axes,catalog=catalog_name,
        candidates=selected,sampled_best=incumbent.tolist(),source_t=ts.tolist(),scope='Sampled Watanabe proposals only.'))
    results={f'M{m}':dict(lower=0.) for m in range(1,9)};tested=[]
    strategy=[dict(AD=dict(kind='Watanabe_2007',n=2,mode='no_otp_hashed'),allocation=1.,allocation_exact='1')]
    for sampled,idx,axis,B,memory in selected:
        if sampled<=results[f'M{memory}']['lower']+1e-11:continue
        ed=record(idx,axis);numerical,_=engine.numerical_worst(q,q,ed,B,strategy)
        if numerical<=results[f'M{memory}']['lower']+1e-11:continue
        cert=engine.certify_policy(q,q,ed,B,strategy,atol=max(2e-11,numerical*1e-7),max_cells=16000)
        assert cert['converged'],cert
        row=dict(lower=max(0.,cert['lower']),ED=ed,B=B,classical_B_rounds=B,strategy=strategy,certificate=cert)
        tested.append(dict(node=idx,axis=axis,B=B,memory=memory,lower=row['lower']))
        for m in range(memory,9):
            if row['lower']>results[f'M{m}']['lower']:results[f'M{m}']=row
        print('BINARY FULL-Z',q,label,memory,row['lower'],flush=True)
    if optimize:
        proposals={json.dumps([r['ED'],r['B']],sort_keys=True):(r['ED'],r['B']) for r in results.values() if 'ED'in r}
        # Include raw-measurement classical policies, which need not maximize
        # the Watanabe-only proposal criterion at memory one.
        for axis in axes:
            ai=AXES.index(axis);p=state[0][...,list(model.AXIS_PERMUTATIONS[axis])].copy();yy=y[0].copy()
            for B in range(17):
                cap=float((yy*np.maximum(0,2*p.max(-1)-1)**2).min())
                if cap<=results['M1']['lower']:break
                ed=record(0,axis);proposals[json.dumps([ed,B],sort_keys=True)]=(ed,B)
                p,s=core.b_step(p);yy=yy*s/2
        for i,(ed,B) in enumerate(proposals.values()):
            save(job/'status.json',dict(stage='full_AD',candidate=i,total=len(proposals),seconds=time.monotonic()-start))
            candidate=engine.solve(q,q,ed,B,nmax=128,tolerance=2e-9,max_iterations=30,incumbent=0.)
            candidate.update(ED=ed,B=B);m0=ed['memory_upper_bound']
            for m in range(m0,9):
                if candidate['lower']>results[f'M{m}']['lower']:results[f'M{m}']=candidate
    result=dict(Q=q,label=label,catalog=catalog_name,axes=axes,complete=True,results=results,tested=tested,
        expanded_AD=optimize,seconds=time.monotonic()-start,
        scope='Feasible fixed policies over every hidden source. Whole-family optimality requires separate upper.',
        source_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest())
    save(path,result);print('BINARY SEARCH DONE',q,label,{k:v['lower'] for k,v in results.items()},flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--q',type=float,required=True);ap.add_argument('--real',action='store_true')
    ap.add_argument('--axes',default='XYZ');ap.add_argument('--label');ap.add_argument('--no-optimize',action='store_true')
    a=ap.parse_args();label=a.label or ('real_'+a.axes.lower() if a.real else 'full')
    run(a.q,'real_binary_catalog' if a.real else 'full_binary_catalog',label,a.axes,not a.no_optimize)
