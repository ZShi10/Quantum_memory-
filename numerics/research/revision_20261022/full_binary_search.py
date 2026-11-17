"""Finite full-intermediate-Clifford search with a fixed hidden-source policy.

Discovery values are not certified optima. Feasible winners are separately
certified over the complete hidden interval before being exported as lowers.
"""
from pathlib import Path
from functools import lru_cache
from fractions import Fraction as F
import argparse, hashlib, json, math, time
import numpy as np
from full_clifford_model import HERE,PROJECT,REV,AXES,bootstrap
from binary_evaluate import floating

def read(p):return json.loads(Path(p).read_bytes())
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.writing.json');tmp.write_text(json.dumps(x,indent=2)+'\n')
    for attempt in range(100):
        try:tmp.replace(p);return
        except PermissionError:
            if attempt==99:raise
            time.sleep(.05)

def setup(q):
    job=HERE/'bb84'/str(q);core,engine,model,ops=bootstrap(job)
    nodes=np.load(HERE/'data/full_binary_catalog.npz')['nodes']
    events=np.array([[(i//4,i%4,k) for i,k in enumerate(r['mapping']) if k>=0] for r in ops],np.int32)
    @lru_cache(None)
    def tree(idx):
        if idx==0:return model.RAW
        op,left,right=map(int,nodes[idx,:3])
        return model.Node(ops[op]['name'],tree(left),tree(right))
    def record(idx,axis):
        ed=model.structural_record(tree(int(idx)),axis,'enlarged_axes')
        ed['full_clifford_node_id']=int(idx)
        ed['catalog_sha256']=read(HERE/'data/full_binary_catalog.json')['sha256']
        assert ed['memory_upper_bound']==int(nodes[idx,5+AXES.index(axis)])
        return ed
    return job,core,engine,model,ops,nodes,events,record

def discovery(q,optimize=False):
    job,core,engine,model,ops,nodes,events,record=setup(q)
    from literature_rates import watanabe
    start=time.monotonic();ts=np.array([0.,1/256,1/64,1/32,1/16,1/8,3/16,1/4,3/8,1/2,3/4,1.])
    z=q*ts;source=np.stack((1-2*q+z,q-z,z,q-z),-1)
    state,y=floating(nodes,events,source)
    er=(y*np.maximum(0,2*state.max(-1)-1)**2).min(-1)
    # These known-rate seeds only screen discovery; whole-family certification
    # never depends on their floating use in candidate selection.
    incumbent=np.zeros(9)
    previous=REV/'recheck_20261021/full_bb84/fixed_curves_extended'
    # The first pass samples every ED and every allowed B at every key axis.
    # After it finds candidates, the continuum engine verifies selected ones.
    candidates=[];batch=2048
    for ai,axis in enumerate(AXES):
        perm=list(model.AXIS_PERMUTATIONS[axis]);memory=nodes[:,5+ai]
        ids=np.flatnonzero(er>incumbent[memory]+1e-12)
        axis_best=np.zeros(9)
        for offset in range(0,len(ids),batch):
            chosen=ids[offset:offset+batch]
            p=state[chosen][...,perm].copy();yield_=y[chosen].copy();mem=memory[chosen]
            bound=er[chosen]
            for B in range(17):
                rates=(yield_*watanabe(p)).min(-1)
                for m in range(1,9):
                    valid=np.flatnonzero(mem<=m)
                    if not len(valid):continue
                    top=valid[np.argsort(rates[valid])[-3:]]
                    for index in top:
                        value=float(rates[index])
                        if value>max(1e-12,axis_best[m]-.03*axis_best[m]):
                            candidates.append((value,int(chosen[index]),axis,B,int(mem[index])))
                    axis_best[m]=max(axis_best[m],float(rates[top[-1]]))
                p,s=core.b_step(p);yield_=yield_*s/2
            if offset%(batch*20)==0:
                save(job/'discovery_status.json',dict(axis=axis,processed=offset,total=len(ids),
                  best=axis_best.tolist(),seconds=time.monotonic()-start))
        incumbent=np.maximum(incumbent,axis_best)
        print('FULL CLIFFORD DISCOVERY AXIS',q,axis,incumbent.tolist(),flush=True)
    del state,y
    candidates=sorted(set(candidates),reverse=True)
    # Keep leading candidates for every memory. Grid winners only propose
    # policies; every claimed lower is a full hidden-interval certificate.
    selected=set()
    for m in range(1,9):
        selected.update(row for row in candidates if row[4]<=m and row[0]>=incumbent[m]*.97)
    selected=sorted(selected,reverse=True)
    save(job/'discovery.json',dict(Q=q,complete=True,candidates=selected,
        sampled_best=incumbent.tolist(),nodes=len(nodes),source_t=ts.tolist(),
        seconds=time.monotonic()-start,scope='Watanabe-only proposals; sampled scores not guarantees or optima.'))
    strategy=[dict(AD=dict(kind='Watanabe_2007',n=2,mode='no_otp_hashed'),allocation=1.,allocation_exact='1')]
    results={f'M{m}':dict(lower=0.) for m in range(1,9)};tested=[]
    for sampled,idx,axis,B,memory in selected:
        if sampled<=results[f'M{memory}']['lower']+1e-10:continue
        ed=record(idx,axis)
        numerical,_=engine.numerical_worst(q,q,ed,B,strategy)
        if numerical<=results[f'M{memory}']['lower']+1e-10:continue
        cert=engine.certify_policy(q,q,ed,B,strategy,atol=max(2e-10,numerical*1e-6),max_cells=16000)
        assert cert['converged'],cert
        row=dict(lower=max(0.,cert['lower']),ED=ed,B=B,strategy=strategy,certificate=cert)
        tested.append(dict(node=idx,axis=axis,B=B,memory=memory,lower=row['lower']))
        for m in range(memory,9):
            if row['lower']>results[f'M{m}']['lower']:results[f'M{m}']=row
        save(job/'feasible.json',dict(Q=q,complete=False,results=results,tested=tested))
        print('FULL-Z LOWER',q,memory,idx,axis,B,row['lower'],flush=True)
    if optimize:
        seen=set()
        for m in range(1,9):
            prior=results[f'M{m}']
            if 'ED' not in prior:continue
            ed,B=prior['ED'],prior['B'];key=json.dumps([ed,B],sort_keys=True)
            if key in seen:continue
            seen.add(key)
            candidate=engine.solve(q,q,ed,B,nmax=128,tolerance=2e-9,max_iterations=30,incumbent=0.)
            candidate.update(ED=ed,B=B)
            memory=ed['memory_upper_bound']
            for budget in range(memory,9):
                if candidate['lower']>results[f'M{budget}']['lower']:results[f'M{budget}']=candidate
    result=dict(Q=q,complete=True,results=results,tested=tested,expanded_AD=optimize,
        seconds=time.monotonic()-start,scope='Achievable fixed-policy full-hidden-interval lowers, not a whole-family optimum.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(job/'feasible.json',result)
    print(json.dumps(dict(Q=q,results={k:v['lower'] for k,v in results.items()},seconds=result['seconds'])),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--q',type=float,required=True)
    ap.add_argument('--optimize',action='store_true');args=ap.parse_args()
    discovery(args.q,args.optimize)
