"""Dense computed curves of an explicit policy portfolio, not an optimum."""
from fractions import Fraction as F
from pathlib import Path
import argparse,csv,hashlib,json,time
import numpy as np
from full_clifford_model import HERE,AXES,bootstrap
from full_binary_search import read,save

def policies():
    job=HERE/'curves';core,engine,model,ops=bootstrap(job);output={}
    from normalize_binary import identity as canonical_identity
    action=[dict(AD=dict(kind='Watanabe_2007',n=2,mode='no_otp_hashed'),allocation=1.,allocation_exact='1')]
    def add(ed,B,strategy,permission,origin):
        content=dict(ED=ed,B=int(B),strategy=strategy)
        equivalent=dict(canonical=canonical_identity(ed),B=int(B),strategy=strategy)
        key=hashlib.sha256(json.dumps(equivalent,sort_keys=True).encode()).hexdigest()[:16]
        if key not in output:output[key]=dict(**content,memory=ed['memory_upper_bound'],permissions=[],origins=[])
        if permission not in output[key]['permissions']:output[key]['permissions'].append(permission)
        output[key]['origins'].append(origin)
    def gettree(nodes,idx):
        if idx==0:return model.RAW
        op,l,r=map(int,nodes[idx,:3]);return model.Node(ops[op]['name'],gettree(nodes,l),gettree(nodes,r))
    for label,catalog_name,permission in [('full','full_binary_catalog','full'),('real_xyz','real_binary_catalog','real_xyz')]:
        nodes=np.load(HERE/'data'/f'{catalog_name}.npz')['nodes']
        for q in ('0.17','0.2','0.22','0.23'):
            file=HERE/'bb84'/q/label/'discovery.json'
            if not file.exists() and label=='full':file=HERE/'bb84'/q/'discovery.json'
            if not file.exists():continue
            doc=read(file);rows=doc['candidates'];seen=set()
            for budget in range(1,9):
                feasible=[r for r in rows if r[4]<=budget]
                if not feasible:continue
                r=max(feasible,key=lambda x:x[0]);_,idx,axis,B,m=r
                if (idx,axis,B)in seen:continue
                seen.add((idx,axis,B));ed=model.structural_record(gettree(nodes,idx),axis,'enlarged_axes')
                assert ed['memory_upper_bound']==m
                add(ed,B,action,permission,str(file.relative_to(HERE)))
    # Public source-independent allocations improve the purely classical
    # reference. They remain fixed when Q changes along a plotted curve.
    old=HERE.parents[1]/'research/revision_20261006/recheck_20261021/full_bb84/baseline/0.17/result.json'
    prior=read(old)['best'];add(prior['ED'],prior['B'],prior['strategy'],'real_xyz',str(old))
    pointer=HERE/'bb84/0.2/refined_real_xyz_latest.json'
    if pointer.exists():
        path=Path(read(pointer)['path']).parent/'feasible.json';r=read(path)['results']['M1']
        add(r['ED'],r.get('B',r.get('classical_B_rounds',0)),r['strategy'],'real_xyz',str(path))
    for row in output.values():
        if 'real_xyz'in row['permissions'] and 'full'not in row['permissions']:row['permissions'].append('full')
    save(job/'policies.json',output);return job,core,engine,output

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--partition',type=int,default=0);ap.add_argument('--parts',type=int,default=1)
    ap.add_argument('--prepare',action='store_true');a=ap.parse_args()
    if a.prepare:
        job,core,engine,pp=policies()
        print('PORTFOLIO',len(pp),[(k,p['memory'],p['permissions']) for k,p in pp.items()]);return
    job=HERE/'curves';core,engine,_,_=bootstrap(job);pp=read(job/'policies.json')
    digest=hashlib.sha256((job/'policies.json').read_bytes()).hexdigest();start=time.monotonic();results=[]
    for i in range(161):
        if i%a.parts!=a.partition:continue
        q=F(16,100)+F(i,2000);file=job/'points'/(str(q).replace('/','-')+'.json')
        if file.exists():
            row=read(file);assert row['policy_sha256']==digest
        else:
            rates={}
            for key,p in pp.items():
                cert=engine.certify_policy(float(q),float(q),p['ED'],p['B'],p['strategy'],atol=2e-8,max_cells=6000)
                rates[key]=dict(lower=max(0.,cert['lower']),certificate=cert)
            row=dict(Q=str(q),policy_sha256=digest,results=rates);save(file,row)
        results.append(row)
        save(job/f'status_{a.partition}.json',dict(status='running',completed=len(results),partition=a.partition,parts=a.parts,Q=str(q),seconds=time.monotonic()-start))
        print('POLICY CURVE',i+1,161,str(q),flush=True)
    save(job/f'status_{a.partition}.json',dict(status='complete',completed=len(results),partition=a.partition,parts=a.parts,seconds=time.monotonic()-start))
if __name__=='__main__':main()
