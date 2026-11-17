"""Extend the frozen plotted portfolio, including known seven/eight-slot policies.

Each family keeps a concrete implementation, even when exact frame transport
allows a common entropy/cost evaluation. Earlier point certificates are reused
only for the identical policy; screened rows are reconsidered if needed.
"""
from fractions import Fraction as F
from pathlib import Path
from functools import lru_cache
import hashlib,json,time
import numpy as np
from full_clifford_model import HERE,bootstrap
from full_binary_search import read,save

def prepare(job,model,ops):
    from normalize_binary import identity
    pp=read(HERE/'curves/policies.json')
    watanabe=[dict(AD=dict(kind='Watanabe_2007',n=2,mode='no_otp_hashed'),allocation=1.,allocation_exact='1')]
    for row in pp.values():row['implementations']={'full':row['ED']}
    def add(ed,B,strategy,permission,origin,include_new=True):
        equivalent=dict(canonical=identity(ed),B=int(B),strategy=strategy)
        key=hashlib.sha256(json.dumps(equivalent,sort_keys=True).encode()).hexdigest()[:16]
        if key not in pp:
            if not include_new:return
            pp[key]=dict(ED=ed,B=int(B),strategy=strategy,memory=ed['memory_upper_bound'],permissions=[],origins=[],implementations={})
        row=pp[key];assert row['memory']==ed['memory_upper_bound']
        if permission not in row['permissions']:row['permissions'].append(permission)
        row['implementations'][permission]=ed
        row['origins'].append(origin)
        if permission=='real_xyz':
            if 'full'not in row['permissions']:row['permissions'].append('full')
            row['implementations'].setdefault('full',ed)
    for label,catalog in [('full','full_binary_catalog'),('real_xyz','real_binary_catalog')]:
        nodes=np.load(HERE/'data'/f'{catalog}.npz')['nodes']
        @lru_cache(None)
        def tree(idx):
            if idx==0:return model.RAW
            op,left,right=map(int,nodes[idx,:3]);return model.Node(ops[op]['name'],tree(left),tree(right))
        for q in ('0.17','0.19','0.2','0.21','0.22','0.23'):
            file=HERE/'bb84'/q/label/'discovery.json'
            if not file.exists() and label=='full':file=HERE/'bb84'/q/'discovery.json'
            if not file.exists():continue
            rows=read(file)['candidates']
            for row in rows:
                _,idx,axis,B,m=row;ed=model.structural_record(tree(int(idx)),axis,'enlarged_axes')
                add(ed,B,watanabe,label,str(file.relative_to(HERE)),False)
            winners={tuple(max((r for r in rows if r[4]<=m),key=lambda r:r[0])) for m in range(1,9)}
            for _,idx,axis,B,m in winners:
                ed=model.structural_record(tree(int(idx)),axis,'enlarged_axes')
                add(ed,B,watanabe,label,str(file.relative_to(HERE)))
    baseline=HERE.parents[1]/'research/revision_20261006/recheck_20261021/full_bb84/baseline/0.17/result.json'
    r=read(baseline)['best'];add(r['ED'],r['B'],r['strategy'],'real_xyz',str(baseline))
    for pointer in HERE.glob('bb84/*/refined_real_xyz_latest.json'):
        path=Path(read(pointer)['path']).parent/'feasible.json'
        for r in read(path)['results'].values():
            if len(r.get('strategy',[]))<=4:
                add(r['ED'],r.get('B',r.get('classical_B_rounds',0)),r['strategy'],'real_xyz',str(path))
    folder=HERE.parents[1]/'revised/reproduction/reproduction/extensions/data'
    for m in (6,7,8):
        path=folder/f'BB24_m{m}_lower_certificate.json';spec=read(path)['specification']
        ed=model.structural_record(model.homogeneous_tree(spec['word']),spec['axis'],'real')
        assert ed['memory_upper_bound']==m
        add(ed,0,watanabe,'real_xyz',str(path))
    for key,row in pp.items():
        assert set(row['permissions'])<=set(row['implementations']),(key,row['permissions'],row['origins'])
    save(job/'policies.json',pp)
    return pp

def main():
    job=HERE/'curves_augmented';job.mkdir(parents=True,exist_ok=True)
    core,engine,model,ops=bootstrap(job)
    pp=read(job/'policies.json') if (job/'policies.json').exists() else prepare(job,model,ops)
    digest=hashlib.sha256((job/'policies.json').read_bytes()).hexdigest()
    ordered=sorted(pp,key=lambda k:(pp[k]['memory'],-len(pp[k]['strategy']),k))
    start=time.monotonic()
    for index in range(161):
        q=F(16,100)+F(index,2000);name=str(q).replace('/','-')+'.json';file=job/'points'/name
        if file.exists():assert read(file)['policy_sha256']==digest;continue
        base=HERE/'curves_screened/points'/name
        while not base.exists():
            save(job/'status.json',dict(stage='waiting_for_base_point',completed=index,Q=str(q)))
            time.sleep(10)
        old=read(base)['results'];inc={f:[0.]*9 for f in ('real_xyz','full')};win={f:[None]*9 for f in inc};rates={}
        for key in ordered:
            policy=pp[key];m=policy['memory'];permission=policy['permissions'];prior=old.get(key,{})
            threshold=min(inc[f][m] for f in permission)
            if prior.get('status')=='full_interval':
                result=dict(prior,reused_from=str(base))
            else:
                tests=[F(0),F(1,8),F(1,4),F(1,2),F(1)]
                upper=prior.get('known_source_upper')
                if upper is None:
                    value=engine.curve(float(q),float(q),[(t,t) for t in tests],policy['ED'],policy['B'],policy['strategy']).v[1]
                    upper=max(0.,float(np.min(value)))
                if upper<=threshold:
                    rates[key]=dict(status='screened',known_source_upper=upper,screening_target=threshold,test_t=list(map(str,tests)))
                    continue
                cert=engine.certify_policy(float(q),float(q),policy['ED'],policy['B'],policy['strategy'],atol=2e-8,max_cells=6000)
                result=dict(status='full_interval',lower=max(0.,cert['lower']),certificate=cert,known_source_upper=upper)
            rates[key]=result
            for f in permission:
                for budget in range(m,9):
                    if result['lower']>inc[f][budget]:inc[f][budget]=result['lower'];win[f][budget]=key
        save(file,dict(Q=str(q),policy_sha256=digest,results=rates,lower_envelopes=inc,winners=win,
            base_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
            scope='Feasible fixed-portfolio envelopes over all hidden t. Concrete real and full implementations retained separately. No continuum-in-Q or whole-family optimality is inferred.'))
        save(job/'status.json',dict(stage='running',completed=index+1,Q=str(q),seconds=time.monotonic()-start))
        print('AUGMENTED CURVE',index+1,161,str(q),flush=True)
    save(job/'status.json',dict(stage='complete',completed=161,policies=len(pp),seconds=time.monotonic()-start))
if __name__=='__main__':main()
