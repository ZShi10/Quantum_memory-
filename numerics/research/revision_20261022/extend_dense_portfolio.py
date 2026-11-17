"""Add certified short-code winners and an explicit real/XZ curve control."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,time
from full_clifford_model import HERE,bootstrap
from full_binary_search import read,save

def main():
    job=HERE/'curves_extended';core,engine,model,_=bootstrap(job)
    from fast_entropy import install
    install()
    from normalize_binary import identity
    base=HERE/'curves_augmented'
    if (job/'policies.json').exists():policies=read(job/'policies.json')
    else:
        policies=read(base/'policies.json')
        for row in policies.values():
            impl=row['implementations'].get('real_xyz')
            if impl and impl['final_axis'] in ('X','Z'):
                row['permissions'].append('real_xz');row['implementations']['real_xz']=impl
        for pointer in HERE.glob('bb84/*/refined_*_latest.json'):
            result=Path(read(pointer)['path']);feasible=result.parent/'feasible.json'
            if not feasible.exists():continue
            label=pointer.name[len('refined_'):-len('_latest.json')]
            for row in read(feasible)['results'].values():
                strategy=row.get('strategy',[])
                if not strategy or len(strategy)>8 or any(v['AD']['n']>8 for v in strategy):continue
                ed=row['ED'];B=row.get('B',row.get('classical_B_rounds',0))
                objective=dict(canonical=identity(ed),B=B,strategy=strategy)
                key=hashlib.sha256(json.dumps(objective,sort_keys=True).encode()).hexdigest()[:16]
                if key not in policies:policies[key]=dict(ED=ed,B=B,strategy=strategy,memory=ed['memory_upper_bound'],permissions=[],implementations={},origins=[])
                target=policies[key]
                permission=['full']+(['real_xyz'] if label.startswith('real') else [])+(['real_xz'] if label=='real_xz' else [])
                for name in permission:
                    if name not in target['permissions']:target['permissions'].append(name)
                    target['implementations'][name]=ed
                target['origins'].append(str(feasible))
        save(job/'policies.json',policies)
    digest=hashlib.sha256((job/'policies.json').read_bytes()).hexdigest();order=sorted(policies,key=lambda k:(policies[k]['memory'],k))
    start=time.monotonic()
    for index in range(161):
        q=F(16,100)+F(index,2000);name=str(q).replace('/','-')+'.json';file=job/'points'/name
        if file.exists():assert read(file)['policy_sha256']==digest;continue
        old=read(base/'points'/name)['results'];inc={name:[0.]*9 for name in ('real_xz','real_xyz','full')};win={name:[None]*9 for name in inc};rates={}
        for key in order:
            policy=policies[key];memory=policy['memory'];permissions=policy['permissions'];prior=old.get(key,{})
            threshold=min(inc[label][memory] for label in permissions)
            upper=prior.get('known_source_upper')
            if upper is None:
                tests=[F(0),F(1,8),F(1,4),F(1,2),F(1)]
                upper=max(0.,float(engine.curve(float(q),float(q),[(t,t) for t in tests],policy['ED'],policy['B'],policy['strategy']).v[1].min()))
            if upper<=threshold:
                rates[key]=dict(status='screened',known_source_upper=upper,screening_target=threshold);continue
            if prior.get('status')=='full_interval':
                result=dict(prior,reused_from=str(base/'points'/name))
                cert=prior['certificate'];tolerance=max(2e-10,abs(cert['upper'])*2e-5)
                improve=memory>=2 and not cert['converged'] and cert['upper']>max(threshold,prior['lower'])+tolerance
            else:result=None;improve=True;tolerance=max(2e-10,upper*2e-5)
            if improve:
                cert=engine.certify_policy(float(q),float(q),policy['ED'],policy['B'],policy['strategy'],atol=tolerance,max_cells=24000)
                result=dict(status='full_interval',lower=max(0.,cert['lower']),certificate=cert,known_source_upper=upper)
                if prior.get('lower',0)>result['lower']:result=dict(prior,reused_from=str(base/'points'/name))
            rates[key]=result
            for label in permissions:
                for m in range(memory,9):
                    if result['lower']>inc[label][m]:inc[label][m]=result['lower'];win[label][m]=key
        save(file,dict(Q=str(q),policy_sha256=digest,results=rates,lower_envelopes=inc,winners=win,
            base_sha256=hashlib.sha256((base/'points'/name).read_bytes()).hexdigest(),
            scope='Fixed-portfolio achievable lower curves over the full hidden interval. No whole-family optimum or interpolation theorem.'))
        save(job/'status.json',dict(stage='running',completed=index+1,Q=str(q),seconds=time.monotonic()-start))
        print('EXTENDED CURVE',index+1,161,str(q),flush=True)
    save(job/'status.json',dict(stage='complete',completed=161,policies=len(policies),seconds=time.monotonic()-start))

if __name__=='__main__':main()
