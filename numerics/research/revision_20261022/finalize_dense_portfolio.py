"""Include late control winners in the dense, matched achievable curves.

The earlier portfolio and its certificates remain immutable. A certificate
is reused only for the identical normalized policy; every additional
operation label has a concrete implementation checked separately.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse,hashlib,json,time
from full_clifford_model import HERE,bootstrap
from full_binary_search import read,save


def prepare(job):
    from normalize_binary import identity
    base=HERE/'curves_extended';policies=read(base/'policies.json')
    additions=[];sources=[]
    files=[]
    for label in ('full','real_xyz','real_xz'):
        for source in HERE.glob(f'bb84/*/{label}/feasible.json'):
            files.append((label,source))
    for pointer in HERE.glob('bb84/*/refined_*_latest.json'):
        source=Path(read(pointer)['path']).parent/'feasible.json'
        if source.exists():files.append((pointer.name[len('refined_'):-len('_latest.json')],source))
    for label,source in files:
        sources.append(dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        for result in read(source)['results'].values():
            if 'ED' not in result or not result.get('strategy'):continue
            assert result['lower']<=max(0.,result['certificate']['lower'])+1e-18
            ed=result['ED'];B=int(result.get('B',result.get('classical_B_rounds',0)));strategy=result['strategy']
            objective=dict(canonical=identity(ed),B=B,strategy=strategy)
            key=hashlib.sha256(json.dumps(objective,sort_keys=True).encode()).hexdigest()[:16]
            # Retain the existing one-slot portfolio. This update adds the
            # late winners at the plotted two-or-more-slot budgets; it is
            # still a finite feasible portfolio, not an exhaustive curve.
            if key not in policies and ed['memory_upper_bound']==1:continue
            if key not in policies:
                policies[key]=dict(ED=ed,B=B,strategy=strategy,memory=ed['memory_upper_bound'],permissions=[],implementations={},origins=[])
                additions.append(key)
            target=policies[key];assert target['memory']==ed['memory_upper_bound'],key
            permissions=['full']+(['real_xyz'] if label.startswith('real') else [])+(['real_xz'] if label=='real_xz' else [])
            for permission in permissions:
                if permission not in target['permissions']:target['permissions'].append(permission)
                target['implementations'][permission]=ed
            origin=str(source)
            if origin not in target['origins']:target['origins'].append(origin)
    save(job/'policies.json',policies)
    save(job/'preparation.json',dict(base_sha256=hashlib.sha256((base/'policies.json').read_bytes()).hexdigest(),
        source_files=sources,added_policies=additions,policies=len(policies)))
    return policies


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--refresh-preparation',action='store_true');args=ap.parse_args()
    job=HERE/'curves_final';core,engine,model,ops=bootstrap(job)
    from fast_entropy import install
    install()
    if args.refresh_preparation:
        assert not list((job/'points').glob('*.json')), 'Never replace a portfolio after its point calculations begin.'
        policies=prepare(job)
    else:policies=read(job/'policies.json') if (job/'policies.json').exists() else prepare(job)
    additions=read(job/'preparation.json')['added_policies']
    if args.prepare_only:
        print(json.dumps(dict(policies=len(policies),added=[dict(key=k,memory=policies[k]['memory'],B=policies[k]['B'],
            N=policies[k]['ED']['successful_ancestry_inputs'],axis=policies[k]['ED']['final_axis'],
            actions=[v['AD'] for v in policies[k]['strategy']]) for k in additions]),indent=2));return
    digest=hashlib.sha256((job/'policies.json').read_bytes()).hexdigest()
    order=sorted(policies,key=lambda k:(policies[k]['memory'],k));base=HERE/'curves_extended';start=time.monotonic()
    for index in range(161):
        q=F(16,100)+F(index,2000);name=str(q).replace('/','-')+'.json';path=job/'points'/name
        if path.exists():assert read(path)['policy_sha256']==digest;continue
        old=read(base/'points'/name)['results'];inc={v:[0.]*9 for v in ('real_xz','real_xyz','full')}
        winners={v:[None]*9 for v in inc};rates={}
        for key in order:
            policy=policies[key];memory=policy['memory'];permissions=policy['permissions'];prior=old.get(key,{})
            if prior.get('status')=='full_interval':
                result=dict(prior,reused_from=str(base/'points'/name))
            else:
                threshold=min(inc[v][memory] for v in permissions);upper=prior.get('known_source_upper')
                if upper is None:
                    tests=[F(0),F(1,8),F(1,4),F(1,2),F(1)]
                    upper=max(0.,float(engine.curve(float(q),float(q),[(t,t) for t in tests],policy['ED'],policy['B'],policy['strategy']).v[1].min()))
                if upper<=threshold:
                    rates[key]=dict(status='screened',known_source_upper=upper,screening_target=threshold);continue
                cert=engine.certify_policy(float(q),float(q),policy['ED'],policy['B'],policy['strategy'],
                    atol=max(2e-10,upper*2e-5),max_cells=24000)
                result=dict(status='full_interval',lower=max(0.,cert['lower']),certificate=cert,known_source_upper=upper)
            rates[key]=result
            for permission in permissions:
                for m in range(memory,9):
                    if result['lower']>inc[permission][m]:inc[permission][m]=result['lower'];winners[permission][m]=key
        previous=read(base/'points'/name)
        for label in inc:
            assert all(a+1e-16>=b for a,b in zip(inc[label],previous['lower_envelopes'][label]))
        save(path,dict(Q=str(q),policy_sha256=digest,results=rates,lower_envelopes=inc,winners=winners,
            base_sha256=hashlib.sha256((base/'points'/name).read_bytes()).hexdigest(),
            scope='Fixed-policy achievable envelopes over every hidden source. Late control winners included; no whole-family or continuum-in-Q optimality claim.'))
        save(job/'status.json',dict(stage='running',completed=index+1,Q=str(q),seconds=time.monotonic()-start))
        print('FINAL CURVE',index+1,161,str(q),flush=True)
    save(job/'status.json',dict(stage='complete',completed=161,policies=len(policies),seconds=time.monotonic()-start))


if __name__=='__main__':main()
