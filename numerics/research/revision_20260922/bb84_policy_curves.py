"""Dense achievable curves for a saved finite catalog of fixed BB84 policies.

Every displayed value is a continuous-correlation lower certificate.
Floating screening only selects which catalog policies to certify; these
curves are not declared whole-generator optimum certificates.
"""
from core import *
from fractions import Fraction as F
from bb84_recursive import prepare
from bb84_joint_extension import certify_policy,numerical_worst
from compound_terminals import values
from known_search import spc_upper
import argparse,hashlib,time


def normalize(ed,B,strategy):
    ed={k:ed[k] for k in ('family','path','primitive','r','final_axis','memory_upper_bound') if k in ed}
    strategy=[dict(AD={k:v['AD'][k] for k in ('kind','code','n','mode') if k in v['AD']},allocation_exact=str(F(v['allocation_exact']))) for v in strategy]
    return dict(ED=ed,B=int(B),strategy=strategy,M=ed['memory_upper_bound'])


def policy_id(row):return hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()[:20]


def catalog(permissions):
    out={};existing={}
    def add(b,p,ed,B,strategy,certificate=None,origin=None):
        row=normalize(ed,B,strategy);tag=policy_id(row);out[tag]=row
        if certificate and 'lower' in certificate:existing[(b,p,tag)]=certificate
    path=DATA/f'bb84_recursive_watanabe_{permissions}_v2_jet_certified.json'
    for v in json.loads(path.read_text())['results'].values():
        if not v.get('AD'):continue
        a=dict(kind='Watanabe_2007',n=2,mode='no_otp_hashed')
        add(v['b'],v['p'],v['ED'],v['AD']['classical_B_rounds'],[dict(AD=a,allocation_exact='1')],v['certificate'])
    for path in DATA.glob(f'bb84_joint_extension_{permissions}_*_n128.json'):
        src=json.loads(path.read_text())
        for v in src['results'].values():
            if v.get('strategy'):
                add(src['b'],src['p'],v['ED'],v['classical_B_rounds'],v['strategy'],v.get('certificate'))
    for axis in ('Z','X') if permissions=='real' else ('Z','X','Y'):
        ed=dict(family='real_decoder_recurrence',path='',final_axis=axis,memory_upper_bound=1)
        for a in (dict(kind='linear',code='XC0000',n=1,mode='no_otp'),dict(kind='Watanabe_2007',n=2,mode='no_otp_hashed')):
            add(None,None,ed,0,[dict(AD=a,allocation_exact='1')])
    return out,existing


def sampled_upper(b,p,row,ts):
    state,y=prepare(b,p,ts,row['ED'],row['B']);gain=np.zeros(len(ts))
    for item in row['strategy']:
        a=item['AD']
        if a['kind']=='SPC' and a['mode']=='no_otp':v=spc_upper(state,np.array([a['n']]))[:,0]
        else:v=values(state,[a])[:,0]
        gain+=float(F(item['allocation_exact']))*v
    return max(0.,float((y*gain).min()))+3e-12


def point(b,p,policies,existing,relative=1e-4,max_cells=20000,absolute=1e-11):
    if b+p>=.5:return dict(b=b,p=p,results={f'M{m}':dict(lower=0.,upper_for_selected_policy=0.,reason='compatible separable input') for m in range(1,9)})
    if b==p==0:return dict(b=b,p=p,results={f'M{m}':dict(lower=1.,upper_for_selected_policy=1.,reason='pure input, direct measurement') for m in range(1,9)})
    ts=np.unique(np.r_[np.linspace(0,1,65),np.logspace(-8,-2,13),1-np.logspace(-8,-2,13)])
    ranking=[(sampled_upper(b,p,row,ts),tag) for tag,row in policies.items()];ranking.sort(reverse=True)
    best={f'M{m}':dict(lower=0.,reason='discard') for m in range(1,9)};computed=0
    cache_dir=DATA/'bb84_curve_certificates';cache_dir.mkdir(exist_ok=True)
    for upper,tag in ranking:
        row=policies[tag];M=row['M'];inc=best[f'M{M}']['lower']
        if upper<=inc+max(absolute,relative*inc):continue
        cache=cache_dir/f'{b:.8f}_{p:.8f}_{tag}.json'
        cert=existing.get((b,p,tag))
        if cache.exists():cert=json.loads(cache.read_text())
        if cert is None:
            # A sampled value is only an upper screen. The subsequent
            # whole-interval certificate, not a floating local minimizer,
            # establishes the displayed lower rate. Repeated golden-section
            # searches are particularly wasteful for long SPC terminals.
            pp,yy=prepare(b,p,ts,row['ED'],row['B'])
            menu=[item['AD'] for item in row['strategy']]
            weights=np.array([float(F(item['allocation_exact'])) for item in row['strategy']])
            numerical=float((yy*(values(pp,menu)@weights)).min())
            if numerical<=inc:continue
            atol=max(absolute,relative*max(0.,numerical))
            cert=certify_policy(b,p,row['ED'],row['B'],row['strategy'],atol=atol,max_cells=max_cells)
            atomic_json(cache,cert);computed+=1
        lower=max(0.,float(cert['lower']))
        for m in range(M,9):
            if lower>best[f'M{m}']['lower']:
                best[f'M{m}']=dict(lower=lower,upper_for_selected_policy=float(cert['upper']),policy=tag,
                                   certificate_converged=bool(cert.get('converged',True)),certificate_cells=cert.get('cells'))
    return dict(b=b,p=p,results=best,new_continuous_certificates=computed)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--permissions',choices=['real','full'],default='real')
    parser.add_argument('--q',nargs='+',type=float);parser.add_argument('--p',type=float);parser.add_argument('--dense',action='store_true')
    parser.add_argument('--relative',type=float,default=1e-4);a=parser.parse_args();policies,existing=catalog(a.permissions)
    digest=hashlib.sha256(json.dumps(policies,sort_keys=True).encode()).hexdigest();registry=DATA/f'bb84_curve_policies_{a.permissions}.json'
    atomic_json(registry,dict(policies=policies,digest=digest))
    path=DATA/f'bb84_policy_curves_{a.permissions}.json';out=json.loads(path.read_text()) if path.exists() else dict(results={})
    for b in [i/2000 for i in range(501)] if a.dense else a.q or [.1,.15,.18,.2,.22,.24]:
        p=b if a.p is None else a.p;key=f'{b:.8f}_{p:.8f}'
        if key in out['results'] and out['results'][key].get('catalog_digest')==digest:continue
        start=time.time();r=point(b,p,policies,existing,a.relative);r.update(catalog_digest=digest,seconds=time.time()-start)
        out['results'][key]=r;out['scope']='Achievable envelope from a finite saved catalog. Each displayed positive rate is a whole-correlation interval lower certificate. Floating candidate screening is not an all-catalog or all-generator optimality proof; independent anchor certificates have separate files.'
        atomic_json(path,out);print('BB curve certified',a.permissions,b,p,{k:v['lower'] for k,v in r['results'].items()},round(time.time()-start,1),flush=True)


if __name__=='__main__':main()
