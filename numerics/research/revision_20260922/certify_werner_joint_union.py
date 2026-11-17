"""Certify the fixed-Clifford core against the registered recurrence union.

Every instrument and flag is covered. A recurrence witness is used only
as a valid incumbent for the union, never as a fixed-block-core witness.
"""
from core import *
from fractions import Fraction as F
from complete_werner_search import load
from screening import relevant_states
from bb84_fixed import physical_kernel
from interval_modes import components
from certify_bb84_recursive import step,watanabe_bound
from long_code_bounds import best_bounds,LongBounds
from spc_interval_dynamic import all_bounds
from certify_finite_upper import bias_setup,bias_step
import stable_intervals as iv
import argparse,time


def raw_bounds(q,weights):
    q=F(str(q));p0=iv.rational(1-F(3,2)*q);p1=iv.rational(q/2);raws=[]
    for m,w in weights.items():
        cols=[]
        for j in range(m+1):
            v=(np.array(1.),np.array(1.))
            for _ in range(m-j):v=iv.mul(v,p0)
            for _ in range(j):v=iv.mul(v,p1)
            cols.append(v)
        powers=tuple(np.array([[v[i] for v in cols]]) for i in (0,1))
        out=iv.positive_dot(powers,w.reshape(-1,m+1).T);raws.append(tuple(v.reshape(-1,4) for v in out))
    return tuple(np.concatenate([v[i] for v in raws]) for i in (0,1))


def terminal(p,ad,kernels,physical):
    n,mode,kind=ad['n'],ad['mode'],ad['kind']
    if kind=='Watanabe_2007':return watanabe_bound(p)
    if kind=='linear':
        k=kernels[ad['code']];kk=k if mode=='no_otp' else physical[ad['code']]
        gain,cost=components(iv.monomial_bounds(p,np.asarray(compositions(n))),kk)
        value=iv.positive_sum((np.maximum(0,gain[0]),np.maximum(0,gain[1])))
        if mode!='no_otp':value=iv.sub(value,iv.rational(F(n-k.k,n)) if mode=='otp_plain' else cost)
    elif kind=='SPC' and mode=='no_otp':
        lo,hi=all_bounds(p,np.array([n]),np.ones((len(p[0]),1),bool));value=lo[:,0],hi[:,0]
    else:
        long=LongBounds(p,n)
        value=(long.spc(n) if kind=='SPC' else long.repetition(n))[('no_otp','otp_plain','otp_compressed').index(mode)]
    return np.maximum(0,value[0]),np.maximum(0,value[1])


def replay(row,raw,mass,offsets,kernels,physical):
    total=(np.array(0.),np.array(0.));details=[]
    for branch in row['branches']:
        i=offsets[row['m']]+branch['state'];ps=(mass[0][i:i+1],mass[1][i:i+1])
        p=iv.div((raw[0][i:i+1],raw[1][i:i+1]),(ps[0][:,None],ps[1][:,None]));y=iv.scale(ps,F(branch['factor'],row['m']))
        for _ in range(branch['AD']['classical_B_rounds']):p,s=step(p,p,'Z');y=iv.scale(iv.mul(y,s),F(1,2))
        value=iv.mul(y,terminal(p,branch['AD'],kernels,physical));total=iv.add(total,(value[0][0],value[1][0]))
        details.append(dict(state=branch['state'],AD=branch['AD'],lower=float(value[0][0]),upper=float(value[1][0])))
    assert abs(float(total[0])-row['rate'])<2e-11,(row['ED'],row['rate'],total)
    return dict(ED=row['ED'],M=row['M'],m=row['m'],lower=max(0.,float(total[0])),upper=float(total[1]),branches=details)


def totals(data,agg,mass,best):
    bi,si,fi=agg;contrib=iv.mul(iv.mul(mass,best),(np.ones_like(mass[0]),np.ones_like(mass[1])))
    weighted=iv.mul((contrib[0][si],contrib[1][si]),iv.point(fi))
    assert int(np.bincount(bi).max())<=256
    lo=np.bincount(bi,weights=weighted[0],minlength=len(data['entries']))
    hi=np.bincount(bi,weights=weighted[1],minlength=len(data['entries']));g=iv.gamma(512)
    return np.maximum(0,iv.down(lo/(1+g))),iv.up(hi/(1-g))


def certify(q,data,weights,offsets,agg,kernels,nmax=128,K=16,tolerance=1e-10):
    start=time.time();raw=raw_bounds(q,weights);mass=iv.positive_sum(raw)
    p=iv.div(raw,(mass[0][:,None],mass[1][:,None]));factor=(np.ones(len(p[0])),np.ones(len(p[0])))
    physical={k.entry['id']:physical_kernel(k) for k in kernels};kd={k.entry['id']:k for k in kernels}
    incumbent=np.zeros(9);external=[];witnesses=[];seen=set()
    for permission in ('real','full'):
        path=DATA/f'werner_v2_{permission}_certified.json'
        if not path.exists():continue
        for v in json.loads(path.read_text())['results'].values():
            if v['Q']!=q:continue
            M=int(v['budget'][1:]);lower=max(0.,float(iv.down(float(v['lower']))));incumbent[M:]=np.maximum(incumbent[M:],lower)
            external.append(dict(source=path.name,budget=M,lower=lower))
    for name in ('werner_short_core_all_modes.json',f'werner_complete_joint_n{nmax}.json'):
        path=DATA/name
        if not path.exists():continue
        for r in json.loads(path.read_text())['results']:
            if r['Q']!=q:continue
            for row in r['results'].values():
                identity=json.dumps(row['branches'],sort_keys=True)+row['ED']
                if identity in seen:continue
                seen.add(identity);v=replay(row,raw,mass,offsets,kd,physical);witnesses.append(v)
                incumbent[v['M']:]=np.maximum(incumbent[v['M']:],v['lower'])
    bestlo=np.zeros(len(p[0]));besthi=bestlo.copy();alive=np.ones(len(p[0]),bool);parameters=bias_setup(raw);history=[]
    memories=np.array([e['M'] for e in data['entries']])
    bi,si,fi=agg;max_weight=np.zeros(len(p[0]));np.maximum.at(max_weight,si,iv.up(fi))
    flag_count=int(np.bincount(bi).max())
    # A raw-pair rate tolerance should not require the same absolute
    # conditional precision on exponentially rare public flags.
    importance=iv.up(iv.up(max_weight*mass[1])*flag_count)
    state_tolerance=iv.down(tolerance/np.maximum(importance,iv.TINY))
    for B in range(K+1):
        c=np.minimum(1,np.maximum(0,iv.up(2*p[1].max(-1)-1)));er=iv.up(factor[1]*iv.up(c*c))
        potential=iv.up(mass[1]*np.maximum(besthi,er));relevant,_,kept=relevant_states(data,agg,potential,incumbent+tolerance)
        removed=alive&~relevant;besthi[removed]=np.maximum(besthi[removed],er[removed]);alive&=relevant
        ids=np.flatnonzero(alive&(er>bestlo+state_tolerance));skipped=alive.copy();skipped[ids]=False
        besthi[skipped]=np.maximum(besthi[skipped],er[skipped]);alive[skipped]=False
        if not len(ids):break
        target=np.maximum(0,(bestlo[ids]+state_tolerance[ids])/np.maximum(factor[1][ids],1e-300))
        lo,hi=best_bounds((p[0][ids],p[1][ids]),kernels,physical,nmax,target)
        value=iv.mul((lo,hi),(factor[0][ids],factor[1][ids]));bestlo[ids]=np.maximum(bestlo[ids],value[0]);besthi[ids]=np.maximum(besthi[ids],value[1])
        low,_=totals(data,agg,mass,(bestlo,besthi))
        for M in range(1,9):incumbent[M]=max(incumbent[M],float(low[memories<=M].max()))
        history.append(dict(B=B,active_states=len(ids),potentially_competitive_instruments=int(kept.sum())))
        print('Werner union core interval',q,B,len(ids),'seconds',round(time.time()-start,1),flush=True)
        if B<K:p,factor,parameters=bias_step(parameters,factor)
    low,hi=totals(data,agg,mass,(bestlo,besthi));results={}
    recurrence_bounds={}
    for permission in ('real','full'):
        path=DATA/f'werner_v2_{permission}_menu_certified.json'
        if path.exists():recurrence_bounds[permission]=json.loads(path.read_text())['results'].get(f'{q:.12g}')
    for M in range(1,9):
        ids=np.flatnonzero(memories<=M);j=int(ids[hi[ids].argmax()]);core_upper=float(hi[j])
        results[f'M{M}']=dict(union_lower=float(incumbent[M]),core_upper=core_upper,limiting_core_ED=data['entries'][j]['id'],
                              core_cannot_improve_union_by_more_than=max(0.,float(iv.up(core_upper-incumbent[M]))))
    return dict(Q=q,results=results,core_witnesses=witnesses,external_lower_sources=external,history=history,seconds=time.time()-start,raw_pair_screening_tolerance=tolerance,
                scope='Complete fixed Clifford terminal instruments m<=8, all flags, independent per-flag B<=16 and AD choices, all short binary codes n<=7 and repetition/SPC<=nmax with paid OTP and Watanabe. Every core candidate upper is enclosed; external recurrence values are lower witnesses only. Combine core_upper with a matching recurrence whole-menu upper before calling the union optimum certified.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--q',nargs='+',type=float,default=[.1,.2,.25,.28,.3,.32]);a=parser.parse_args()
    data,weights,offsets,agg=load();kernels=load_codes();path=DATA/'werner_joint_union_core_certified.json'
    out=json.loads(path.read_text()) if path.exists() else dict(results={})
    for q in a.q:
        key=f'{q:.12g}'
        if key in out['results']:continue
        out['results'][key]=certify(q,data,weights,offsets,agg,kernels);atomic_json(path,out)
        print('Werner union core certified',q,out['results'][key]['results'],flush=True)


if __name__=='__main__':main()
