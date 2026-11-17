"""Outward relaxation for every m<=4 fixed Clifford instrument.

At each exact compatible state, each public ED flag may independently
choose any B<=K and any registered classical action. This deliberately
enlarges the fixed unknown-state protocol class. Its minimum over test
states is an upper bound, not an achievable hidden-parameter policy.
"""
from core import *
from fractions import Fraction as F
from libraries import complete_short_entries
from certify_bb84_recursive import source,step
from long_code_bounds import best_bounds
from bb84_fixed import physical_kernel
import stable_intervals as iv
import argparse,time


def incumbent(b,p,permission):
    path=DATA/f'bb84_joint_extension_{permission}_{b:.5f}_{p:.5f}_n128.json'
    out=np.zeros(3)
    if path.exists():
        src=json.loads(path.read_text())
        for m in (1,2):out[m]=src['results'][f'M{m}']['lower']
    return out


def raw_boxes(b,p,ts,entries):
    initial=source(b,p,[(t,t) for t in ts]);power={}
    raw=[];owners=[]
    for j,e in enumerate(entries):
        m=e['m']
        if m not in power:power[m]=iv.monomial_bounds(initial,np.array(compositions(m)))
        tensor=np.asarray(e['tensor'])
        value=iv.positive_dot(power[m],tensor.reshape(-1,tensor.shape[-1]).T)
        raw.append(tuple(v.reshape(len(ts),-1,4).transpose(1,0,2) for v in value))
        owners.extend([j]*len(tensor))
    return tuple(np.concatenate([r[k] for r in raw]) for k in (0,1)),np.array(owners)


def certify(b,p,permission='real',nmax=128,K=16,tolerance=1e-8):
    dest=DATA/f'bb84_block_relaxation_{permission}_{b:.5f}_{p:.5f}_n{nmax}.json'
    start=time.time();entries=complete_short_entries(b==p)
    if permission=='real':entries=[e for e in entries if e['real']]
    # Exact test states can be added without changing the protocol class.
    ts=[F(0),F(1)]
    incumbent_path=DATA/f'bb84_joint_extension_{permission}_{b:.5f}_{p:.5f}_n128.json'
    if incumbent_path.exists():
        previous=json.loads(incumbent_path.read_text())['results']
        for row in (previous['M1'],previous['M2']):
            t=row.get('certificate',{}).get('argmin')
            if t is not None and 0<=t<=1:ts.append(F(str(t)))
    ts=sorted(set(ts))
    if dest.exists():
        saved=json.loads(dest.read_text())
        if saved.get('complete') and all(v['dominated_by_recursive_witness'] for v in saved['results'].values()):return
    raw,owner=raw_boxes(b,p,ts,entries);mass=iv.positive_sum(raw)
    m=np.array([e['m'] for e in entries]);M=np.array([e['M'] for e in entries]);assert M.max()<=2
    weight=iv.scale(mass,F(1))
    weight=iv.div(weight,iv.point(m[owner,None]))
    good=mass[0]>0
    denominator=(np.where(good,mass[0],1.)[...,None],np.where(good,mass[1],1.)[...,None])
    state=iv.div(raw,denominator)
    # An exactly zero probability flag contributes zero. A nonzero uncertain
    # flag receives the full probability simplex, never an invented state.
    state=(np.where(good[...,None],state[0],0.),np.where(good[...,None],state[1],1.))
    shape=mass[0].shape;state=tuple(v.reshape(-1,4) for v in state)
    factor=(np.ones(state[0].shape[0]),np.ones(state[0].shape[0]))
    best=np.zeros(len(factor[0]));active=np.ones(len(entries),bool);upper=np.full(len(entries),np.inf)
    lower=incumbent(b,p,permission);kernels=load_codes();physical={k.entry['id']:physical_kernel(k) for k in kernels}
    flagcount=max(np.bincount(owner));rounds=[]
    def aggregate(values):
        terms=iv.up(weight[1]*values.reshape(shape));total=np.zeros((len(entries),len(ts)))
        np.add.at(total,owner,terms)
        return iv.up(total/(1-iv.gamma(2*flagcount))).min(-1)
    for B in range(K+1):
        concurrence=np.minimum(1,np.maximum(0,iv.up(2*state[1].max(-1)-1)))
        er=iv.up(factor[1]*iv.up(concurrence*concurrence))
        potential=aggregate(np.maximum(best,er))
        upper[active]=potential[active]
        active &= potential>lower[M]+tolerance
        ids=np.flatnonzero(np.repeat(active[owner],len(ts)))
        ids=ids[(factor[1][ids]>0)&(er[ids]>best[ids])&(weight[1].reshape(-1)[ids]>0)]
        for first in range(0,len(ids),96):
            selected=ids[first:first+96]
            importance=iv.up(weight[1].reshape(-1)[selected]*flagcount*factor[1][selected])
            localtol=iv.down(tolerance/np.maximum(importance,1e-300))
            selected_owner=owner[selected//len(ts)]
            allocated=iv.down(lower[M[selected_owner]]/np.maximum(importance,1e-300))
            target=np.maximum(best[selected]/np.maximum(factor[1][selected],iv.TINY),np.maximum(localtol,allocated))
            _,bound=best_bounds((state[0][selected],state[1][selected]),kernels,physical,nmax,target)
            best[selected]=np.maximum(best[selected],iv.up(factor[1][selected]*bound))
        assert np.isfinite(best).all()
        if B==K:upper[active]=aggregate(best)[active]
        rounds.append(dict(B=B,active_instruments=int(active.sum()),conditional_states=len(ids)))
        out=dict(b=b,p=p,permissions=permission,nmax=nmax,K=K,rounds=rounds,complete=B==K or not active.any(),
                 test_parameters=[str(t) for t in ts],incumbent={f'M{k}':float(lower[k]) for k in (1,2)},
                 results={f'M{k}':dict(upper=float(upper[M<=k].max()),
                     dominated_by_recursive_witness=bool(upper[M<=k].max()<=lower[k]+tolerance)) for k in (1,2)},
                 entries=[dict(ED=e['id'],M=e['M'],m=e['m'],upper=float(u)) for e,u in zip(entries,upper)],
                 scope='All m<=4 complete terminal Clifford instruments under the stated permission; all flags; per-flag B<=K; full registered AD menu. Test-statewise choices relax a fixed compound protocol. Upper only; combine with matching recurrence certificate for union claims.',
                 seconds=time.time()-start,tolerance=tolerance)
        atomic_json(dest,out)
        print('BB block upper',b,p,permission,B,out['results'],'active',active.sum(),'seconds',round(time.time()-start,1),flush=True)
        if not active.any():break
        if B<K:
            state,s=step(state,state,'Z');factor=iv.scale(iv.mul(factor,s),F(1,2))
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--q',type=float,nargs='+',required=True)
    parser.add_argument('--p',type=float);parser.add_argument('--permissions',choices=['real','full'],default='real')
    a=parser.parse_args()
    for b in a.q:certify(b,b if a.p is None else a.p,a.permissions)


if __name__=='__main__':main()
