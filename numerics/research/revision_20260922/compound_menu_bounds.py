"""Signed complete-menu outward coefficients for feasible BB84 duals."""
from core import *
from fractions import Fraction as F
from compound_terminals import actions,kernels,physical
from interval_modes import components,signed_mul,signed_sum
from certify_bb84_recursive import prepare,watanabe_bound
from long_code_bounds import LongBounds
import stable_intervals as iv


def menu_bounds(p,nmax=128):
    menu=actions(nmax);lo=np.zeros((len(p[0]),len(menu)));hi=lo.copy();cache={}
    power={n:iv.monomial_bounds(p,np.asarray(compositions(n))) for n in {k.n for k in kernels().values()}}
    long=LongBounds(p,nmax) if nmax>=8 else None
    for i,a in enumerate(menu):
        kind,n,mode=a['kind'],a['n'],a['mode'];key=(kind,a.get('code',n))
        if key not in cache:
            if kind=='Watanabe_2007':cache[key]=[watanabe_bound(p)]
            elif kind=='linear':
                k=kernels()[a['code']];gain,_=components(power[n],k)
                no=iv.positive_sum((np.maximum(0,gain[0]),np.maximum(0,gain[1])))
                gain,cost=components(power[n],physical(a['code']))
                gross=iv.positive_sum((np.maximum(0,gain[0]),np.maximum(0,gain[1])))
                cache[key]=[no,iv.sub(gross,iv.rational(F(n-k.k,n))),iv.sub(gross,cost)]
            elif kind=='repetition':cache[key]=long.repetition(n,clip_net=False)
            else:cache[key]=long.spc(n,clip_net=False)
        value=cache[key][0 if kind=='Watanabe_2007' else ('no_otp','otp_plain','otp_compressed').index(mode)]
        lo[:,i],hi[:,i]=value
        if kind=='SPC' and mode=='no_otp':lo[:,i]=0. # Hellinger expression is an upper relaxation here.
    if nmax>=8:
        from spc_interval_dynamic import all_bounds
        ids=[i for i,a in enumerate(menu) if a['kind']=='SPC' and a['mode']=='no_otp']
        lengths=np.array([menu[i]['n'] for i in ids]);needed=np.ones((len(p[0]),len(ids)),bool)
        a,b=all_bounds(p,lengths,needed);lo[:,ids]=a;hi[:,ids]=b
    assert np.isfinite(lo).all() and np.isfinite(hi).all()
    return lo,hi


def dual_upper(b,p,row,nmax=128):
    weights=[max(F(0),F(str(v))) for v in row['sampled_dual']];total=sum(weights)
    assert total>0;weights=[v/total for v in weights];ids=[i for i,v in enumerate(weights) if v]
    ts=[F(str(row['samples'][i])) for i in ids]
    state,y=prepare(b,p,[(t,t) for t in ts],row['ED'],row['classical_B_rounds'])
    bounds=menu_bounds(state,nmax)
    w=[iv.rational(weights[i]) for i in ids]
    scale=iv.mul(y,(np.array([v[0] for v in w]),np.array([v[1] for v in w])))
    weighted=signed_mul(bounds,(scale[0][:,None],scale[1][:,None]))
    average=signed_sum(weighted,axis=0);upper=max(0.,float(np.max(average[1])))
    return dict(upper=upper,test_parameters=[str(t) for t in ts],
                dual_weights_exact=[str(weights[i]) for i in ids],
                arithmetic='Exact feasible rational dual measure, signed net OTP coefficients, outward all-action rate bounds. Feasibility does not rely on the LP solver tolerance.')


def verify():
    from compound_terminals import values
    pp=np.array([[.61,.17,.08,.14],[.45,.05,.45,.05],[1.,0.,0.,0.]])
    lo,hi=menu_bounds((pp,pp),128);vv=values(pp,actions(128))
    assert np.all(vv>=lo-3e-13) and np.all(vv<=hi+3e-13)
    # Signed OTP loss must remain present in the dual coefficients.
    assert np.any(hi<-.1)
    atomic_json(DATA/'compound_menu_bounds_checks.json',dict(status='passed',comparisons=int(vv.size),max_width=float(np.max(hi-lo))))
    print('signed menu bounds',vv.size,float(np.max(hi-lo)),flush=True)


if __name__=='__main__':verify()
