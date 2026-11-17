"""Whole finite-menu upper certificate for the streamwise compound bound."""
from bb84_compound import *
from interval_modes import components,signed_mul,signed_sum,best_short_bounds
from certify_finite_upper import raw_bb
import stable_intervals as iv


def menu_values(menu,entry,b,p,ts):
    b,p=F(str(b)),F(str(p));initial=[]
    for t in ts:
        z=min(b,p)*F(str(t));initial.append([1-b-p+z,b-z,z,p-z])
    bounds=[[iv.rational(v) for v in row] for row in initial]
    bell=(np.array([[v[0] for v in row] for row in bounds]),np.array([[v[1] for v in row] for row in bounds]))
    powers=iv.monomial_bounds(bell,np.asarray(compositions(entry['m'])))
    lows=np.zeros((len(ts),menu.count));highs=lows.copy()
    for branch,tensor in enumerate(entry['tensor']):
        raw=iv.positive_dot(powers,np.asarray(tensor).T);mass=iv.positive_sum(raw);active=mass[0]>0
        safe=(np.where(active,mass[0],1),np.where(active,mass[1],1))
        state=iv.div(raw,(safe[0][:,None],safe[1][:,None]))
        local={n:iv.monomial_bounds(state,np.asarray(compositions(n))) for n in {k.n for k in menu.kernels}}
        scale=iv.scale(mass,F(1,entry['m']))
        for o in menu.options:
            if o['branch']!=branch:continue
            k,kk=o['kernel'],o['entropy_kernel'];gain,cost=components(local[k.n],kk)
            gross=iv.positive_sum((np.maximum(0,gain[0]),np.maximum(0,gain[1])))
            if o['mode']=='no_otp':cost=(np.zeros(len(ts)),np.zeros(len(ts)))
            elif o['mode']=='otp_plain':cost=iv.rational(F(k.n-k.k,k.n))
            net=signed_mul(iv.sub(gross,cost),scale)
            lows[:,o['alpha']],highs[:,o['alpha']]=net
            if not active.all():
                assert np.all(mass[1][~active]<1e-300)
                lows[~active,o['alpha']]=-mass[1][~active];highs[~active,o['alpha']]=mass[1][~active]
    return lows,highs


def dual_upper(menu,entry,b,p,ts,dual):
    weights=[max(F(0),F(str(v))) for v in dual];total=sum(weights);assert total>0
    weights=[v/total for v in weights];ids=[i for i,v in enumerate(weights) if v]
    values=menu_values(menu,entry,b,p,[ts[i] for i in ids]);wb=[iv.rational(weights[i]) for i in ids]
    factor=(np.array([v[0] for v in wb])[:,None],np.array([v[1] for v in wb])[:,None])
    avg=signed_sum(signed_mul(values,factor),axis=0)
    best=np.array([max(0.,float(avg[1][menu.groups==j].max())) for j in range(len(menu.raw))])
    upper=float(iv.positive_sum((best,best))[1])
    return dict(upper=upper,test_parameters=[str(F(str(ts[i]))) for i in ids],
                dual_weights_exact=[str(weights[i]) for i in ids],
                arithmetic='Outward binary64 with analytic log remainder; exact feasible rational dual; signed OTP net coefficients; simplex support.')


def certify_file(path):
    source=json.loads(path.read_text());b,p=source['b'],source['p'];start=time.time()
    entries=[e for e in complete_short_entries(b==p) if source['permissions']=='full' or e['real']]
    assert source['entry_count']==len(entries),'The file is a single-entry probe, not the complete core.'
    kernels=load_codes(source['nmax']);physical={k.entry['id']:physical_kernel(k) for k in kernels}
    upper=np.full(len(entries),np.inf)
    for t in (F(0),F(1,4),F(1,2),F(3,4),F(1)):
        raw,starts,ms=raw_bb(entries,F(str(b)),F(str(p)),t);mass=iv.positive_sum(raw);active=mass[0]>0
        safe=(np.where(active,mass[0],1),np.where(active,mass[1],1))
        state=iv.div(raw,(safe[0][:,None],safe[1][:,None]));lo=np.zeros(len(ms));hi=np.ones(len(ms))
        ids=np.flatnonzero(active&(state[1].max(-1)>.5));hi[active]=0
        lo[ids],hi[ids]=best_short_bounds((state[0][ids],state[1][ids]),kernels,physical)
        contribution=iv.div(iv.mul(mass,(lo,hi)),(ms,ms))
        totalhi=iv.up(np.add.reduceat(contribution[1],starts[:-1])/(1-iv.gamma(256)))
        upper=np.minimum(upper,totalhi)
    certs={}
    for i,e in enumerate(entries):
        row=source['evaluated'].get(e['id'])
        if row is None:continue
        if 'sampled_dual' not in row:row=solve(e,b,p,kernels)
        menu=CompoundMenu(raw_polynomials(e,b,p),e['m'],kernels)
        c=dual_upper(menu,e,b,p,row['samples'],row['sampled_dual']);certs[e['id']]=c;upper[i]=min(upper[i],c['upper'])
    results={}
    for M in (1,2):
        ids=[i for i,e in enumerate(entries) if e['M']<=M];row=source['results'].get(f'M{M}',{})
        lower=str(max(Decimal(0),Decimal(row.get('certificate',{}).get('lower',row.get('lower','0')))))
        bound=float(max(upper[ids]));lim=ids[int(np.argmax(upper[ids]))]
        results[f'M{M}']=dict(lower=lower,upper=bound,gap_upper=str(Decimal.from_float(bound)-Decimal(lower)),
                              lower_ED=row.get('ED'),limiting_upper_ED=entries[lim]['id'])
    out=dict(b=b,p=p,permissions=source['permissions'],nmax=source['nmax'],entry_count=len(entries),results=results,
             per_entry_upper={e['id']:float(v) for e,v in zip(entries,upper)},sampled_dual_certificates=certs,
             seconds=time.time()-start,scope='Fixed single m<=4 terminal ED; all n<=N codes and three Du modes with fixed mixtures. Streamwise positive post-IR entropy, signed paid OTP; no B prefix or Watanabe action. Continuous-z lower and whole-menu outward upper; direct-sum reduction includes fixed component mixtures.')
    dest=path.with_stem(path.stem+'_certified');atomic_json(dest,out);print('BB compound certified',b,p,results,'seconds',round(time.time()-start,1),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',nargs='+',required=True);a=parser.parse_args()
    for name in a.source:certify_file(DATA/name)


if __name__=='__main__':main()
