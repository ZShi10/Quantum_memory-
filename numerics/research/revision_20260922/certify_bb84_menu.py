"""Finite-menu upper certificates paired with continuous-state lower bounds."""
from bb84_fixed import *
from interval_modes import best_short_bounds,dual_upper
from certify_finite_upper import raw_bb
import stable_intervals as iv


def certify_file(path):
    source=json.loads(path.read_text());b,p=source['b'],source['p'];started=time.time()
    entries=[e for e in complete_short_entries(b==p) if source['permissions']=='full' or e['real']]
    allkernels=load_codes(source['nmax'],all_codes=True)
    kernels=[k for k in allkernels if k.entry['indecomposable']]
    physical={k.entry['id']:physical_kernel(k) for k in kernels}
    upper=np.full(len(entries),np.inf)
    anchors=[F(0),F(1,4),F(1,2),F(3,4),F(1)]
    for t in anchors:
        raw,starts,ms=raw_bb(entries,F(str(b)),F(str(p)),t)
        mass=iv.positive_sum(raw);active=mass[0]>0
        safe=(np.where(active,mass[0],1),np.where(active,mass[1],1))
        state=iv.div(raw,(safe[0][:,None],safe[1][:,None]))
        lo=np.zeros(len(ms));hi=np.ones(len(ms))
        ids=np.flatnonzero(active&(state[1].max(-1)>.5))
        hi[active]=0
        lo[ids],hi[ids]=best_short_bounds((state[0][ids],state[1][ids]),kernels,physical)
        contribution=iv.div(iv.mul(mass,(lo,hi)),(ms,ms))
        totalhi=iv.up(np.add.reduceat(contribution[1],starts[:-1])/(1-iv.gamma(256)))
        upper=np.minimum(upper,totalhi)
    certificates={}
    for i,e in enumerate(entries):
        row=source['evaluated'].get(e['id'])
        if row is None:continue
        if 'sampled_dual' not in row:
            row=solve(e,b,p,allkernels,tolerance=1e-8)
        if not row.get('strategy'):continue
        menu=Menu(raw_polynomials(e,b,p),e['m'],allkernels)
        c=dual_upper(menu,e,b,p,row['samples'],row['sampled_dual'])
        certificates[e['id']]=c;upper[i]=min(upper[i],c['upper'])
    results={}
    for M in (1,2):
        ids=[i for i,e in enumerate(entries) if e['M']<=M]
        row=source['results'].get(f'M{M}',{})
        lower=row.get('certificate',{}).get('lower',row.get('lower','0'))
        lower=str(max(Decimal(0),Decimal(lower)))
        bound=float(max(upper[ids]))
        results[f'M{M}']=dict(lower=lower,upper=bound,
                              gap_upper=str(Decimal.from_float(bound)-Decimal(lower)),
                              lower_ED=row.get('ED'),limiting_upper_ED=entries[ids[int(np.argmax(upper[ids]))]]['id'])
    output=dict(b=b,p=p,permissions=source['permissions'],nmax=source['nmax'],entry_count=len(entries),
                results=results,per_entry_upper={e['id']:float(v) for e,v in zip(entries,upper)},
                sampled_dual_certificates=certificates,seconds=time.time()-started,
                scope='Complete registered fixed m<=4 terminal instrument class, one fixed ED choice; n<=N all binary code classes, all three Du modes, fixed randomized allocations and retention. No B prefix. Whole-z lower and whole-menu outward upper.')
    dest=path.with_name(path.stem+'_certified.json');atomic_json(dest,output)
    print('BB84 finite certified',source['permissions'],b,p,results,'seconds',round(time.time()-started,1),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',nargs='+',required=True);args=parser.parse_args()
    for name in args.source:certify_file(DATA/name)


if __name__=='__main__':main()
