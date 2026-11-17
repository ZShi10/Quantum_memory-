"""Complete finite registered recurrence/AD upper certificates for BB84.

Every candidate is bounded, including candidates omitted by floating
discovery. Exact feasible duals sharpen the known-state relaxations.
"""
from core import *
from fractions import Fraction as F
from recurrence_interval_library import generate_bounds,records,step
from recurrence_equivalence import representatives,key as canonical_key
from certify_bb84_recursive import source
from compound_menu_bounds import dual_upper
from long_code_bounds import best_bounds
from bb84_fixed import physical_kernel
import stable_intervals as iv
import argparse,time,hashlib


def certify(path,tolerance=1e-8):
    src=json.loads(path.read_text());b,p=src['b'],src['p'];permission=src['permissions'];nmax=src['nmax'];K=src['K']
    dest=path.with_name(path.stem+'_menu_certified.json');started=time.time()
    out=json.loads(dest.read_text()) if dest.exists() else dict(b=b,p=p,permissions=permission,nmax=nmax,K=K,duals={},rounds=[])
    if out.get('complete') and all(v['converged'] for v in out['results'].values()):return
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if out.get('source_digest')!=digest:
        # New feasible dual measures can tighten previously completed
        # rounds. Their old upper bounds remain valid, but skipping them
        # would silently leave the newly resolved candidates unresolved.
        out['rounds']=[];out['source_digest']=digest
    if out.get('interval_generator')!='unnormalized_pumping_v1':
        out['rounds']=[];out['interval_generator']='unnormalized_pumping_v1'
    # A completed certificate remains valid, but new lower witnesses can
    # improve its gap without any need to change the upper.
    lower=np.array([0.]+[src['results'][f'M{m}']['lower'] for m in range(1,9)])
    anchors=[F(0),F(1,64),F(1,8),F(1,2),F(1)]
    pp,yy=generate_bounds(source(b,p,[(t,t) for t in anchors]),permission)
    M,rr=records(permission);rep=representatives(rr,b==p);ids=np.flatnonzero(rep==np.arange(len(rep)))
    pp=tuple(v[ids] for v in pp);yy=tuple(v[ids] for v in yy);M=M[ids]
    evaluated={}
    def identity(ed,B):
        return json.dumps([canonical_key(ed) if b==p else ed,B],sort_keys=True)
    for row in src['evaluated'].values():
        if 'sampled_dual' not in row:continue
        tag=identity(row['ED'],row['classical_B_rounds'])
        if tag not in out['duals']:
            out['duals'][tag]=dual_upper(b,p,row,nmax);atomic_json(dest,out)
        evaluated[tag]=out['duals'][tag]['upper']
    kernels=load_codes();physical={k.entry['id']:physical_kernel(k) for k in kernels}
    for B in range(K+1):
        if B<len(out['rounds']):
            if B<K:
                pp,s=step(pp,pp,'Z');yy=iv.scale(iv.mul(yy,s),F(1,2))
            continue
        concurrence=np.maximum(0,iv.up(2*pp[1].max(-1)-1))
        er=np.minimum(yy[1],iv.up(yy[1]*iv.up(concurrence*concurrence)))
        upper=er.min(-1);anchor=er.argmin(-1)
        for j,original in enumerate(ids):
            tag=identity(rr[original],B)
            if tag in evaluated:upper[j]=min(upper[j],evaluated[tag])
        active=np.flatnonzero(upper>lower[M]+tolerance)
        for start in range(0,len(active),128):
            local=active[start:start+128];a=anchor[local];scale=yy[1][local,a]
            target=np.maximum(0,(lower[M[local]]+tolerance)/np.maximum(scale,1e-300))
            _,bound=best_bounds((pp[0][local,a],pp[1][local,a]),kernels,physical,nmax,target)
            upper[local]=np.minimum(upper[local],iv.up(scale*bound))
        assert np.isfinite(upper).all()
        budgets={f'M{m}':float(upper[M<=m].max()) for m in range(1,9)}
        unresolved=np.flatnonzero(upper>lower[M]+tolerance)
        order=unresolved[np.argsort(-(upper[unresolved]-lower[M[unresolved]]))]
        out['rounds'].append(dict(B=B,upper=budgets,known_state_evaluations=len(active),
                                 unresolved=[dict(original_index=int(ids[i]),ED=rr[ids[i]],upper=float(upper[i])) for i in order]))
        atomic_json(dest,out)
        print('BB full menu upper',b,p,permission,B,budgets,'unresolved',len(unresolved),'seconds',round(time.time()-started,1),flush=True)
        if B<K:pp,s=step(pp,pp,'Z');yy=iv.scale(iv.mul(yy,s),F(1,2))
    out['results']={}
    for m in range(1,9):
        key=f'M{m}';hi=max(r['upper'][key] for r in out['rounds']);lo=lower[m]
        assert hi+1e-14>=lo
        out['results'][key]=dict(lower=float(lo),upper=hi,gap_upper=float(iv.up(hi-lo)),converged=hi-lo<=tolerance)
    out.update(complete=True,seconds=time.time()-started,tolerance=tolerance,
               scope='All registered depth<=7 four-decoder words, constant-decoder raw pumps<=160, prescribed final bases, explicit live-register schedules, fixed B<=K and fixed mixtures of all registered classical actions. Every candidate bounded by exact-point interval relaxations; selected LP dual measures repaired exactly. Whole-z lower witnesses from source; no unrestricted protocol optimality claim.')
    atomic_json(dest,out);print('BB menu certified',b,p,permission,out['results'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',nargs='+',required=True);parser.add_argument('--tolerance',type=float,default=1e-8)
    a=parser.parse_args()
    for name in a.source:certify(DATA/name,a.tolerance)


if __name__=='__main__':main()
