"""Fast independent SPC certificate for a raw Y readout of symmetric BB84.

After any classical B prefix this source satisfies p_X=p_Y exactly. For a
word with w bit errors, the phase-complement kernel entropy is B**w K[n-w],
where B=p_X+p_Y. Pairing zero-bit phase words gives the O(n**2) expression
below. This identity is NOT used for a generic Bell-diagonal state.
"""
from pathlib import Path
from fractions import Fraction as F
from math import comb
import argparse,hashlib,json,time
import numpy as np
from full_clifford_model import HERE,bootstrap
from full_binary_search import read,save


def paired_rates(p,lengths):
    import interval_jets as j
    import stable_intervals as iv
    nmax=max(lengths);a=p[...,0];c=p[...,3];b=(p[...,1]+p[...,2]).clip()
    one=j.Jet((np.ones(len(p.v[0])),np.ones(len(p.v[0]))))
    powers=[]
    for v in (a,c,b):
        pp=[one]
        for _ in range(nmax):pp.append((pp[-1]*v).clip())
        powers.append(j.stack(pp))
    aa,cc,bb=powers
    mm=np.array([m for m in range(1,nmax+1) for k in range(m//2+1)])
    kk=np.array([k for m in range(1,nmax+1) for k in range(m//2+1)])
    x=(aa[:,mm-kk]*cc[:,kk]).clip();y=(aa[:,kk]*cc[:,mm-kk]).clip()
    coefficients=[iv.rational(F(comb(int(m),int(k)),2 if 2*k==m else 1)) for m,k in zip(mm,kk)]
    weights=j.Jet(tuple(np.array([v[i] for v in coefficients]) for i in (0,1)))
    loss=(j.h(x)+j.h(y)-j.h((x+y).clip()))*weights
    # K_0=1: with no zero-bit positions the phase-complement bit is fair.
    kernel=j.stack([one]+[loss[:,mm==m].sum() for m in range(1,nmax+1)])
    h=j.h(p);total=h.sum();difference=h.dot(np.array([1,0,0,1]))-h.dot(np.array([0,1,1,0]))
    d=1-2*b;dp=[one]
    for _ in range(nmax):dp.append(dp[-1]*d)
    result={}
    for n in lengths:
        w=np.arange(n+1);coef=[iv.rational(F(comb(n,int(k)))) for k in w]
        coefficient=j.Jet(tuple(np.array([v[i] for v in coef]) for i in (0,1)))
        weighted=bb[:,w]*kernel[:,n-w]*coefficient;value=j.Jet.constant(0)
        for sign,s in ((1,0),(-1,1)):
            mass=((1+sign*dp[n])/2).clip()
            raw=n*(total+sign*dp[n-1]*difference)/2
            lost=weighted[:,w%2==s].sum()
            value=value+j.positive((n-1)*mass-raw+j.h(mass)+lost)/n
        result[n]=value
    return result


def rates(p,lengths):
    """One-dimensional Hellinger walk with an even/odd bit-error flag.

    A zero-bit symbol shifts a by +/-1 with weight sqrt(pI*pZ).
    The two equal bit-error symbols have combined weight pX+pY, leave a
    unchanged, and toggle the syndrome parity. The likelihood is a*u,
    so summing the second displacement loses no information.
    """
    import interval_jets as j
    import stable_intervals as iv
    from spc_jet_dynamic import sqrt_jet,logarithm_ratio,ambiguity_jet,abs_jet
    nmax=max(lengths);count=len(p.v[0]);safe=(p.v[0][:,0]>1e-150)&(p.v[0][:,3]>1e-150)
    if not safe.all():
        out={}
        if safe.any():a=rates(p[np.flatnonzero(safe)],lengths)
        other=paired_rates(p[np.flatnonzero(~safe)],lengths)
        for n in lengths:
            vl=np.empty(count);vh=np.empty(count);dl=np.empty(count);dh=np.empty(count)
            for ids,result in [(np.flatnonzero(safe),a if safe.any() else None),(np.flatnonzero(~safe),other)]:
                if len(ids):vl[ids],vh[ids],dl[ids],dh[ids]=*result[n].v,*result[n].d
            out[n]=j.Jet((vl,vh),(dl,dh))
        return out
    g=sqrt_jet(p[...,0]*p[...,3]);b=(p[...,1]+p[...,2]).clip()
    u=logarithm_ratio(p[...,0],p[...,3]);h=j.h(p);H=h.sum();Hp=h.dot(np.array([1,0,0,1]))-h.dot(np.array([0,1,1,0]))
    one=j.Jet((np.ones(count),np.ones(count)));d=1-2*b;dp=[one]
    for _ in range(nmax):dp.append(dp[-1]*d)
    initial=np.zeros((count,1,2));initial[:,:,0]=1.;distribution=(initial.copy(),initial.copy());out={}
    for n in range(1,nmax+1):
        previous=distribution;nxt=(np.zeros((count,2*n+1,2)),np.zeros((count,2*n+1,2)))
        gg=iv.mul((g.v[0][:,None,None],g.v[1][:,None,None]),previous)
        bb=iv.mul((b.v[0][:,None,None],b.v[1][:,None,None]),tuple(v[:,:,::-1] for v in previous))
        for term,sl in ((gg,(slice(None),slice(None,-2),slice(None))),(gg,(slice(None),slice(2,None),slice(None))),
                        (bb,(slice(None),slice(1,-1),slice(None)))):
            value=iv.add(tuple(v[sl] for v in nxt),term)
            for i in (0,1):nxt[i][sl]=np.maximum(0,value[i])
        distribution=nxt
        if n not in lengths:continue
        derivative=(np.zeros_like(nxt[0]),np.zeros_like(nxt[1]))
        dg=j.smul((g.d[0][:,None,None],g.d[1][:,None,None]),previous)
        db=j.smul((b.d[0][:,None,None],b.d[1][:,None,None]),tuple(v[:,:,::-1] for v in previous))
        for term,sl in ((dg,(slice(None),slice(None,-2),slice(None))),(dg,(slice(None),slice(2,None),slice(None))),
                        (db,(slice(None),slice(1,-1),slice(None)))):
            value=iv.add(tuple(v[sl] for v in derivative),j.smul(term,iv.rational(F(n))))
            for i in (0,1):derivative[i][sl]=value[i]
        displacement=np.arange(-n,n+1);L=abs_jet(u[:,None]*j.Jet((displacement,displacement)))
        weighted=j.Jet(distribution,derivative)*ambiguity_jet(L)[:,:,None];value=j.Jet.constant(0)
        for sign,s in ((1,0),(-1,1)):
            mass=((1+sign*dp[n])/2).clip();physical=n*(H+sign*dp[n-1]*Hp)/2
            value=value+j.positive((n-1)*mass-physical+j.h(mass)+weighted[:,:,s].sum())/n
        out[n]=value
    return out


def validate():
    core,engine,model,_=bootstrap(HERE/'diagnostics/symmetric_spc')
    import interval_jets as j
    from compound_terminals import spc_no_jet
    from spc_jet_dynamic import spc_jet
    from spc_dynamic_fast import all_rates
    rng=np.random.default_rng(20261022);raw=rng.uniform(.01,1,(12,3));raw/=raw.sum(-1)[:,None]
    p=np.stack((raw[:,0],raw[:,1]/2,raw[:,1]/2,raw[:,2]),-1)
    point=j.Jet((p,p));lengths=[2,3,4,7,8,11,19,37,73]
    fast=rates(point,lengths);paired=paired_rates(point,lengths);numerical=all_rates(p,np.array(lengths));largest=0.
    for n in lengths:
        generic=spc_no_jet(point,n) if n<=11 else spc_jet(point,n)
        a=fast[n];assert np.all(a.v[0]<=generic.v[1]) and np.all(generic.v[0]<=a.v[1]),n
        assert np.all(a.v[0]<=paired[n].v[1]) and np.all(paired[n].v[0]<=a.v[1]),n
        error=np.max(np.abs((a.v[0]+a.v[1])/2-numerical[:,lengths.index(n)]));largest=max(largest,float(error))
        assert error<2e-11,(n,error)
    # Full raw-Y+B family, including zero-touching source endpoints.
    intervals=[(F(0),F(0)),(F(1),F(1))]+[(F(i,32),F(i+1,32)) for i in range(32)]
    ed=model.structural_record(model.RAW,'Y','enlarged_axes')
    for B in (0,1,2,3):
        state,_=j.prepare(.23,.23,intervals,ed,B);fast=rates(state,[3,8,13])
        for n in (3,8,13):
            generic=spc_no_jet(state,n);a=fast[n]
            assert np.all(a.v[0]<=generic.v[1]) and np.all(generic.v[0]<=a.v[1]),(B,n)
            assert np.all(a.d[0]<=generic.d[1]) and np.all(generic.d[0]<=a.d[1]),(B,n,'derivative')
    result=dict(status='passed',random_sources=12,lengths=lengths,source_cells=34,B_prefixes=[0,1,2,3],
        numerical_max_difference=largest,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        proof_identity='pX=pY; pair each phase word with its global complement, sum the fair bit-error positions first. K0=1. All binomial weights enclosed from exact integers.')
    save(HERE/'diagnostics/symmetric_spc/validation.json',result);print(json.dumps(result),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate',action='store_true');ap.add_argument('--candidate',type=Path)
    ap.add_argument('--q',type=float,default=.23);ap.add_argument('--atol',type=float,default=5e-10)
    ap.add_argument('--resume-work',type=Path);args=ap.parse_args()
    if args.validate:validate();return
    assert args.candidate
    candidate=read(args.candidate)['candidate'];ed=candidate['ED'];B=candidate['B'];strategy=candidate['strategy']
    assert ed['tree']=={'kind':'raw'} and ed['final_axis']=='Y'
    identity=dict(Q=args.q,candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        formula_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),atol=args.atol)
    job=HERE/'bb84'/str(args.q)/'symmetric_lower'/hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    core,engine,model,_=bootstrap(job)
    from fast_entropy import install
    install()
    import interval_jets as j
    from compound_terminals import terminal_jet
    lengths=sorted({v['AD']['n'] for v in strategy if v['AD']['kind']=='SPC' and v['AD']['mode']=='no_otp'})
    cache={}
    def curve(b,p,intervals,record,rounds,allocation,coarse=False):
        assert b==p==args.q and record==ed and rounds==B and allocation==strategy
        key=tuple(intervals)
        if key in cache:return cache[key]
        state,y=j.prepare(b,p,intervals,record,rounds)
        cached=rates(state,lengths) if lengths else {};value=j.Jet.constant(0)
        for item in allocation:
            a=item['AD'];v=cached[a['n']] if a['kind']=='SPC' and a['mode']=='no_otp' else terminal_jet(state,a)
            value=value+F(item['allocation_exact'])*v
        result=y*value
        if len(cache)>4:cache.clear()
        cache[key]=result;return result
    engine.curve=curve;save(job/'spec.json',identity);start=time.monotonic()
    if args.resume_work:
        import shutil
        previous=read(args.resume_work/'spec.json')
        assert previous['candidate_sha256']==identity['candidate_sha256'] and previous['Q']==args.q
        copied=[]
        for src in (args.resume_work/'data/bb84_interval_work').glob('*.json'):
            target=job/'data/bb84_interval_work'/src.name;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copyfile(src,target)
            copied.append(dict(name=src.name,sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
        save(job/'reused_interval_cover.json',dict(source=str(args.resume_work),files=copied,
            note='Existing valid full-domain boxes are retained. Further subdivision uses an independently validated equivalent formula.'))
    cert=engine.certify_policy(args.q,args.q,ed,B,strategy,atol=args.atol,max_cells=100000)
    result=dict(complete=True,Q=args.q,lower=max(0.,cert['lower']),ED=ed,B=B,strategy=strategy,certificate=cert,
        identity=identity,seconds=time.monotonic()-start,scope='One fixed achievable raw-Y policy over the whole hidden interval; no family optimality claim.')
    save(job/'result.json',result);save(job.parent/'latest.json',dict(path=str(job/'result.json'),sha256=hashlib.sha256((job/'result.json').read_bytes()).hexdigest()))
    print(json.dumps(dict(lower=result['lower'],certificate=cert,seconds=result['seconds'])),flush=True)


if __name__=='__main__':main()
