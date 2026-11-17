"""Numerical upper certificates for the complete finite protocol classes.

All probability/entropy calculations use float_enclosures. BB84 evaluates
EVERY candidate at a single admissible z, an upper bound on its minimax
value; the independent whole-policy certificate supplies the lower bound.
"""
from common import *
import argparse,json,time,hashlib
from fractions import Fraction as F
import numpy as np
import float_enclosures as iv
from libraries import complete_short_entries,code_kernels
from algebra import compositions
from complete_werner_search import load
from screening import incumbents,relevant_states


def bell_bb(eb,ep,t):
    z=min(eb,ep)*t;v=[1-eb-ep+z,eb-z,z,ep-z];a=[iv.rational(x) for x in v]
    return np.array([[x[0] for x in a]]),np.array([[x[1] for x in a]])


def raw_bb(entries,eb,ep,t):
    p=bell_bb(eb,ep,t);powers={m:iv.monomial_bounds(p,np.asarray(compositions(m))) for m in range(1,5)}
    lows=[];highs=[];starts=[0];ms=[]
    for e in entries:
        a,b=iv.positive_dot(powers[e['m']],np.asarray(e['tensor']).reshape(-1,powers[e['m']][0].shape[1]).T)
        lows.append(a.reshape(-1,4));highs.append(b.reshape(-1,4));starts.append(starts[-1]+len(lows[-1]));ms.extend([e['m']]*len(lows[-1]))
    return (np.concatenate(lows),np.concatenate(highs)),np.array(starts),np.array(ms)


def bb_upper(eb,ep,entries,kernels,source):
    dest=DATA/f'bb84_upper_{float(eb):.5f}_{float(ep):.5f}.json'
    saved=json.loads(dest.read_text()) if dest.exists() else dict(eb=str(eb),ep=str(ep),results={})
    started=time.time()
    for mode,record in source['results'].items():
        if mode in saved['results']:continue
        t=F(record['argmin']);raw,starts,ms=raw_bb(entries,eb,ep,t)
        ps=iv.positive_sum(raw);positive=ps[0]>0
        safe=(np.where(positive,ps[0],1),np.where(positive,ps[1],1))
        state=iv.div(raw,(safe[0][:,None],safe[1][:,None]))
        # No Bell entanglement means no positive virtual hashing yield.
        active=np.flatnonzero(positive&(state[1].max(1)>.5));lower=np.zeros(len(ms));upper=lower.copy()
        upper[~positive]=1 # Vanishing/underflow mass, at most one bit per pair.
        lo,hi,_=iv.best_code_bounds((state[0][active],state[1][active]),kernels)
        lower[active]=lo;upper[active]=hi
        contribution=iv.div(iv.mul(ps,(lower,upper)),(ms,ms))
        g=iv.gamma(20)
        totalhi=iv.up(np.add.reduceat(contribution[1],starts[:-1])/(1-g))
        gate,M=mode.split('_M');M=int(M)
        ids=[i for i,e in enumerate(entries) if e['M']<=M and (gate=='full' or e['real'])]
        ub=float(totalhi[ids].max())
        saved['results'][mode]=dict(upper=ub,test_t=str(t),active_states=len(active))
        saved.update(scope='Rigorous binary64 forward-error and outward interval upper, including an analytic logarithm-series remainder. Complete m<=4,n<=7,single-AD-round class at one admissible t; valid minimax upper.',seconds=time.time()-started)
        dest.write_text(json.dumps(saved,indent=2))
        print('Certified finite BB upper',float(eb),float(ep),mode,ub,'seconds',round(time.time()-started,1),flush=True)


def bernoulli_update(state,factor):
    columns=[(state[0][:,i],state[1][:,i]) for i in range(4)];a,x,y,z=columns
    out=[iv.add(iv.mul(a,a),iv.mul(z,z)),iv.add(iv.mul(x,x),iv.mul(y,y)),iv.scale(iv.mul(x,y),2),iv.scale(iv.mul(a,z),2)]
    raw=(np.maximum(0,np.stack([v[0] for v in out],-1)),np.stack([v[1] for v in out],-1));mass=iv.positive_sum(raw)
    return iv.div(raw,(mass[0][:,None],mass[1][:,None])),iv.scale(iv.mul(factor,mass),F(1,2))


def bias_setup(raw):
    def col(j):return raw[0][:,j],raw[1][:,j]
    a=iv.add(col(0),col(3));b=iv.add(col(1),col(2))
    flip=a[0]+a[1]<b[0]+b[1]
    rr=(raw[0].copy(),raw[1].copy())
    for v in rr:v[flip]=v[flip][:,[1,0,3,2]]
    a=iv.add((rr[0][:,0],rr[1][:,0]),(rr[0][:,3],rr[1][:,3]))
    b=iv.add((rr[0][:,1],rr[1][:,1]),(rr[0][:,2],rr[1][:,2]))
    def bias(i,j,den):
        v=iv.sub((rr[0][:,i],rr[1][:,i]),(rr[0][:,j],rr[1][:,j]))
        lo=np.where((v[0]<=0)&(v[1]>=0),0,np.minimum(np.abs(v[0]),np.abs(v[1])))
        hi=np.maximum(np.abs(v[0]),np.abs(v[1]));x=iv.div((lo,hi),den)
        return np.maximum(0,x[0]),np.minimum(1,x[1])
    return iv.div(b,a),bias(0,3,a),bias(1,2,b)


def bias_step(parameters,factor):
    beta,c0,c1=parameters;one=(np.ones(len(beta[0])),np.ones(len(beta[0])))
    new=[iv.mul(v,v) for v in parameters];bn,d0,d1=new
    denom=iv.scale(iv.add(one,bn),2)
    minus0=iv.sub(one,d0);minus1=iv.sub(one,d1)
    minus0=(np.maximum(0,minus0[0]),minus0[1]);minus1=(np.maximum(0,minus1[0]),minus1[1])
    numerators=[iv.add(one,d0),iv.mul(bn,iv.add(one,d1)),iv.mul(bn,minus1),minus0]
    pp=(np.stack([v[0] for v in numerators],-1),np.stack([v[1] for v in numerators],-1))
    p=iv.div(pp,(denom[0][:,None],denom[1][:,None]))
    current=iv.add(one,beta);yield_=iv.div(iv.add(one,bn),iv.scale(iv.mul(current,current),2))
    return p,iv.mul(factor,yield_),new


def werner_upper(q,data,weights,offsets,aggregation,kernels,K=16,tol=1e-12):
    p0=iv.rational(1-F(3,2)*q);p1=iv.rational(q/2);raws=[]
    for m,w in weights.items():
        cols=[]
        for j in range(m+1):
            val=(np.array(1.),np.array(1.))
            for _ in range(m-j):val=iv.mul(val,p0)
            for _ in range(j):val=iv.mul(val,p1)
            cols.append(val)
        powers=(np.array([[v[0] for v in cols]]),np.array([[v[1] for v in cols]]))
        a,b=iv.positive_dot(powers,w.reshape(-1,m+1).T);raws.append((a.reshape(-1,4),b.reshape(-1,4)))
    raw=(np.concatenate([v[0] for v in raws]),np.concatenate([v[1] for v in raws]));ps=iv.positive_sum(raw)
    state=iv.div(raw,(ps[0][:,None],ps[1][:,None]));factor=(np.ones(len(ps[0])),np.ones(len(ps[0])))
    bestlo=np.zeros(len(ps[0]));besthi=bestlo.copy();visited=[];started=time.time();first=0
    initial_concurrence=np.minimum(1,np.maximum(0,iv.up(2*state[1].max(1)-1)))
    initial_enthi=iv.up(initial_concurrence*initial_concurrence)
    incumbent=incumbents(q)
    relevant,_,kept=relevant_states(data,aggregation,iv.up(ps[1]*initial_enthi),incumbent)
    print('Certified instrument screen',float(q),'keep',int(kept.sum()),'of',len(kept),'states',int(relevant.sum()),flush=True)
    checkpoint=DATA/f'werner_work_{float(q):.5f}.npz';digest=hashlib.sha256(Path(iv.__file__).read_bytes()).hexdigest()
    if checkpoint.exists():
        c=np.load(checkpoint)
        if str(c['engine_sha'])==digest and int(c['K'])==K:
            first=int(c['next_round'])
            bestlo=c['bestlo'];besthi=c['besthi'];visited=list(map(int,c['visited']))
            print('Resume Werner interval',float(q),'round',first,flush=True)
        c.close()
    parameters=bias_setup(raw)
    besthi[~relevant]=np.maximum(besthi[~relevant],initial_enthi[~relevant])
    # Reconstruct powers directly; repeated normalization otherwise widens
    # intervals artificially even for states tending to a separable limit.
    for _ in range(first):state,factor,parameters=bias_step(parameters,factor)
    for k in range(first,K+1):
        concurrence=np.minimum(1,np.maximum(0,iv.up(2*state[1].max(1)-1)))
        # Bell relative entropy <= concurrence^2 (binary entropy series).
        enthi=iv.up(factor[1]*iv.up(concurrence*concurrence))
        still_relevant,_,_=relevant_states(data,aggregation,iv.up(ps[1]*np.maximum(besthi,enthi)),incumbent)
        removed=relevant&~still_relevant
        besthi[removed]=np.maximum(besthi[removed],enthi[removed])
        relevant&=still_relevant
        active=np.flatnonzero(relevant&(enthi>bestlo+tol))
        excluded=np.ones(len(bestlo),dtype=bool);excluded[active]=False
        besthi[excluded]=np.maximum(besthi[excluded],enthi[excluded]);visited.append(len(active))
        if not len(active):break
        lo,hi,_=iv.best_code_bounds((state[0][active],state[1][active]),kernels)
        a,b=iv.mul((lo,hi),(factor[0][active],factor[1][active]));bestlo[active]=np.maximum(bestlo[active],a);besthi[active]=np.maximum(besthi[active],b)
        print('Werner enclosure',float(q),'round',k,'states',len(active),'seconds',round(time.time()-started,1),flush=True)
        state,factor,parameters=bias_step(parameters,factor)
        temp=checkpoint.with_suffix('.tmp.npz')
        np.savez(temp,next_round=k+1,state_lo=state[0],state_hi=state[1],factor_lo=factor[0],factor_hi=factor[1],
                 bestlo=bestlo,besthi=besthi,visited=visited,engine_sha=digest,K=K)
        temp.replace(checkpoint)
    bi,si,fi=aggregation
    # fi is c/m rounded to nearest; exact rational c/m is enclosed by neighbors.
    contrib=iv.mul(iv.mul((bestlo[si],besthi[si]),(ps[0][si],ps[1][si])),iv.point(fi))
    totalslo=np.bincount(bi,weights=contrib[0],minlength=len(data['entries']));totalshi=np.bincount(bi,weights=contrib[1],minlength=len(data['entries']))
    g=iv.gamma(256);totalslo=np.maximum(0,iv.down(totalslo/(1+g)));totalshi=iv.up(totalshi/(1-g))
    results={}
    for M in range(1,5):
        ids=[i for i,e in enumerate(data['entries']) if e['M']<=M]
        results[f'M{M}']=dict(lower=float(max(incumbent[M],totalslo[ids].max())),upper=float(max(incumbent[M],totalshi[ids].max())))
    return dict(Q=str(q),K=K,results=results,active_states=visited,seconds=time.time()-started,
                scope='Complete m<=8, n<=7, pooled B^k k<=K class. Positive-polynomial directed enclosures, analytic log series, relative-entropy/concurrence-squared upper pruning, certified-incumbent instrument screening. IEEE binary64 and standard gamma_n bounds.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bb',nargs='*');parser.add_argument('--werner',nargs='*');args=parser.parse_args();ks=code_kernels()
    if args.werner:
        data,weights,offsets,agg=load()
        for q in args.werner:
            dest=DATA/f'werner_upper_{float(q):.5f}.json'
            if dest.exists():continue
            result=werner_upper(F(q),data,weights,offsets,agg,ks);dest.write_text(json.dumps(result,indent=2));print(result,flush=True)
    if args.bb is not None:
        cache={}
        paths=[DATA/f'bb84_complete_{float(v.split(",")[0]):.5f}_{float(v.split(",")[1]):.5f}.json' for v in args.bb] if args.bb else sorted(DATA.glob('bb84_complete_*.json'))
        for path in paths:
            if path.stem.endswith('_certified'):continue
            src=json.loads(path.read_text());eb=F(str(src['eb']));ep=F(str(src['ep']));sym=eb==ep
            if sym not in cache:cache[sym]=complete_short_entries(sym)
            bb_upper(eb,ep,cache[sym],ks,src)


if __name__=='__main__':main()
