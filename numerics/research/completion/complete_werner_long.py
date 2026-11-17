"""No-ED-shortlist endpoint extension of the complete Werner quantum class.

All repetition and SPC lengths 1..128, not a sparse selection of lengths.
Short linear codes n<=7 are also evaluated. Each branch is processed
separately after k<=16 pooled B rounds. Pruning uses explicit entanglement
and full-physical-entropy upper bounds, with reported numerical tolerances.
"""
from common import *
import argparse,json,time
import numpy as np
from complete_werner_search import load
from libraries import code_kernels
from rates import best_codes,entropy,xlog2x
from streaming import rate as repetition_rate
from spc_long import spc_rate
from spc_batched import rates as spc_rates
from spc_dynamic import all_rates as dynamic_spc_rates
from screening import incumbents,relevant_states


def spc_upper_array(p,ns):
    d=1-2*(p[:,1]+p[:,2]);h=entropy(p)
    hp=(-xlog2x(p)*np.array([1,-1,-1,1])).sum(1)
    result=np.zeros((len(p),len(ns)))
    overlap0=2*np.sqrt(p[:,0]*p[:,3]);overlap1=2*np.sqrt(p[:,1]*p[:,2])
    for sign in (1,-1):
        den=1+sign*d[:,None]**ns[None,:]
        safe=np.maximum(den,1e-300)
        conditional=(h[:,None]+sign*d[:,None]**(ns[None,:]-1)*hp[:,None])/safe+np.log2(safe/2)/ns[None,:]
        overlap=((overlap0+overlap1)[:,None]**ns[None,:]+sign*(overlap0-overlap1)[:,None]**ns[None,:])/2
        lost_entropy=np.minimum(den/2,np.maximum(0,overlap))
        ub=den/2*(1-1/ns[None,:]-conditional)+lost_entropy/ns[None,:]
        result+=np.where(den>1e-16,np.maximum(0.,ub),0.)
    return result+2e-13


def optimize(q,data,weights,offsets,aggregation,kernels,K=16):
    raw=np.concatenate([w@np.array([(1-1.5*q)**(m-j)*(q/2)**j for j in range(m+1)]) for m,w in weights.items()])
    ps=raw.sum(1);state=raw/ps[:,None];best=np.zeros(len(ps));chosen=np.full(len(ps),-1,dtype=np.int32)
    relevant,_,kept=relevant_states(data,aggregation,ps*np.maximum(0,2*state.max(1)-1)**2,incumbents(q),padding=1e-12)
    print('Instrument bound screen',q,'keep',int(kept.sum()),'of',len(kept),'states',int(relevant.sum()),flush=True)
    rounds=np.zeros(len(ps),dtype=np.int16);lengths=rounds.copy();kind=np.zeros(len(ps),dtype=np.int8);factor=np.ones(len(ps))
    counts=[];spc_evals=0;started=time.time();ns=np.arange(8,129);first=0
    work=DATA/f'long_work_{q:.5f}.npz'
    if work.exists():
        w=np.load(work);first=int(w['next_round']);state=w['state'];factor=w['factor'];best=w['best'];chosen=w['chosen'];rounds=w['rounds'];lengths=w['lengths'];kind=w['kind']
        counts=json.loads(str(w['counts']));spc_evals=int(w['spc_evals'])
        w.close()
        print('Resume long Werner',q,'B',first,flush=True)
    for k in range(first,K+1):
        upper=factor*np.maximum(0,2*state.max(1)-1)**2+3e-14
        still_relevant,_,_=relevant_states(data,aggregation,ps*np.maximum(best+1e-12,upper),incumbents(q),padding=1e-12)
        relevant&=still_relevant
        ids=np.flatnonzero(relevant&(upper>best+1e-12))
        if not len(ids):break
        local=state[ids].copy();bitflip=local[:,1]+local[:,2]>.5
        local[bitflip]=local[bitflip][:,[1,0,3,2]]
        r=best_codes(local,kernels);rates=r['rate']*factor[ids];take=rates>best[ids]
        at=ids[take];best[at]=rates[take];chosen[at]=r['code'][take];rounds[at]=k;kind[at]=0
        nrep=np.arange(1,129);a=local[:,0]+local[:,3];b=local[:,1]+local[:,2]
        ph0=local[:,3]/a;ph1=np.divide(local[:,2],b,out=np.zeros_like(b),where=b>0)
        eps=np.broadcast_to((a-b)[:,None],(len(ids),len(nrep))).copy()
        rep=repetition_rate(eps,ph0[:,None],ph1[:,None],0,nrep[None,:])*factor[ids,None]
        nj=rep.argmax(1);rr=rep[np.arange(len(ids)),nj];take=rr>best[ids]+1e-15
        at=ids[take];best[at]=rr[take];rounds[at]=k;lengths[at]=nrep[nj[take]];kind[at]=1
        ub=spc_upper_array(local,ns)*factor[ids,None]
        needed=ub>best[ids,None]+2e-13
        this_round=int(needed.sum());spc_evals+=this_round
        values=dynamic_spc_rates(local,ns,needed)*factor[ids,None]
        nc=values.argmax(1);r=values[np.arange(len(ids)),nc];take=r>best[ids]+1e-15
        at=ids[take];best[at]=r[take];rounds[at]=k;lengths[at]=ns[nc[take]];kind[at]=2
        counts.append(dict(k=k,active=len(ids),spc_evaluations=this_round))
        print('Long complete Werner',q,'B',k,'states',len(ids),'SPC',this_round,'seconds',round(time.time()-started,1),flush=True)
        a,x,y,z=np.moveaxis(state,-1,0);out=np.stack((a*a+z*z,x*x+y*y,2*x*y,2*a*z),-1)
        mass=out.sum(1);factor*=mass/2;state=out/mass[:,None]
        temp=work.with_suffix('.tmp.npz');np.savez(temp,next_round=k+1,state=state,factor=factor,best=best,chosen=chosen,
                    rounds=rounds,lengths=lengths,kind=kind,counts=json.dumps(counts),spc_evals=spc_evals);temp.replace(work)
    bi,si,fi=aggregation;totals=np.bincount(bi,weights=fi*ps[si]*best[si],minlength=len(data['entries']));results={}
    for M in range(1,5):
        ids=[i for i,e in enumerate(data['entries']) if e['M']<=M];at=max(ids,key=lambda i:totals[i]);e=data['entries'][at];branches=[]
        for b in e['branches']:
            j=offsets[e['m']]+b['state']
            if best[j]<=0:continue
            rec={**b,'binary_rounds':int(rounds[j]),'canonicalize_bit':True,'contribution':float(b['factor']*ps[j]/e['m']*best[j])}
            if kind[j]==0:rec.update(kind='linear',code=kernels[chosen[j]].entry['id'])
            else:rec.update(kind='repetition' if kind[j]==1 else 'SPC',n=int(lengths[j]))
            branches.append(rec)
        results[f'M{M}']=dict(Q=q,rate=float(totals[at]),ED=e['id'],m=e['m'],M=e['M'],branches=branches)
    return dict(Q=q,results=results,round_counts=counts,SPC_evaluations=spc_evals,seconds=time.time()-started)


def main():
    p=argparse.ArgumentParser();p.add_argument('--q',nargs='*',type=float);p.add_argument('--part',default='');a=p.parse_args()
    data,weights,offsets,agg=load();ks=code_kernels();dest=DATA/('werner_complete_long'+('_'+a.part if a.part else '')+'.json')
    saved=json.loads(dest.read_text()) if dest.exists() else dict(results=[])
    done={r['Q'] for r in saved['results']}
    for q in a.q or [.20,.23,.26,.27,.28,.285,.29,.295,.30,.305,.31,.315,.32]:
        if q in done:continue
        r=optimize(q,data,weights,offsets,agg,ks);saved['results'].append(r)
        saved.update(scope='Complete m<=8 fixed terminal class; all ED branches; all codes n<=7 plus all repetition/SPC lengths <=128; k<=16 pooled B rounds. No ED shortlist. Numerical upper-bound pruning with conditional tolerance 1e-12 and SPC tolerance 2e-13; independent rigorous finite upper certificates currently cover the short-code subfamily.')
        dest.write_text(json.dumps(saved,separators=(',',':')))
        print('Long final',q,[(k,v['rate'],v['ED']) for k,v in r['results'].items()],flush=True)


if __name__=='__main__':main()
