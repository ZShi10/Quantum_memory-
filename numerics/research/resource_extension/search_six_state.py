from common import *
import json,time,argparse
import numpy as np
from algebra import compositions
from rates import best_codes,entropy,xlog2x
from kernels import load
from search_werner import werner_blocks
from spc_long import spc_rate
from streaming import rate as repetition_rate


def library():
    entries=werner_blocks(json.loads((OLD/'data/blocks.json').read_text())['entries'])
    for e in entries:e['source']='old_complete_short';e['acceptance']='all ED branches'
    seen={}
    for e in json.loads((DATA/'graphs.json').read_text())['entries']:
        counts=np.asarray(e['characters']);m=e['m'];out=np.zeros((4,m+1),dtype=int)
        for j,typ in enumerate(compositions(m)):out[:,m-typ[0]]+=counts[:,j]
        key=(m,out.tobytes())
        if key not in seen or e['M']<seen[key]['M']:
            seen[key]={**e,'character_weights':out.tolist(),'source':'extended_graph_zero'}
    entries+=list(seen.values())
    return entries


def raw_input(e,q):
    m=e['m']
    if 'werner_weights' in e:
        powers=np.array([(1-1.5*q)**(m-w)*(q/2)**w for w in range(m+1)])
        return np.asarray(e['werner_weights'])@powers
    powers=np.array([(1-2*q)**w for w in range(m+1)])
    raw=np.asarray(e['character_weights'])@powers/(2**(m+1))
    assert raw.min()>-1e-12
    return np.maximum(raw,0)[None,:]


def b_step(p):
    a,x,y,z=np.moveaxis(p,-1,0);out=np.stack((a*a+z*z,x*x+y*y,2*x*y,2*a*z),-1)
    s=out.sum(-1);return out/s[...,None],s/2


def spc_upper(p,n):
    b=p[1]+p[2];d=1-2*b;h=float(entropy(p));hp=float(-xlog2x(p)*np.array([1,-1,-1,1])@np.ones(4))
    result=0
    for sign in (1,-1):
        den=1+sign*d**n
        if den<=1e-16:continue
        conditional=(h+sign*d**(n-1)*hp)/den+np.log2(den/2)/n
        result+=max(0.,den/2*(1-conditional))
    return result+1e-13


def long_branch(p,weight,initial,rounds=24):
    best=initial.copy();state=p.copy();factor=1.;evaluated=0
    lengths=list(range(2,65))+[80,96,112,128]
    for k in range(rounds+1):
        a=state[0]+state[3];b=state[1]+state[2]
        if a<b:state=state[[1,0,3,2]];a,b=b,a
        eps=a-b;phi0=state[3]/a;phi1=state[2]/b if b else 0
        ns=np.arange(1,129)
        rr=np.asarray(repetition_rate(np.full(len(ns),eps),phi0,phi1,0,ns))*factor*weight
        idx=int(rr.argmax())
        if rr[idx]>best['contribution']+1e-15:
            best=dict(contribution=float(rr[idx]),binary_rounds=k,kind='repetition',n=int(ns[idx]),canonicalize_bit=True)
        for n in lengths:
            if weight*factor*spc_upper(state,n)<=best['contribution']+2e-14:continue
            r=weight*factor*spc_rate(state,n);evaluated+=1
            if r>best['contribution']+1e-15:best=dict(contribution=float(r),binary_rounds=k,kind='SPC',n=n,canonicalize_bit=True)
        state,yield_=b_step(state);factor*=float(yield_)
    return best,evaluated


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--q',nargs='*',type=float);parser.add_argument('--skip-long',action='store_true');args=parser.parse_args()
    qs=args.q or [.10,.15,.18,.20,.21,.22,.23,.24,.25,.26,.27,.275,.28,.285,.29,.295,.30,.305,.31,.315,.32]
    entries=library();ks=load();start=time.time();output=[]
    print('Six-state candidates',len(entries),'codes',len(ks),flush=True)
    for q in qs:
        raws=[raw_input(e,q) for e in entries];starts=np.cumsum([0]+[len(r) for r in raws]);raw=np.concatenate(raws)
        ps=raw.sum(1);p=np.divide(raw,ps[:,None],out=np.full_like(raw,.25),where=ps[:,None]>0)
        ms=np.concatenate([np.full(len(r),e['m']) for r,e in zip(raws,entries)]);weight=ps/ms
        rates=np.zeros(len(p));chosen=np.full(len(p),-1,dtype=int);rounds=np.zeros(len(p),dtype=int)
        state=p.copy();factor=np.ones(len(p))
        for k in range(17):
            values=best_codes(state,ks);r=values['rate']*factor*weight;take=r>rates+1e-15
            rates[take]=r[take];chosen[take]=values['code'][take];rounds[take]=k
            state,yield_=b_step(state);factor*=yield_
        totals=np.add.reduceat(rates,starts[:-1]);eligible={M:np.array([i for i,e in enumerate(entries) if e['M']<=M]) for M in range(1,9)}
        shortlist=set()
        for ids in eligible.values():shortlist.update(map(int,ids[np.argsort(totals[ids])[-3:]]))
        protocols={};long_evaluations=0
        for bi in shortlist:
            branches=[]
            for j in range(starts[bi],starts[bi+1]):
                ci=chosen[j];record=dict(contribution=float(rates[j]),binary_rounds=int(rounds[j]),
                                       kind='linear',code=ks[ci].entry['id'] if ci>=0 else None)
                if not args.skip_long and p[j].max()>.5+1e-12:
                    record,num=long_branch(p[j],weight[j],record);long_evaluations+=num
                record['syndrome']=int(j-starts[bi]);branches.append(record)
            protocols[bi]=dict(Q=q,ED=entries[bi]['id'],m=entries[bi]['m'],M=entries[bi]['M'],
                               rate=sum(v['contribution'] for v in branches),branches=branches,
                               short_code_rate=float(totals[bi]))
        result={}
        for M in range(1,9):
            choices=[p for i,p in protocols.items() if entries[i]['M']<=M]
            result[f'M{M}']=max(choices,key=lambda r:r['rate'])
        output.append(dict(Q=q,results=result,long_SPC_evaluations=long_evaluations))
        print('six-state',q,[(M,result[f'M{M}']['rate'],result[f'M{M}']['ED']) for M in (1,2,4,8)],'seconds',round(time.time()-start,1),flush=True)
        selected={v['ED'] for item in output for v in item['results'].values()}
        saved=dict(scope='Complete short blocks plus selected m<=14, M<=8 zero-syndrome graph family. All code orbits n<=7 and selected n=8; B prefixes <=16. Long endpoint refinement of top three EDs per memory budget: prefixes <=24, repetition n<=128, SPC n=2..64 plus 80,96,112,128. Achievable search, not a global optimum.',
                   results=output,protocol_library={e['id']:e for e in entries if e['id'] in selected},seconds=time.time()-start)
        (DATA/'six_state.json').write_text(json.dumps(saved,separators=(',',':')))


if __name__=='__main__':main()
