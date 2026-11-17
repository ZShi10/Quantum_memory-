from common import *
from libraries import code_kernels
from rates import best_codes
import json,time,argparse
import numpy as np


def load():
    data=json.loads((DATA/'werner_complete_library.json').read_text());weights={};offsets={};at=0
    for m in range(1,9):
        offsets[m]=at;weights[m]=np.load(DATA/f'werner_states_m{m}.npz')['weights'];at+=len(weights[m])
    block=[];state=[];factor=[]
    for i,e in enumerate(data['entries']):
        for s,c in e['grouped_states']:block.append(i);state.append(offsets[e['m']]+s);factor.append(c/e['m'])
    return data,weights,offsets,(np.array(block),np.array(state),np.array(factor))


def optimize(q,data,weights,offsets,aggregation,kernels,K=16,tolerance=1e-12):
    raw=np.concatenate([w@np.array([(1-1.5*q)**(m-j)*(q/2)**j for j in range(m+1)]) for m,w in weights.items()])
    ps=raw.sum(1);p=raw/ps[:,None];best=np.zeros(len(p));chosen=np.full(len(p),-1,dtype=np.int32);rounds=np.zeros(len(p),dtype=np.int16)
    factor=np.ones(len(p));state=p.copy();visited=[]
    for k in range(K+1):
        # Every candidate's hashing yield is a virtual entanglement yield.
        # Convexity of entanglement of formation gives E_F<=2*pmax-1 for
        # a Bell-diagonal pair, by mixing a separable boundary state with
        # its largest Bell projector. This bounds all later LOCC stages.
        upper=factor*np.maximum(0.,2*state.max(1)-1)+3e-14
        active=np.flatnonzero(upper>best+tolerance)
        visited.append(len(active))
        if not len(active):break
        values=best_codes(state[active],kernels)
        rates=values['rate']*factor[active];take=rates>best[active]
        ids=active[take];best[ids]=rates[take];chosen[ids]=values['code'][take];rounds[ids]=k
        a,x,y,z=np.moveaxis(state,-1,0);out=np.stack((a*a+z*z,x*x+y*y,2*x*y,2*a*z),-1)
        mass=out.sum(1);factor*=mass/2;state=out/mass[:,None]
    bi,si,fi=aggregation;totals=np.bincount(bi,weights=fi*ps[si]*best[si],minlength=len(data['entries']))
    results={}
    for M in range(1,5):
        ids=[i for i,e in enumerate(data['entries']) if e['M']<=M];at=max(ids,key=lambda i:totals[i]);e=data['entries'][at]
        branches=[]
        for b in e['branches']:
            j=offsets[e['m']]+b['state'];ci=chosen[j]
            if ci<0:continue
            branches.append({**b,'binary_rounds':int(rounds[j]),'code':kernels[ci].entry['id'],
                             'contribution':float(b['factor']*ps[j]/e['m']*best[j])})
        results[f'M{M}']=dict(Q=q,rate=float(totals[at]),ED=e['id'],m=e['m'],M=e['M'],branches=branches)
    return dict(Q=q,results=results,active_states_per_AD_round=visited)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--q',nargs='*',type=float);parser.add_argument('--rounds',type=int,default=16);args=parser.parse_args()
    data,weights,offsets,agg=load();kernels=code_kernels();started=time.time()
    qs=args.q or [.10,.15,.18,.20,.21,.22,.23,.24,.25,.26,.27,.275,.28,.285,.29,.295,.30,.305,.31,.315,.32]
    path=DATA/f'werner_complete_k{args.rounds}.json';saved=json.loads(path.read_text()) if path.exists() else dict(results=[])
    done={r['Q'] for r in saved['results']}
    for q in qs:
        if q in done:continue
        r=optimize(q,data,weights,offsets,agg,kernels,args.rounds);saved['results'].append(r)
        path.write_text(json.dumps(saved,separators=(',',':')))
        print('Complete Werner optimization',q,[(k,v['rate'],v['ED']) for k,v in r['results'].items()],
              'active',r['active_states_per_AD_round'],'seconds',round(time.time()-started,1),flush=True)
    saved.update(scope='Complete fixed Clifford one-bit terminal instruments m<=8; all ED branches; all nonzero binary linear codes n<=7 via direct-sum domination; pooled B prefixes k=0..K. Entanglement-yield pruning has conditional tolerance 1e-12 plus floating guards. This is finite-family numerical optimization; outward-rounded global error certification is a separate step.',
                 K=args.rounds,m_max=8,n_max=7,seconds_this_run=time.time()-started)
    saved['results'].sort(key=lambda r:r['Q']);path.write_text(json.dumps(saved,separators=(',',':')))


if __name__=='__main__':main()
