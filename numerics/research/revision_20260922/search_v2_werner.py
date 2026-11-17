"""Frozen four-decoder recurrence family, same AD menu for every candidate."""
from core import *
from recurrence_v2 import generate,PRIMITIVES
from known_search import optimize
from recurrence_equivalence import representatives
import argparse,time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--q',nargs='+',type=float)
    parser.add_argument('--permissions',choices=['real','full'],default='real')
    parser.add_argument('--dense',action='store_true');a=parser.parse_args();kernels=load_codes()
    path=DATA/f'werner_v2_{a.permissions}.json';output=json.loads(path.read_text()) if path.exists() else dict(results=[])
    done={r['Q'] for r in output['results']}
    for q in np.linspace(0,1/3,500) if a.dense else a.q or [.1,.2,.25,.28,.3,.32]:
        if float(q) in done:continue
        start=time.time();p=np.array([1-1.5*q,q/2,q/2,q/2]);states,y,M,records=generate(p,7,a.permissions)
        rep=representatives(records);ids=np.flatnonzero(rep==np.arange(len(rep)))
        compact={int(original):i for i,original in enumerate(ids)};seeds=[];seen=set()
        for old in sorted(output['results'],key=lambda v:abs(v['Q']-q))[:3]:
            for previous in old['results'].values():
                index=compact[int(rep[previous['source_index']])];ad=previous['AD']
                identity=(index,json.dumps(ad,sort_keys=True))
                if identity not in seen:seeds.append(dict(index=index,AD=ad));seen.add(identity)
        r=optimize(states[ids],y[ids],M[ids],[records[i] for i in ids],kernels,with_watanabe=True,seeds=seeds)
        for result in r['results'].values():result['source_index']=int(ids[result['source_index']])
        r.update(Q=float(q),candidate_count=len(records),equivalence_representatives=len(ids),seconds=time.time()-start)
        output['results'].append(r);output['generator']=PRIMITIVES
        output['scope']='Every word of depth<=7 in four explicit real two-input decoders (ZZ,XX,YY,XZ); constant-decoder raw pumps r<=160; final X/Z (real) or X/Y/Z (full); n<=7 all codes and repetition/SPC n<=128; Du three net modes plus Watanabe terminal; B<=16. Explicit recursive memory upper bounds; floating candidate search.'
        atomic_json(path,output);print('V2 Werner',a.permissions,q,{k:v['rate'] for k,v in r['results'].items()},round(time.time()-start,1),flush=True)


if __name__=='__main__':main()
