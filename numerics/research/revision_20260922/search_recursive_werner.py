"""Resumable Werner recurrence + raw pumping + AD mode optimization."""
from core import *
from recursive_library import generate,greedy,pumping_outputs
from known_search import optimize
import argparse,time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--q',nargs='+',type=float)
    parser.add_argument('--permissions',choices=['real','full'],default='real')
    parser.add_argument('--depth',type=int,default=7)
    parser.add_argument('--rmax',type=int,default=160)
    parser.add_argument('--long-max',type=int,default=128)
    parser.add_argument('--rounds',type=int,default=16)
    parser.add_argument('--dense',action='store_true')
    args=parser.parse_args();kernels=load_codes()
    dest=DATA/f'recursive_werner_{args.permissions}_d{args.depth}_n{args.long_max}.json'
    output=json.loads(dest.read_text()) if dest.exists() else dict(results=[],configuration=vars(args))
    for key in ('permissions','depth','rmax','long_max','rounds'):
        assert output['configuration'][key]==getattr(args,key),('Incompatible cached scope',key)
    done={r['Q'] for r in output['results']}
    qs=np.linspace(0,1/3,500) if args.dense else args.q or [.05,.1,.15,.2,.25,.28,.3,.32]
    for q in qs:
        if float(q) in done:continue
        start=time.time();raw=np.array([1-1.5*q,q/2,q/2,q/2])
        p,y,M,meta=generate(raw,args.depth,args.permissions)
        p2,y2,M2,meta2=pumping_outputs(raw,args.rmax,args.permissions)
        p=np.concatenate((p,p2));y=np.r_[y,y2];M=np.r_[M,M2];meta+=meta2
        result=optimize(p,y,M,meta,kernels,K=args.rounds,long_max=args.long_max)
        result.update(Q=float(q),candidate_count=len(p),seconds=time.time()-start,greedy=greedy(raw,args.depth))
        output['results'].append(result);output['results'].sort(key=lambda r:r['Q'])
        atomic_json(dest,output)
        print(json.dumps(dict(Q=q,seconds=result['seconds'],rates={k:v['rate'] for k,v in result['results'].items()},counts=result['round_counts'])),flush=True)


if __name__=='__main__':main()
