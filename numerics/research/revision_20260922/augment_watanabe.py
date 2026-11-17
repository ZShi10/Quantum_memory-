"""Add the strong hashed two-way terminal without rerunning Du's code search."""
from core import *
from recursive_library import generate,pumping_outputs,greedy,PRE,terminal_schedule
from literature_rates import watanabe,vollbrecht_verstraete
import argparse,time


def optimize(p,y,M,records,rounds=16):
    best=np.zeros(len(p));chosen=np.zeros(len(p),int)
    for k in range(rounds+1):
        values=y*watanabe(p);take=values>best
        best[take]=values[take];chosen[take]=k
        p,s=b_step(p);y=y*s/2
    result={}
    for budget in range(1,int(M.max())+1):
        ids=np.flatnonzero(M<=budget);i=int(ids[np.argmax(best[ids])])
        result[f'M{budget}']=dict(rate=float(best[i]),ED=records[i],
                                  AD=dict(kind='Watanabe_2007',n=2,mode='no_otp_hashed',classical_B_rounds=int(chosen[i])))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--permissions',choices=['real','full'],default='real')
    parser.add_argument('--dense',action='store_true');parser.add_argument('--q',nargs='+',type=float)
    a=parser.parse_args();path=DATA/f'watanabe_augmentation_{a.permissions}.json'
    saved=json.loads(path.read_text()) if path.exists() else dict(results=[])
    done={r['Q'] for r in saved['results']}
    qs=np.linspace(0,1/3,500) if a.dense else a.q or [.1,.2,.25,.28,.3,.32]
    for q in qs:
        if float(q) in done:continue
        start=time.time();raw=np.array([1-1.5*q,q/2,q/2,q/2])
        p,y,M,meta=generate(raw,7,a.permissions)
        pp,yy,mm,dd=pumping_outputs(raw,160,a.permissions)
        result=optimize(np.r_[p,pp],np.r_[y,yy],np.r_[M,mm],meta+dd)
        greedy_rows=[]
        for r in greedy(raw,7):
            candidates=[]
            for axis in ('Z','X','Y'):
                state=np.array(r['p'])[list(PRE[axis])]
                candidates.append(dict(axis=axis,**terminal_schedule(r['path'],axis),
                                       Watanabe_rate=float(r['yield_per_raw_pair']*watanabe(state)),
                                       VV_rate=float(r['yield_per_raw_pair']*vollbrecht_verstraete(state)),
                                       hashing_rate=float(r['yield_per_raw_pair']*max(0,1-entropy(state)))))
            greedy_rows.append(dict(depth=r['depth'],path=r['path'],choices=candidates))
        saved['results'].append(dict(Q=float(q),results=result,greedy=greedy_rows,
                                     direct_Watanabe_rate=float(watanabe(raw)),seconds=time.time()-start))
        saved['scope']='All registered homogeneous recurrences d<=7 and raw pumps r<=160; B<=16; Watanabe Eq.(4) terminal. Greedy+VV rows are a bounded-depth converted classical terminal comparison, not the full AEPP*(4) combined protocol.'
        atomic_json(path,saved)
        print('Watanabe',a.permissions,q,round(time.time()-start,2),{k:r['rate'] for k,r in result.items()},flush=True)


if __name__=='__main__':main()
