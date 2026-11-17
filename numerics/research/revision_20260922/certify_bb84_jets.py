"""First-derivative interval certificates for recurrence/Watanabe policies."""
from core import *
from fractions import Fraction as F
import interval_jets as jets
import argparse,heapq,time


def certify(b,p,record,atol=1e-11,rtol=2e-6,max_cells=100000):
    heap=[];counter=0;incumbent=np.inf;argmin=None
    def add(intervals):
        nonlocal counter,incumbent,argmin
        box=jets.policy(b,p,intervals,record)
        mids=[((a+c)/2,(a+c)/2) for a,c in intervals]
        middle=jets.policy(b,p,mids,record);lower=jets.mean_value_lower(box,middle,intervals)
        for (a,c),lo,hi in zip(intervals,lower,middle.v[1]):
            heapq.heappush(heap,(float(lo),counter,a,c));counter+=1
            if hi<incumbent:incumbent=float(hi);argmin=float((a+c)/2)
    endpoints=jets.policy(b,p,[(F(0),F(0)),(F(1),F(1))],record).v[1]
    incumbent=float(endpoints.min());argmin=float(endpoints.argmin())
    add([(F(0),F(1))])
    while counter<max_cells and incumbent-heap[0][0]>atol+rtol*abs(incumbent):
        intervals=[]
        for _ in range(min(32,len(heap))):
            lo,_,a,c=heapq.heappop(heap);mid=(a+c)/2;intervals.extend(((a,mid),(mid,c)))
        add(intervals)
    return dict(lower=float(heap[0][0]),upper=incumbent,gap=float(incumbent-heap[0][0]),argmin=argmin,
                cells=counter,converged=incumbent-heap[0][0]<=atol+rtol*abs(incumbent),atol=atol,rtol=rtol,
                arithmetic='Outward interval values and first derivatives, analytic log remainder, midpoint mean-value bounds, whole compact interval.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--q',nargs='*',type=float)
    parser.add_argument('--budgets',nargs='+',type=int,default=list(range(1,9)));a=parser.parse_args()
    src=json.loads((DATA/a.source).read_text());path=DATA/(Path(a.source).stem+'_jet_certified.json')
    out=json.loads(path.read_text()) if path.exists() else dict(results={});cache={}
    for row in src['results']:
        if a.q and row['b'] not in a.q:continue
        for M in a.budgets:
            key=f'{row["b"]:.8g}_{row["p"]:.8g}_M{M}'
            if key in out['results'] and out['results'][key]['certificate'].get('converged',True):continue
            record=row['results'][f'M{M}'];start=time.time()
            if record['AD'] is None:cert=dict(lower=0.,upper=0.,status='no_positive_candidate')
            else:
                identity=(row['b'],row['p'],json.dumps(record['ED'],sort_keys=True),record['AD']['classical_B_rounds'])
                if identity not in cache:cache[identity]=certify(row['b'],row['p'],record)
                cert=cache[identity]
            out['results'][key]=dict(b=row['b'],p=row['p'],budget=M,**record,certificate=cert)
            atomic_json(path,out);print('BB jet certified',key,cert,'seconds',round(time.time()-start,1),flush=True)


if __name__=='__main__':main()
