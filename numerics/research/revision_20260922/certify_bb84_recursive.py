"""Continuous-correlation guarantees for fixed recurrence/Watanabe policies."""
from core import *
from fractions import Fraction as F
from recurrence_v2 import MAPS
from recursive_library import PRE
from interval_modes import signed_mul
import stable_intervals as iv
import argparse,heapq,time


def clip_prob(a):return np.maximum(0,a[0]),np.minimum(1,np.maximum(0,a[1]))


def step(p,q,primitive):
    lo=np.zeros_like(p[0]);hi=np.zeros_like(p[1])
    for i,j,k in MAPS[primitive]:
        v=iv.mul((p[0][:,i],p[1][:,i]),(q[0][:,j],q[1][:,j]))
        lo[:,k],hi[:,k]=iv.add((lo[:,k],hi[:,k]),v)
    raw=(np.maximum(0,lo),np.maximum(0,hi));mass=clip_prob(iv.positive_sum(raw));good=mass[0]>0
    safe=(np.where(good,mass[0],1),np.where(good,mass[1],1))
    out=clip_prob(iv.div(raw,(safe[0][:,None],safe[1][:,None])))
    out[0][~good]=0;out[1][~good]=1
    return out,mass


def source(b,p,intervals):
    b,p=F(str(b)),F(str(p));z=[(min(b,p)*F(a),min(b,p)*F(c)) for a,c in intervals]
    rows=[[(1-b-p+a,1-b-p+c),(b-c,b-a),(a,c),(p-c,p-a)] for a,c in z]
    return np.array([[iv.rational(v[0])[0] for v in row] for row in rows]),np.array([[iv.rational(v[1])[1] for v in row] for row in rows])


def prepare(b,p,intervals,record,rounds):
    initial=source(b,p,intervals);state=initial;y=(np.ones(len(intervals)),np.ones(len(intervals)))
    family=record['family']
    if family=='real_decoder_recurrence':
        for primitive in record['path']:
            state,s=step(state,state,primitive);y=iv.scale(iv.mul(y,s),F(1,2))
    elif family=='real_decoder_pumping':
        for _ in range(record['r']-1):
            state,s=step(state,initial,record['primitive'])
            y=iv.div(iv.mul(y,s),iv.add((np.ones(len(intervals)),np.ones(len(intervals))),y))
    elif family=='homogeneous_recurrence':
        for axis in record['path']:
            state=tuple(v[:,list(PRE[axis])] for v in state)
            state,s=step(state,state,'Z');y=iv.scale(iv.mul(y,s),F(1,2))
    else:
        initial=tuple(v[:,list(PRE[record['preparation_axis']])] for v in initial);state=initial
        for _ in range(record['r']-1):
            state,s=step(state,initial,'Z')
            y=iv.div(iv.mul(y,s),iv.add((np.ones(len(intervals)),np.ones(len(intervals))),y))
    state=tuple(v[:,list(PRE[record['final_axis']])] for v in state)
    for _ in range(rounds):state,s=step(state,state,'Z');y=iv.scale(iv.mul(y,s),F(1,2))
    return state,y


def h(a):
    out=iv.entropy_interval(clip_prob(a));return np.maximum(0,out[0]),np.maximum(0,out[1])


def watanabe_bound(p):
    one=(np.ones(len(p[0])),np.ones(len(p[0])))
    entropy=iv.positive_sum(h(p))
    def col(i):return p[0][:,i],p[1][:,i]
    a=clip_prob(iv.add(col(0),col(3)));b=clip_prob(iv.add(col(1),col(2)))
    v=iv.mul(a,b);v=(v[0],np.minimum(.25,v[1]))
    u=iv.add(iv.mul(col(0),col(1)),iv.mul(col(3),col(2)))
    u=(u[0],np.minimum(u[1],v[1]))
    rest=iv.add(iv.mul(col(0),col(2)),iv.mul(col(3),col(1)));rest=clip_prob(rest)
    extra=iv.sub(iv.add(h(u),h(rest)),h(v))
    first=iv.add(iv.sub(one,entropy),extra)
    out,s=step(p,p,'Z')
    second=signed_mul(iv.scale(s,F(1,2)),iv.sub(one,iv.positive_sum(h(out))))
    return np.maximum.reduce((np.zeros(len(p[0])),first[0],second[0])),np.maximum.reduce((np.zeros(len(p[0])),first[1],second[1]))


def policy_bound(b,p,intervals,record):
    state,y=prepare(b,p,intervals,record['ED'],record['AD']['classical_B_rounds'])
    return iv.mul(y,watanabe_bound(state))


def certify(b,p,record,atol=1e-10,rtol=1e-5,max_cells=250000):
    heap=[];counter=0;incumbent=np.inf;argmin=None
    def add(intervals):
        nonlocal counter,incumbent,argmin
        low,high=policy_bound(b,p,intervals,record)
        mids=[((a+c)/2,(a+c)/2) for a,c in intervals]
        _,mh=policy_bound(b,p,mids,record)
        for (a,c),lo,hi in zip(intervals,low,mh):
            heapq.heappush(heap,(float(lo),counter,a,c));counter+=1
            if hi<incumbent:incumbent=float(hi);argmin=float((a+c)/2)
    add([(F(0),F(1))])
    while counter<max_cells and incumbent-heap[0][0]>atol+rtol*abs(incumbent):
        children=[]
        for _ in range(min(64,len(heap))):
            lo,_,a,c=heapq.heappop(heap);mid=(a+c)/2
            children.extend(((a,mid),(mid,c)))
        add(children)
    return dict(lower=float(heap[0][0]),upper=incumbent,gap=float(incumbent-heap[0][0]),
                argmin=argmin,cells=counter,converged=incumbent-heap[0][0]<=atol+rtol*abs(incumbent),
                atol=atol,rtol=rtol,
                arithmetic='Rational BB source; outward vector probability propagation and entropy; analytic log-series remainder; whole [0,1] subdivision.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--q',nargs='*',type=float)
    parser.add_argument('--budgets',nargs='+',type=int,default=[1,2,4,6,8]);a=parser.parse_args()
    src=json.loads((DATA/a.source).read_text());path=DATA/(Path(a.source).stem+'_certified.json')
    out=json.loads(path.read_text()) if path.exists() else dict(results={});cache={}
    for row in src['results']:
        if a.q and row['b'] not in a.q:continue
        for M in a.budgets:
            key=f'{row["b"]:.8g}_{row["p"]:.8g}_M{M}'
            if key in out['results']:continue
            record=row['results'][f'M{M}'];start=time.time()
            if record['AD'] is None:cert=dict(lower=0.,upper=0.,status='no_positive_candidate')
            else:
                identity=(row['b'],row['p'],json.dumps(record['ED'],sort_keys=True),record['AD']['classical_B_rounds'])
                if identity not in cache:cache[identity]=certify(row['b'],row['p'],record)
                cert=cache[identity]
            out['results'][key]=dict(b=row['b'],p=row['p'],budget=M,**record,certificate=cert)
            atomic_json(path,out);print('BB recursive certified',key,cert,'seconds',round(time.time()-start,1),flush=True)


if __name__=='__main__':main()
