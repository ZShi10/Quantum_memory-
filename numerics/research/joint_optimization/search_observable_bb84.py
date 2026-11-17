"""Fixed ED with AD chosen from observable output statistics."""
from pathlib import Path
import heapq,json,time,argparse
import numpy as np
from algebra import span
from rates import best_codes
from bernstein import evaluate,split,GainCurve
from search_bb84 import load_libraries,anchor_upper_bounds

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data'

def observable(entry):
    m=entry['m'];mask=(1<<m)-1
    for p in span(entry['checks']+[entry['logical_Z']]):
        x=p&mask;z=p>>m
        if x&z:return p
    return None

def certify(raw,m,kernels,atol=3e-9,max_cells=15000):
    heap=[];counter=0;incumbent=float('inf');argmin=None;best_example=None
    def add(a,b,branchraw):
        nonlocal counter,incumbent,argmin,best_example
        value=evaluate(branchraw,.5);ps=value.sum(1)
        p=np.divide(value,ps[:,None],out=np.full_like(value,.25),where=ps[:,None]>0)
        best=best_codes(p,kernels);sample=float(ps@best['rate']/m)
        lower=0.;policy=[]
        for s,ci in enumerate(best['code']):
            if ci<0:continue
            kernel=kernels[int(ci)];probs,entropy=kernel.components(p[s])
            keep=(kernel.k*probs-entropy)>0
            curve=GainCurve(branchraw[s],kernel,keep,scale=1/m)
            branch_lower=curve.interval()[0]-2e-12
            if branch_lower<=0:continue
            lower+=branch_lower
            policy.append(dict(syndrome=s,code=kernel.entry['id'],keep_classes=np.flatnonzero(keep).tolist()))
        if sample<incumbent:incumbent,argmin,best_example=sample,(a+b)/2,policy
        heapq.heappush(heap,(lower,counter,a,b,branchraw,policy));counter+=1
    add(0.,1.,raw)
    # Include boundary values in the upper bound on a continuous minimum.
    for t in (0.,1.):
        value=evaluate(raw,t);ps=value.sum(1)
        p=np.divide(value,ps[:,None],out=np.full_like(value,.25),where=ps[:,None]>0)
        sample=float(ps@best_codes(p,kernels)['rate']/m)
        if sample<incumbent:incumbent,argmin=sample,t
    while counter<max_cells and incumbent-heap[0][0]>atol:
        _,_,a,b,branchraw,_=heapq.heappop(heap)
        left,right=split(branchraw)
        add(a,(a+b)/2,left);add((a+b)/2,b,right)
    lower=heap[0][0]
    return dict(lower=lower,upper=incumbent+2e-11,argmin=argmin,cells=counter,
                converged=incumbent-lower<=atol,
                policy=[dict(interval=[a,b],processing=p) for _,_,a,b,_,p in sorted(heap,key=lambda v:v[2])])

def search(eb,ep):
    start=time.time();entries,_,kernels=load_libraries(eb==ep)
    blind=json.loads((DATA/f'bb84_{eb:.5f}_{ep:.5f}.json').read_text())['results']
    initial=blind['real_M1']
    best={key:dict(lower=initial['lower'],ED=initial['ED'],source='unobservable X/Z fixed strategy') for key in blind}
    targets=[]
    for e in entries:
        p=observable(e)
        if p is not None:targets.append((e,p))
    entries=[e for e,p in targets]
    upper,raws=anchor_upper_bounds(entries,eb,ep,kernels)
    evaluated={}
    for i in np.argsort(-upper):
        e=entries[i];keys=[k for k in best if e['M']<=int(k[-1]) and (k.startswith('full') or e['real'])]
        if all(upper[i]<=best[k]['lower']+2e-8 for k in keys):continue
        result=certify(raws[i],e['m'],kernels)
        evaluated[int(i)]=result
        for key in keys:
            if result['lower']>best[key]['lower']:
                best[key]={**result,'ED':e['id'],'M':e['M'],'m':e['m'],'observable_pauli':targets[i][1]}
        print('observable BB84',eb,ep,e['id'],'M',e['M'],'bounds',result['lower'],result['upper'],'cells',result['cells'],'seconds',round(time.time()-start,1),flush=True)
    for key,result in best.items():
        eligible=[i for i,e in enumerate(entries) if e['M']<=int(key[-1]) and (key.startswith('full') or e['real'])]
        u=max([initial['finite_class_upper']]+[min(upper[i],evaluated[i]['upper']) if i in evaluated else upper[i] for i in eligible])
        result['finite_class_upper']=float(u);result['gap']=float(u-result['lower'])
    return dict(eb=eb,ep=ep,seconds=time.time()-start,results=best,
                scope='Quantum Clifford block fixed before z; AD rule adapts to statistically identifiable z. Product X/Z circuits use the fixed-strategy minimax. m<=4, n<=6. Numerical continuous-input bounds.')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--eb',type=float,default=.2);parser.add_argument('--ep',type=float)
    args=parser.parse_args();ep=args.eb if args.ep is None else args.ep
    result=search(args.eb,ep)
    (DATA/f'bb84_observable_{args.eb:.5f}_{ep:.5f}.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
