"""Pilot-identifiable BB84: a finite decision rule may choose ED and AD.

M>=2 real gates can identify z by a Bell-basis test; full M=1 can use Y.
For real M=1 the oracle is only an optimistic bound, not an attainable rate.
All production circuits and AD codes still come from the declared library.
"""
from pathlib import Path
import heapq,json,time,argparse
import numpy as np
from search_bb84 import load_libraries,raw_polynomials
from bernstein import evaluate,split,GainCurve
from rates import best_codes

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data'

def restrict(raw,a,b):
    if b<1:raw=split(raw,b)[0]
    if a>0:raw=split(raw,a/b)[1]
    return raw

class Oracle:
    def __init__(self,eb,ep):
        self.entries,_,self.kernels=load_libraries(eb==ep)
        self.raws=[raw_polynomials(e,eb,ep) for e in self.entries]
        self.starts=np.cumsum([0]+[len(r) for r in self.raws])
        self.ms=np.concatenate([np.full(len(raw),e['m']) for raw,e in zip(self.raws,self.entries)])
        self.groups={m:[] for m in range(1,5)}
        for i,e in enumerate(self.entries):self.groups[e['m']].append(i)
        self.degree_arrays={m:np.concatenate([self.raws[i] for i in ids]) for m,ids in self.groups.items() if ids}
        self.degree_positions={m:np.concatenate([np.arange(self.starts[i],self.starts[i+1]) for i in ids]) for m,ids in self.groups.items() if ids}
        self.cache={}

    def point(self,t):
        if t in self.cache:return self.cache[t]
        value=np.empty((len(self.ms),4))
        for m,array in self.degree_arrays.items():value[self.degree_positions[m]]=evaluate(array,t)
        ps=value.sum(1);p=np.divide(value,ps[:,None],out=np.full_like(value,.25),where=ps[:,None]>0)
        codes=best_codes(p,self.kernels)
        contributions=ps/self.ms*codes['rate']
        rates=np.add.reduceat(contributions,self.starts[:-1])
        # Save only selected code indices and total rates; states are cheap
        # to re-evaluate for the single chosen candidate.
        result=(rates,codes['code']);self.cache[t]=result
        return result

    def certify(self,mode,M,atol=5e-9,max_cells=2500):
        eligible=np.array([i for i,e in enumerate(self.entries) if e['M']<=M and (mode=='full' or e['real'])])
        heap=[];counter=0;incumbent=float('inf');argmin=None
        def add(a,b):
            nonlocal counter,incumbent,argmin
            t=(a+b)/2;rates,codes=self.point(t);bi=int(eligible[np.argmax(rates[eligible])]);entry=self.entries[bi]
            sample=float(rates[bi])
            if sample<incumbent:incumbent,argmin=sample,t
            raw=restrict(self.raws[bi],a,b);value=evaluate(raw,.5);ps=value.sum(1)
            p=np.divide(value,ps[:,None],out=np.full_like(value,.25),where=ps[:,None]>0)
            lower=0.;processing=[]
            for s,ci in enumerate(codes[self.starts[bi]:self.starts[bi+1]]):
                if ci<0:continue
                kernel=self.kernels[int(ci)];mass,h=kernel.components(p[s]);keep=kernel.k*mass-h>0
                branch_lower=GainCurve(raw[s],kernel,keep,scale=1/entry['m']).interval()[0]-2e-12
                # A clipped zero must be implemented by actually discarding
                # this branch throughout the policy cell.
                if branch_lower<=0:continue
                lower+=branch_lower
                processing.append(dict(syndrome=s,code=kernel.entry['id'],keep_classes=np.flatnonzero(keep).tolist()))
            policy=dict(interval=[a,b],ED=entry['id'],M=entry['M'],processing=processing)
            heapq.heappush(heap,(lower,counter,a,b,policy));counter+=1
        add(0.,1.)
        for t in (0.,1.):
            rates,_=self.point(t);sample=float(rates[eligible].max())
            if sample<incumbent:incumbent,argmin=sample,t
        while counter<max_cells and incumbent-heap[0][0]>atol:
            _,_,a,b,_=heapq.heappop(heap);add(a,(a+b)/2);add((a+b)/2,b)
        lower=heap[0][0]
        return dict(lower=lower,upper=incumbent+2e-11,argmin=argmin,cells=counter,
                    converged=incumbent-lower<=atol,achievable=mode=='full' or M>=2,
                    policy=[v[4] for v in sorted(heap,key=lambda v:v[2])])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--eb',type=float,default=.20);parser.add_argument('--ep',type=float)
    args=parser.parse_args();ep=args.eb if args.ep is None else args.ep;start=time.time()
    oracle=Oracle(args.eb,ep);results={}
    for mode,M in [('real',1),('real',2),('real',3),('real',4),('full',1),('full',2)]:
        r=oracle.certify(mode,M);results[f'{mode}_M{M}']=r
        print('pilot BB84',args.eb,ep,mode,M,r['lower'],r['upper'],'cells',r['cells'],'seconds',round(time.time()-start,1),flush=True)
    result=dict(eb=args.eb,ep=ep,results=results,seconds=time.time()-start,
                scope='Finite m<=4,n<=6 library; negligible pilot stage identifies z for full M>=1 or real M>=2. Real M=1 is an optimistic state-informed oracle only. Continuous numerical enclosures.')
    (DATA/f'bb84_pilot_{args.eb:.5f}_{ep:.5f}.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
