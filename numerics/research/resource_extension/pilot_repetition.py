"""A fair long-repetition BB84 baseline with a legal parameter pilot.

M=1 full uses a Y pilot; M=2 real/full can use a Bell-basis pilot. The
production rule may choose a direct axis or a raw-pair pumping length.
Continuous bounds use the stable low-degree ED ratio construction.
"""
from common import *
import json,time,heapq,argparse
import numpy as np
from streaming import direct,pumping,repeated_parameters,binary_entropy
from bernstein import split


def restrict(coeff,a,b):
    if b<1:coeff=split(coeff,b)[0]
    if a>0:coeff=split(coeff,a/b)[1]
    return coeff


def optimize(parameters,rtol=2e-4,atol=1e-18,kmax=64,max_cells=6000):
    ks=np.arange(kmax+1);ns=np.arange(1,129);N=(2.**ks[:,None])*ns[None,:]
    heap=[];counter=0;upper=float('inf')
    def point(t):
        values=np.array([p.values(t) for p in parameters]);eps,p0,p1,y=values.T
        with np.errstate(divide='ignore',invalid='ignore'):
            alpha=np.arctanh(np.clip(eps,0,1));ek=np.tanh(alpha[:,None]*(2.**ks[None,:]))
        step=(1+ek**2)/4;prefix=np.c_[np.ones(len(parameters)),np.cumprod(step[:,:-1],axis=1)]
        e,d0,d1=repeated_parameters(eps[:,None,None],p0[:,None,None],p1[:,None,None],N[None,:,:])
        final=(((1+ek[:,:,None])/2)**ns+((1-ek[:,:,None])/2)**ns)/ns
        rate=y[:,None,None]*prefix[:,:,None]*final*np.maximum((1-e)*d0+e*d1-binary_entropy(e),0.)
        index=np.unravel_index(rate.argmax(),rate.shape)
        return float(rate[index]),(int(index[0]),int(index[1]),int(ns[index[2]]))
    def add(a,b):
        nonlocal upper,counter
        sample,(i,k,n)=point((a+b)/2);upper=min(upper,sample)
        parameter=parameters[i];arrays=[(restrict(x,a,b),restrict(y,a,b)) for x,y in parameter.ratios]
        lower=parameter.lower(arrays,k,n)
        rule=dict(interval=[a,b],ED=parameter.name,parameters=parameter.metadata,M=parameter.memory,binary_rounds=k,final_n=n,lower=lower)
        heapq.heappush(heap,(lower,counter,a,b,rule));counter+=1
    add(0.,1.)
    upper=min(upper,point(0.)[0],point(1.)[0])
    while counter<max_cells and upper-heap[0][0]>atol+rtol*upper:
        _,_,a,b,_=heapq.heappop(heap);add(a,(a+b)/2);add((a+b)/2,b)
    return dict(lower=heap[0][0],upper=upper,cells=counter,converged=upper-heap[0][0]<=atol+rtol*upper,
                policy=[x[4] for x in sorted(heap,key=lambda x:x[2])])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--memory',type=int,default=1)
    parser.add_argument('--permissions',choices=['real','full'],default='full')
    args=parser.parse_args();assert args.permissions=='full' or args.memory>=2
    results=[];started=time.time();suffix='_real' if args.permissions=='real' else ''
    for q in (.20,.21,.22,.23,.24,.245):
        parameters=[direct(q,q,j) for j in range(3 if args.permissions=='full' else 2)]
        if args.memory>=2:parameters+=[pumping(q,q,r,0) for r in range(2,33)]
        result=optimize(parameters,atol=0. if args.permissions=='real' else 1e-18);results.append(dict(Q=q,**result))
        print('Long pilot repetition M',args.memory,q,result['lower'],result['upper'],'cells',result['cells'],'seconds',round(time.time()-started,1),flush=True)
        (DATA/f'bb84_long_repetition_M{args.memory}{suffix}.json').write_text(json.dumps(dict(permissions=args.permissions,
            scope='Legal parameter pilot (Y for full, two-register Bell-basis for real); allowed direct axes and, for M2, selected raw pumping r<=32. Binary AD prefixes <=64 and final repetition n<=128. Continuous numerical bounds; not an all-code optimum.',results=results),separators=(',',':')))


if __name__=='__main__':main()
