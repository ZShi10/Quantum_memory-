"""Finite pumping/direct families followed by long, pooled repetition AD."""
from pathlib import Path
import json,time,argparse
import numpy as np
from streaming import pumping,direct,repeated_parameters,binary_entropy

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data'

def sample_table(ed,ks,ns,ts):
    eps,p0,p1,y=np.array([ed.values(t) for t in ts]).T
    with np.errstate(divide='ignore',invalid='ignore'):
        alpha=np.arctanh(np.clip(eps,0.,1.))
    ek=np.tanh(2.**ks[:,None]*alpha)
    factors=(1+ek**2)/4
    previous=np.r_[np.ones((1,len(ts))),np.cumprod(factors[:-1],axis=0)]
    N=(2.**ks[:,None,None])*ns[None,:,None]
    e,d0,d1=repeated_parameters(eps[None,None,:],p0[None,None,:],p1[None,None,:],N)
    gain=np.maximum((1-e)*d0+e*d1-binary_entropy(e),0.)
    final=(((1+ek[:,None,:])/2)**ns[None,:,None]+((1-ek[:,None,:])/2)**ns[None,:,None])/ns[None,:,None]
    return (y[None,None,:]*previous[:,None,:]*final*gain).min(2)

def search(eb,ep,rmax=80,kmax=160,nmax=8):
    start=time.time();eds=[direct(eb,ep,axis) for axis in range(3)]
    for axis in range(2):
        if axis and eb==ep:continue
        for r in range(2,rmax+1):eds.append(pumping(eb,ep,r,axis))
    ks=np.arange(kmax+1);ns=np.arange(1,nmax+1);ts=np.linspace(0,1,17)
    candidates=[]
    for ei,ed in enumerate(eds):
        table=sample_table(ed,ks,ns,ts)
        for ki,ni in zip(*np.where(table>0)):
            candidates.append((float(table[ki,ni]),ei,int(ks[ki]),int(ns[ni])))
    candidates.sort(reverse=True)
    best={name:dict(lower=0.,upper=0.,ED=None) for name in ('real_M1','full_M1','real_M2','full_M2')}
    for upper,ei,k,n in candidates:
        ed=eds[ei];isY=ed.metadata.get('family')=='direct' and ed.metadata.get('axis')==2
        targets=[name for name in best if (name.startswith('full') or not isY) and ed.memory<=int(name[-1])]
        if all(upper<=best[name]['lower']*(1+1e-4) for name in targets):continue
        cert=ed.certify(k,n,rtol=1e-5)
        record=dict(**cert,ED=ed.name,memory=ed.memory,parameters=ed.metadata,binary_rounds=k,final_n=n,effective_AD_length=str(2**k*n))
        for name in targets:
            if cert['lower']>best[name]['lower']:best[name]=record
        print('streaming',eb,ep,ed.name,'k,n',k,n,'lower',cert['lower'],'relgap',cert['relative_gap'],'seconds',round(time.time()-start,1),flush=True)
    # All candidates were either bounded continuously or sample-pruned.
    # We report attained strategies, not a tight optimality certificate over
    # a larger recursion or arbitrary fixed-memory class.
    return dict(eb=eb,ep=ep,rmax=rmax,kmax=kmax,nmax=nmax,seconds=time.time()-start,
                scope='Direct measurement or one-level raw-pair pumping, followed by pooled binary AD and a final repetition block; no twirl; continuous z bounds.',results=best)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--eb',type=float,default=.22);parser.add_argument('--ep',type=float)
    parser.add_argument('--rmax',type=int,default=80);parser.add_argument('--kmax',type=int,default=160)
    args=parser.parse_args();ep=args.eb if args.ep is None else args.ep
    result=search(args.eb,ep,args.rmax,args.kmax)
    path=DATA/f'streaming_{args.eb:.5f}_{ep:.5f}.json'
    path.write_text(json.dumps(result,indent=2,default=lambda x:x.item()))
    print('Saved',path.name,flush=True)

if __name__=='__main__':main()
