"""Floating Hellinger walk with reusable displacement entropy factors.

This is discovery arithmetic. The independent interval and Decimal
implementations continue to provide numerical certificates.
"""
from core import *


def ambiguity(u,v,a,b):
    with np.errstate(invalid='ignore',over='ignore',divide='ignore'):
        likelihood=np.abs(np.where(a[None,:,:]!=0,u[:,None,None]*a,0)+np.where(b[None,:,:]!=0,v[:,None,None]*b,0))
        t=np.exp(-likelihood);ratio=np.divide(np.log1p(t),t,out=np.ones_like(t),where=t>0)
        result=np.exp(-likelihood/2)*((1+t)*ratio+likelihood)/(2*np.log(2.))
    return np.nan_to_num(result,nan=0.,posinf=0.)


def all_rates(p,lengths,needed=None,batch=12):
    p=np.asarray(p);lengths=np.asarray(lengths,int)
    if needed is None:needed=np.ones((len(p),len(lengths)),bool)
    result=np.zeros(needed.shape);maxneeded=np.max(np.where(needed,lengths[None,:],0),axis=1)
    indices=np.flatnonzero(maxneeded>0);indices=indices[np.argsort(maxneeded[indices])];column={int(n):j for j,n in enumerate(lengths)}
    for start in range(0,len(indices),batch):
        ids=indices[start:start+batch];q=p[ids];g0=np.sqrt(q[:,0]*q[:,3]);g1=np.sqrt(q[:,1]*q[:,2])
        with np.errstate(divide='ignore',invalid='ignore'):
            u=np.log(q[:,0])-np.log(q[:,3]);v=np.log(q[:,1])-np.log(q[:,2])
        h=entropy(q);hp=(-xlog2x(q)*np.array([1,-1,-1,1])).sum(1);d=1-2*(q[:,1]+q[:,2])
        largest=int(maxneeded[ids].max());requested=[int(n) for n in lengths if needed[ids,column[n]].any()]
        cached=None
        if sum((n+1)**2 for n in requested)>3*(2*largest+1)**2:
            aa=np.arange(-largest,largest+1)[:,None];bb=np.arange(-largest,largest+1)[None,:]
            cached=ambiguity(u,v,aa,bb)
        distribution=np.ones((len(ids),1,1))
        for n in range(1,largest+1):
            nxt=np.zeros((len(ids),n+1,n+1));aa=g0[:,None,None]*distribution;bb=g1[:,None,None]*distribution
            nxt[:,:-1,:-1]+=aa;nxt[:,1:,1:]+=aa;nxt[:,1:,:-1]+=bb;nxt[:,:-1,1:]+=bb;distribution=nxt
            if n not in column:continue
            j=column[n];take=np.flatnonzero(needed[ids,j])
            if not len(take):continue
            ii=np.arange(n+1)[:,None];jj=np.arange(n+1)[None,:];a=n-ii-jj;b=jj-ii
            amb=ambiguity(u[take],v[take],a,b) if cached is None else cached[take[:,None,None],a[None,:,:]+largest,b[None,:,:]+largest]
            weighted=distribution[take]*amb;even=(ii+jj)%2==0
            loss=np.stack((weighted[:,even].sum(1),weighted[:,~even].sum(1)),-1)
            sign=np.array([1,-1]);ps=(1+sign[None,:]*d[take,None]**n)/2
            physical=n/2*(h[take,None]+sign[None,:]*d[take,None]**(n-1)*hp[take,None])+xlog2x(ps)
            gain=(n-1)*ps-physical+loss;result[ids[take],j]=np.maximum(gain,0.).sum(1)/n
    return result


def verify():
    from spc_dynamic import all_rates as original
    import time
    rng=np.random.default_rng(6659);p=np.vstack((rng.dirichlet(np.ones(4),24),np.eye(4),[[.5,0,.5,0],[.6,.2,0,.2]]))
    lengths=np.arange(8,129);needed=rng.random((len(p),len(lengths)))>.15
    start=time.time();ref=original(p,lengths,needed);old_time=time.time()-start
    start=time.time();vv=all_rates(p,lengths,needed);new_time=time.time()-start
    err=float(np.max(np.abs(vv-ref)));assert err<3e-14
    result=dict(status='passed',comparisons=vv.size,max_error=err,old_seconds=old_time,new_seconds=new_time)
    atomic_json(DATA/'spc_dynamic_fast_checks.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':verify()
