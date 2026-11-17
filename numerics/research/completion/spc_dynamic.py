"""Exact SPC formula by a two-dimensional Hellinger random walk.

All lengths <=N cost O(N^3) per Bell input, versus O(N^4) for separately
enumerating four-category types at every length. Floating evaluation;
the independent type/Decimal formula supplies witness certificates.
"""
import numpy as np
from rates import xlog2x,entropy


def all_rates(p,lengths,needed=None,batch=12):
    p=np.asarray(p);lengths=np.asarray(lengths,dtype=int)
    if needed is None:needed=np.ones((len(p),len(lengths)),dtype=bool)
    result=np.zeros(needed.shape);maxneeded=np.max(np.where(needed,lengths[None,:],0),axis=1)
    indices=np.flatnonzero(maxneeded>0);indices=indices[np.argsort(maxneeded[indices])]
    column={int(n):j for j,n in enumerate(lengths)};ln2=np.log(2.)
    for start in range(0,len(indices),batch):
        ids=indices[start:start+batch];q=p[ids];g0=np.sqrt(q[:,0]*q[:,3]);g1=np.sqrt(q[:,1]*q[:,2])
        with np.errstate(divide='ignore',invalid='ignore'):
            u=np.log(q[:,0])-np.log(q[:,3]);v=np.log(q[:,1])-np.log(q[:,2])
        h=entropy(q);hp=(-xlog2x(q)*np.array([1,-1,-1,1])).sum(1);d=1-2*(q[:,1]+q[:,2])
        distribution=np.ones((len(ids),1,1))
        for n in range(1,int(maxneeded[ids].max())+1):
            nxt=np.zeros((len(ids),n+1,n+1));aa=g0[:,None,None]*distribution;bb=g1[:,None,None]*distribution
            nxt[:,:-1,:-1]+=aa;nxt[:,1:,1:]+=aa;nxt[:,1:,:-1]+=bb;nxt[:,:-1,1:]+=bb;distribution=nxt
            if n not in column:continue
            j=column[n];take=np.flatnonzero(needed[ids,j])
            if not len(take):continue
            ii=np.arange(n+1)[:,None];jj=np.arange(n+1)[None,:];a=n-ii-jj;b=jj-ii
            with np.errstate(invalid='ignore',over='ignore',divide='ignore'):
                likelihood=np.abs(np.where(a[None,:,:]!=0,u[take,None,None]*a,0)+np.where(b[None,:,:]!=0,v[take,None,None]*b,0))
                t=np.exp(-likelihood);ratio=np.divide(np.log1p(t),t,out=np.ones_like(t),where=t>0)
                ambiguity=np.exp(-likelihood/2)*((1+t)*ratio+likelihood)/(2*ln2)
            ambiguity=np.nan_to_num(ambiguity,nan=0.,posinf=0.);weighted=distribution[take]*ambiguity
            even=(ii+jj)%2==0;loss=np.stack((weighted[:,even].sum(1),weighted[:,~even].sum(1)),-1)
            sign=np.array([1,-1]);ps=(1+sign[None,:]*d[take,None]**n)/2
            physical=n/2*(h[take,None]+sign[None,:]*d[take,None]**(n-1)*hp[take,None])+xlog2x(ps)
            gain=(n-1)*ps-physical+loss
            result[ids[take],j]=np.maximum(gain,0.).sum(1)/n
    return result
