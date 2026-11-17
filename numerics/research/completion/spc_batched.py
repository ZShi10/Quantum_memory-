"""Vectorized type-orbit SPC entropy; same formula as spc_long.py."""
from functools import lru_cache
from math import lgamma
import numpy as np
from rates import xlog2x
from algebra import compositions


@lru_cache(None)
def table(n):
    types=np.asarray(compositions(n),dtype=np.int16);facts=np.array([lgamma(i+1) for i in range(n+1)])
    mult=facts[n]-facts[types].sum(1);parity=((types[:,1]+types[:,2])%2).astype(int)
    flags=np.stack((parity==0,parity==1),-1).astype(float)
    return types.astype(float).T,mult,flags


def rates(p,n):
    p=np.asarray(p);types,logmult,flags=table(n);out=np.zeros(len(p));batch=min(512,max(1,1000000//types.shape[1]))
    for start in range(0,len(p),batch):
        end=min(len(p),start+batch);state=p[start:end]
        with np.errstate(divide='ignore'):lp=np.log(np.where(state>0,state,1))
        first=lp@types;second=lp[:,[3,2,1,0]]@types
        if (state==0).any():
            first=np.where((state==0).astype(float)@(types>0)>0,-np.inf,first)
            second=np.where((state[:,[3,2,1,0]]==0).astype(float)@(types>0)>0,-np.inf,second)
        logw=np.logaddexp(first,second);mass=np.exp(logw+logmult[None,:]-np.log(2.))
        with np.errstate(invalid='ignore'):hmass=np.where(mass>0,-mass*logw/np.log(2.),0.)@flags
        ps=mass@flags;out[start:end]=np.maximum((n-1)*ps-hmass-xlog2x(ps),0).sum(1)/n
    return out
