"""Fast numeric and outward-rounded evaluation of the complete binary DAG."""
import numpy as np
from numba import njit

@njit(cache=True)
def floating(nodes,events,source):
    size=len(nodes);nt=len(source)
    p=np.zeros((size,nt,4));y=np.zeros((size,nt))
    p[0]=source;y[0]=1.
    for n in range(1,size):
        op,l,r=nodes[n,:3]
        for t in range(nt):
            for e in range(8):
                i,j,k=events[op,e];p[n,t,k]+=p[l,t,i]*p[r,t,j]
            s=0.
            for k in range(4):s+=p[n,t,k]
            if s>0:
                for k in range(4):p[n,t,k]/=s
            d=y[l,t]+y[r,t]
            if d>0:y[n,t]=s*(y[l,t]/d)*y[r,t]
    return p,y

@njit(cache=True)
def down(x):return max(0.,np.nextafter(x,-np.inf))
@njit(cache=True)
def up(x):return np.nextafter(x,np.inf)

@njit(cache=True)
def outward(nodes,events,source_lo,source_hi):
    size=len(nodes);nt=len(source_lo)
    lo=np.zeros((size,nt,4));hi=np.zeros((size,nt,4))
    yl=np.zeros((size,nt));yh=np.zeros((size,nt))
    lo[0]=source_lo;hi[0]=source_hi;yl[0]=1.;yh[0]=1.
    for n in range(1,size):
        op,l,r=nodes[n,:3]
        for t in range(nt):
            for e in range(8):
                i,j,k=events[op,e]
                lo[n,t,k]=down(lo[n,t,k]+down(lo[l,t,i]*lo[r,t,j]))
                hi[n,t,k]=up(hi[n,t,k]+up(hi[l,t,i]*hi[r,t,j]))
            sl=0.;sh=0.
            for k in range(4):sl=down(sl+lo[n,t,k]);sh=up(sh+hi[n,t,k])
            sh=min(sh,1.)
            for k in range(4):
                lo[n,t,k]=down(lo[n,t,k]/sh) if sh>0 else 0.
                hi[n,t,k]=min(1.,up(hi[n,t,k]/sl)) if sl>0 else 1.
            dl=down(yl[l,t]+yl[r,t]);dh=up(yh[l,t]+yh[r,t])
            if dh>0:yl[n,t]=down(down(sl*down(yl[l,t]/dh))*yl[r,t])
            if dl>0:yh[n,t]=min(yh[l,t],yh[r,t],up(up(sh*up(yh[l,t]/dl))*yh[r,t]))
            else:yh[n,t]=up(sh*min(yh[l,t],yh[r,t]))
    return lo,hi,yl,yh
