"""Outward evaluation of every registered decoder record, batched sources."""
from core import *
from fractions import Fraction as F
from recurrence_v2 import PRIMITIVES,MAPS,generate
from recursive_library import PRE
import stable_intervals as iv


def raw_step(p,q,primitive):
    shape=np.broadcast_shapes(p[0].shape,q[0].shape)
    raw=(np.zeros(shape),np.zeros(shape))
    for i,j,k in MAPS[primitive]:
        v=iv.mul((p[0][...,i],p[1][...,i]),(q[0][...,j],q[1][...,j]))
        out=iv.add((raw[0][...,k],raw[1][...,k]),v)
        for t in (0,1):raw[t][...,k]=np.maximum(0,out[t])
    return raw


def step(p,q,primitive):
    raw=raw_step(p,q,primitive)
    mass=iv.positive_sum(raw);mass=(np.maximum(0,mass[0]),np.minimum(1,mass[1]))
    good=mass[0]>0;safe=(np.where(good,mass[0],1),np.where(good,mass[1],1))
    out=iv.div(raw,(safe[0][...,None],safe[1][...,None]))
    return (np.where(good[...,None],np.maximum(0,out[0]),0.),
            np.where(good[...,None],np.minimum(1,out[1]),1.)),mass


def generate_bounds(initial,permissions='real',depth=7,rmax=160):
    """Input arrays (sources,4), output (records,sources,4), original order."""
    initial=tuple(v[None,...] for v in initial);frontier=initial
    y=tuple(np.ones(initial[0].shape[:-1]) for _ in (0,1))
    bases=('Z','X') if permissions=='real' else ('Z','X','Y');ps=[];ys=[]
    for d in range(depth+1):
        ps.append(tuple(np.stack([v[...,list(PRE[a])] for a in bases],1).reshape(-1,initial[0].shape[1],4) for v in frontier))
        ys.append(tuple(np.repeat(v,len(bases),axis=0) for v in y))
        if d<depth:
            shape=(frontier[0].shape[0]*4,)+frontier[0].shape[1:]
            nxt=tuple(np.empty(shape) for _ in (0,1));yy=tuple(np.empty(shape[:-1]) for _ in (0,1))
            for i,primitive in enumerate(PRIMITIVES):
                state,s=step(frontier,frontier,primitive);scale=iv.scale(iv.mul(y,s),F(1,2))
                for j in (0,1):nxt[j][i::4]=state[j];yy[j][i::4]=scale[j]
            frontier=nxt;y=yy
    for primitive in PRIMITIVES:
        raw=initial;mass=tuple(np.ones(initial[0].shape[:-1]) for _ in (0,1))
        prefix=mass;state=initial;y=mass
        for r in range(1,rmax+1):
            if r>1:
                prefix=iv.add(prefix,mass);raw=raw_step(raw,initial,primitive);mass=iv.positive_sum(raw)
                state=iv.div(raw,(mass[0][...,None],mass[1][...,None]))
                state=(np.maximum(0,state[0]),np.minimum(1,state[1]));y=iv.div(mass,prefix)
            for axis in bases:
                ps.append(tuple(v[...,list(PRE[axis])] for v in state));ys.append(y)
    return (tuple(np.concatenate([v[i] for v in ps],axis=0) for i in (0,1)),
            tuple(np.concatenate([v[i] for v in ys],axis=0) for i in (0,1)))


def records(permissions='real',depth=7,rmax=160):
    _,_,M,rr=generate(np.array([.7,.1,.1,.1]),depth,permissions,rmax)
    return M,rr


def verify():
    from certify_bb84_recursive import source
    count=0;largest=0.
    for b,p in ((.2,.2),(.13,.27)):
        ts=[F(0),F(1,64),F(1,2),F(1)]
        initial=source(b,p,[(t,t) for t in ts]);a,y=generate_bounds(initial,'full')
        zz=min(b,p)*np.array(ts,float);raw=np.stack((1-b-p+zz,b-zz,zz,p-zz),-1)
        fp,fy,_,_=generate(raw,7,'full')
        # Floating source decimal conversion can be an ulp outside the exact
        # rational source. This is a diagnostic, not a proof substitute.
        assert np.all(fp>=a[0]-3e-14) and np.all(fp<=a[1]+3e-14)
        assert np.all(fy>=y[0]-3e-14) and np.all(fy<=y[1]+3e-14)
        count+=fp.size+fy.size;largest=max(largest,float(np.max(a[1]-a[0])))
    atomic_json(DATA/'recurrence_interval_library_checks.json',dict(status='passed',checks=count,max_probability_width=largest))
    print('recurrence interval checks',count,largest,flush=True)


if __name__=='__main__':verify()
