"""One fixed classical terminal action; signed net OTP for compound inputs."""
from core import *
from ad_modes import MODES,OTPKernel,repetition_components,spc_otp_components
from literature_rates import watanabe
from spc_dynamic_fast import all_rates as spc_rates
from bb84_fixed import physical_kernel
from functools import lru_cache
from math import comb,factorial
from fractions import Fraction as F
import interval_jets as j
import stable_intervals as iv


@lru_cache(None)
def kernels():return {k.entry['id']:k for k in load_codes()}

@lru_cache(None)
def physical(code):return physical_kernel(kernels()[code])


def actions(nmax=128):
    result=[]
    for k in kernels().values():
        for mode in MODES:result.append(dict(kind='linear',code=k.entry['id'],n=k.n,mode=mode))
    for n in range(8,nmax+1):
        for kind in ('repetition','SPC'):
            for mode in MODES:result.append(dict(kind=kind,n=n,mode=mode))
    result.append(dict(kind='Watanabe_2007',n=2,mode='no_otp_hashed'))
    return result


def values(p,menu):
    """No clipping of paid OTP net rates. All positive parts are gross."""
    p=np.asarray(p).reshape(-1,4);out=np.zeros((len(p),len(menu)));cache={}
    spc_ns=sorted({a['n'] for a in menu if a['kind']=='SPC' and a['mode']=='no_otp'})
    spc=spc_rates(p,np.array(spc_ns)) if spc_ns else None
    for i,a in enumerate(menu):
        key=(a['kind'],a.get('code',a['n']))
        if key not in cache:
            if a['kind']=='Watanabe_2007':cache[key]=watanabe(p)[:,None]
            elif a['kind']=='linear':
                k=kernels()[a['code']];g,c=OTPKernel(k.entry).components(p);gross=np.maximum(g,0).sum(-1)
                cache[key]=np.column_stack((k.evaluate(p),gross-(k.n-k.k)/k.n,gross-c))
            elif a['kind']=='repetition':
                no,g,c=repetition_components(p,a['n']);gross=np.maximum(g,0).sum(-1)
                cache[key]=np.column_stack((np.maximum(no,0).sum(-1),gross-(a['n']-1)/a['n'],gross-c))
            else:
                g,c=spc_otp_components(p,a['n']);gross=np.maximum(g,0).sum(-1)
                no=spc[:,spc_ns.index(a['n'])] if a['n'] in spc_ns else np.zeros(len(p))
                cache[key]=np.column_stack((no,gross-1/a['n'],gross-c))
        out[:,i]=cache[key][:,0 if a['kind']=='Watanabe_2007' else MODES.index(a['mode'])]
    return out


def monomials_jet(p,types):
    types=np.asarray(types,int);n=int(types.max());powers=[]
    for i in range(4):
        pp=[j.Jet((np.ones(p.v[0].shape[:-1]),np.ones(p.v[0].shape[:-1])))]
        for _ in range(n):pp.append((pp[-1]*p[...,i]).clip())
        powers.append(pp)
    result=j.Jet((np.ones(p.v[0].shape[:-1]+(len(types),)),np.ones(p.v[0].shape[:-1]+(len(types),))))
    for i in range(4):result=(result*j.stack([powers[i][t] for t in types[:,i]])).clip()
    return result


def code_jet(p,kernel):
    power=monomials_jet(p,kernel.types);joint=power.dot(kernel.polys.T).clip();ps=power.dot(kernel.marginal.T).clip()
    gain=kernel.k*ps-j.h(joint).dot(kernel.hist.T)+j.h(ps)
    gross=j.positive(gain).dot(kernel.multiplicity)/kernel.n
    cost=j.h(ps).dot(kernel.multiplicity)/kernel.n
    return gross,cost


@lru_cache(None)
def spc_type_table(n):
    rows=[];coeff=[];parities=[];fac=[factorial(i) for i in range(n+1)]
    for typ in compositions(n):
        flip=tuple(reversed(typ))
        if typ>flip:continue
        rows.append(typ);parities.append((typ[1]+typ[2])%2)
        coeff.append(iv.rational(F(fac[n]//np.prod([fac[t] for t in typ],dtype=object),2 if typ==flip else 1)))
    return np.array(rows),np.array(parities),tuple(np.array([v[i] for v in coeff]) for i in (0,1))


def spc_no_jet(p,n):
    types,parities,coeff=spc_type_table(n);ent=[j.Jet.constant(0),j.Jet.constant(0)]
    for start in range(0,len(types),512):
        stop=min(start+512,len(types));typ=types[start:stop]
        a=monomials_jet(p,typ);b=monomials_jet(p,typ[:,::-1]);merged=(a+b).clip()
        weighted=j.h(merged)*j.Jet((coeff[0][start:stop],coeff[1][start:stop]))
        for s in (0,1):ent[s]=ent[s]+weighted[...,parities[start:stop]==s].sum()
    d=1-2*(p[...,1]+p[...,2]);ps=[((1+sign*d**n)/2).clip() for sign in (1,-1)]
    return sum((j.positive((n-1)*s-e+j.h(s)) for s,e in zip(ps,ent)),j.Jet.constant(0))/n


def terminal_jet(p,action):
    kind,mode,n=action['kind'],action['mode'],action['n']
    if kind=='Watanabe_2007':return j.watanabe(p)
    if kind=='linear':
        k=kernels()[action['code']];kk=k if mode=='no_otp' else physical(action['code']);gross,cost=code_jet(p,kk)
        if mode=='otp_plain':return gross-F(n-k.k,n)
        return gross if mode=='no_otp' else gross-cost
    if kind=='SPC':
        if mode=='no_otp':
            from spc_jet_dynamic import spc_jet
            return spc_jet(p,n)
        d=1-2*(p[...,1]+p[...,2]);entropy=j.h(p);hp=entropy.dot(np.array([1,0,0,1]))-entropy.dot(np.array([0,1,1,0]))
        htotal=entropy.sum();gross=j.Jet.constant(0);cost=j.Jet.constant(0)
        for sign in (1,-1):
            ps=((1+sign*d**n)/2).clip();raw_h=n*(htotal+sign*d**(n-1)*hp)/2
            gross=gross+j.positive(n*ps-raw_h+j.h(ps))/n;cost=cost+j.h(ps)/n
        return gross-(F(1,n) if mode=='otp_plain' else cost)
    a=(p[...,0]+p[...,3]).clip();b=(p[...,1]+p[...,2]).clip()
    u=p[...,0]-p[...,3];v=p[...,1]-p[...,2]
    gross=j.Jet.constant(0);cost=j.Jet.constant(0)
    if mode!='no_otp':
        h0=((j.h(p[...,0])+j.h(p[...,3])-j.h(a))/a).clip()
        h1=((j.h(p[...,1])+j.h(p[...,2])-j.h(b))/b).clip()
    for w in range(n//2+1):
        aa=(a**(n-w)*b**w).clip();bb=(a**w*b**(n-w)).clip();ps=(aa+bb).clip()
        mult=F(comb(n,w),2 if 2*w==n else 1)
        if mode=='no_otp':
            cc=u**(n-w)*v**w;dd=u**w*v**(n-w)
            raw=j.stack([(aa+cc)/2,(bb+dd)/2,(bb-dd)/2,(aa-cc)/2]).clip()
            gain=ps-j.h(raw).sum()+j.h(ps)
        else:
            hmass=j.h(aa)+j.h(bb)+aa*((n-w)*h0+w*h1)+bb*(w*h0+(n-w)*h1)-j.h(ps)
            gain=n*ps-hmass
        gross=gross+mult*j.positive(gain)/n;cost=cost+mult*j.h(ps)/n
    return gross-(F(n-1,n) if mode=='otp_plain' else cost) if mode!='no_otp' else gross
