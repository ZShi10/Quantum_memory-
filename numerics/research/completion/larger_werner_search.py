"""Evaluate every member of a frozen larger graph family, no ED shortlist."""
from common import *
from math import comb
from fractions import Fraction as F
import json,time
import numpy as np
from algebra import compositions
from libraries import code_kernels
from rates import best_codes
from streaming import rate as repetition_rate
from complete_werner_long import spc_upper_array
from spc_dynamic import all_rates
from screening import incumbents
from interval_certificate import IV


def library():
    cost=json.loads((DATA/'graph_memory_exact.json').read_text())
    large=json.loads((DATA/'memory_eight_graphs.json').read_text())
    old=json.loads((EXT/'data/graphs.json').read_text())['entries']
    seen={}
    for original in old+large['entries']:
        e=original.copy();m=e['m']
        if m<=8:continue
        if not e['id'].startswith('LARGE'):
            e['M']=cost['graphs'][cost['entry_to_graph'][e['id']]]['M']
        if 'character_weights' not in e:
            w=np.zeros((4,m+1),dtype=np.int64)
            counts=np.array(e['characters'])
            for j,t in enumerate(compositions(m)):w[:,m-t[0]]+=counts[:,j]
            e['character_weights']=w.tolist()
        key=(m,str(e['character_weights']))
        if key not in seen or e['M']<seen[key]['M']:seen[key]=e
    entries=list(seen.values());transforms={}
    for e in entries:
        m=e['m']
        if m not in transforms:
            transforms[m]=np.array([[sum((-1)**a*comb(j,a)*comb(m-j,w-a)*3**(w-a)
                for a in range(max(0,w-m+j),min(j,w)+1)) for w in range(m+1)] for j in range(m+1)],dtype=object)
        raw=np.array(e['character_weights'],dtype=object)@transforms[m]
        den=2**(m+1)
        assert all(x>=0 and x%den==0 for x in raw.ravel())
        w=raw//den;assert sum(w.ravel())==den
        e['positive_werner_weights']=w.tolist()
    return entries


def search(q,entries,ks):
    qf=F(str(q));raw_exact=[];upper=[]
    for e in entries:
        m=e['m'];powers=[(1-F(3,2)*qf)**(m-j)*(qf/2)**j for j in range(m+1)]
        raw=[sum((int(c)*v for c,v in zip(row,powers)),F()) for row in e['positive_werner_weights']]
        raw_exact.append(raw);ps=sum(raw);c=max(F(),2*max(raw)-ps)
        upper.append(float((IV(c)*IV(c)/IV(ps*m)).hi) if ps else 0.)
    raw=np.array(raw_exact,dtype=float);ps=raw.sum(1);state=raw/ps[:,None]
    sizes=np.array([e['m'] for e in entries]);weight=ps/sizes;inc=incumbents(q)
    relevant=np.array([upper[i]+1e-15>=inc[e['M']] for i,e in enumerate(entries)])
    best=np.zeros(len(entries));choice=[None]*len(entries);factor=np.ones(len(entries));ns=np.arange(8,129);nr=np.arange(1,129)
    for k in range(17):
        ids=np.flatnonzero(relevant&(weight*factor*np.maximum(0,2*state.max(1)-1)**2>best+1e-15))
        if not len(ids):break
        p=state[ids].copy();flip=p[:,1]+p[:,2]>.5;p[flip]=p[flip][:,[1,0,3,2]]
        rates=best_codes(p,ks);r=rates['rate']*weight[ids]*factor[ids]
        for j in np.flatnonzero(r>best[ids]+1e-16):
            i=ids[j];best[i]=r[j];choice[i]=dict(kind='linear',code=ks[rates['code'][j]].entry['id'],binary_rounds=k)
        a=p[:,0]+p[:,3];b=p[:,1]+p[:,2]
        values=repetition_rate(np.broadcast_to((a-b)[:,None],(len(ids),len(nr))),p[:,3,None]/a[:,None],p[:,2,None]/b[:,None],0,nr[None,:])*weight[ids,None]*factor[ids,None]
        at=values.argmax(1);r=values[np.arange(len(ids)),at]
        for j in np.flatnonzero(r>best[ids]+1e-16):
            i=ids[j];best[i]=r[j];choice[i]=dict(kind='repetition',n=int(nr[at[j]]),binary_rounds=k)
        mask=spc_upper_array(p,ns)*weight[ids,None]*factor[ids,None]>best[ids,None]+1e-15
        values=all_rates(p,ns,mask)*weight[ids,None]*factor[ids,None];at=values.argmax(1);r=values[np.arange(len(ids)),at]
        for j in np.flatnonzero(r>best[ids]+1e-16):
            i=ids[j];best[i]=r[j];choice[i]=dict(kind='SPC',n=int(ns[at[j]]),binary_rounds=k)
        a,x,y,z=np.moveaxis(state,-1,0);out=np.stack((a*a+z*z,x*x+y*y,2*x*y,2*a*z),-1);mass=out.sum(1);factor*=mass/2;state=out/mass[:,None]
    results={}
    for M in range(1,9):
        ids=[i for i,e in enumerate(entries) if e['M']<=M and choice[i] is not None]
        if not ids:continue
        i=max(ids,key=lambda j:best[j]);e=entries[i]
        results[f'M{M}']=dict(Q=q,rate=float(best[i]),ED=e['id'],m=e['m'],M=e['M'],branches=[dict(**choice[i],syndrome=0,canonicalize_bit=True,contribution=float(best[i]))])
    return dict(Q=q,results=results,screened_by_proven_upper=int((~relevant).sum()),total_entries=len(entries),
                memory_7_8_max_upper=max((upper[i] for i,e in enumerate(entries) if e['M']>=7),default=0.))


def main():
    entries=library();ks=code_kernels();dest=DATA/'larger_werner.json'
    saved=json.loads(dest.read_text()) if dest.exists() else dict(results=[])
    saved['protocol_library']={e['id']:e for e in entries};done={r['Q'] for r in saved['results']};start=time.time()
    for q in [.10,.15,.18,.20,.21,.22,.23,.24,.25,.26,.27,.275,.28,.285,.29,.295,.30,.305,.31,.315,.32]:
        if q in done:continue
        r=search(q,entries,ks);saved['results'].append(r)
        saved['scope']='Frozen graph families m=9..20, zero ED syndrome, exact per-measurement memory M<=8, all codes n<=7 and all repetition/SPC n<=128, pooled B prefixes<=16. Every candidate evaluated or excluded by a certified concurrence upper compared with a feasible short-core incumbent. Larger constructed family, not complete Clifford enumeration.'
        dest.write_text(json.dumps(saved,separators=(',',':')))
        print('Larger Werner',q,'screened',r['screened_by_proven_upper'],'/',len(entries),[(k,v['rate'],v['ED']) for k,v in r['results'].items() if k in ('M2','M4','M8')],'seconds',round(time.time()-start,1),flush=True)
    saved['status']='passed';dest.write_text(json.dumps(saved,separators=(',',':')))


if __name__=='__main__':main()
