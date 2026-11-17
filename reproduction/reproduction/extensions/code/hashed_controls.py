"""Common public hashed-reconciliation control with exact stopping arithmetic.

Classical menu: fixed B prefixes 0..16 followed by Watanabe et al. (2007),
with both entropy bounds belonging to that same public protocol. No AD masks.
Fixed source point is an upper relaxation for an unknown BB84 correlation.
"""
from __future__ import annotations
import json,time,argparse,math
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import mpmath as mp
from block_controls import real_graphs
from stopping import *
from bell_core import B,axis,watanabe
ROOT=Path(__file__).resolve().parents[1]

# Interval entropy, with exact rational input conversion.
mp.mp.dps=90  # Preserve interval endpoints exactly when converting to mp.mpf.
mp.iv.dps=32
iv=mp.iv

def ive(x):
    x=F(x);return iv.mpf(x.numerator)/x.denominator

def low(x):return mp.mpf(x._mpi_[0])
def high(x):return mp.mpf(x._mpi_[1])
def ent(p):
    out=iv.mpf(0)
    for x in p:
        if high(x)==0:continue
        if low(x)<=0:raise ArithmeticError('Nonpositive log interval')
        out-=x*iv.log(x)/iv.log(2)
    return out

def wi(p):
    a=p[0]+p[3];b=p[1]+p[2];r1=1-ent(p)
    if high(a*b)>0:
        t=(p[0]*p[1]+p[3]*p[2])/(a*b)
        r1+=a*b*ent([t,1-t])
    u=[p[0]**2+p[3]**2,p[1]**2+p[2]**2,2*p[1]*p[2],2*p[0]*p[3]];s=sum(u)
    r2=s/2*(1-ent([x/s for x in u]))
    return max(mp.mpf(0),high(r1),high(r2))

def classical_float(q,bmax=16,axes=('Z','X')):
    best=0.;which=None
    for a in axes:
        p=[float(x) for x in axis(q,a)];y=1.
        for b in range(bmax+1):
            r=y*watanabe(p)
            if r>best:best=r;which=(a,b)
            p,s=B(p);y*=float(s)/2
    return best,which

def classical_upper(q,bmax=16,axes=('Z','X')):
    """Upper endpoints for the complete named finite menu, no floating pruning."""
    best=mp.mpf(0)
    for a in axes:
        p=[ive(x) for x in axis(q,a)];y=iv.mpf(1)
        for b in range(bmax+1):
            # Bound all remaining classical processing; avoid needless extreme tails.
            source=high(y*iv.mpf(max(mp.mpf(0),2*max(high(x) for x in p)-1))**2)
            if source<mp.mpf('1e-16'):
                best=max(best,source);break
            # Small deviations from probability bounds are addressed by lower precision retry.
            if min(low(x) for x in p)<0:raise ArithmeticError('Negative probability')
            try:r=high(y*iv.mpf(wi(p)))
            except ArithmeticError:
                # A conservative source bound covers zero-mass interval cancellations.
                f=max(high(x) for x in p);r=high(y*iv.mpf(max(0,2*f-1))**2)
            best=max(best,r)
            p0,p1,p2,p3=p
            u=[p0*p0+p3*p3,p1*p1+p2*p2,2*p1*p2,2*p0*p3];s=sum(u)
            p=[x/s for x in u];y*=s/2
    # Decimal rational strictly above the computed interval endpoint.
    scale=10**25
    return F(int(mp.ceil(best*scale))+2,scale)

def run(Q,z=F(0),price=F(1,10000),nmax=5,certify=False):
    p=[1-2*Q+z,Q-z,z,Q-z];count=0;fail=[];maxroot=0.;start=time.monotonic();cache={}
    best_by_Nm={};policy=[]
    for name,adj in real_graphs(nmax):
        n=len(adj)
        for dual in range(1,1<<n):
            weights=branch_weights(adj,dual,p);mass=[sum(w) for w in weights];reward=[]
            for w,P in zip(weights,mass):
                if not P:reward.append(F(0));continue
                q=tuple(x/P for x in w)
                if q not in cache:
                    cache[q]=classical_upper(q) if certify else classical_float(q)[0]
                reward.append(P*cache[q])
            root=stopping_test(adj,dual,mass,reward,price)
            count+=1;maxroot=max(maxroot,float(root))
            if root>0:fail.append({'graph':name,'dual':dual,'root':float(root)})
            if not certify:
                mem=graph_data(adj)[-1]
                for m in range(mem,4):
                    lowr,upr=optimum_stopping(adj,dual,[float(x) for x in mass],[float(x) for x in reward],m,2e-9)
                    key=f'N{n}_m{m}'
                    if key not in best_by_Nm or upr>best_by_Nm[key]['upper']:
                        best_by_Nm[key]={'lower_numerical':lowr,'upper':upr,'graph':name,'adj':adj,'dual':dual}
        print('hashed',name,count,'not excluded',len(fail),'cache',len(cache),'sec',round(time.monotonic()-start,1),flush=True)
    rec={'Q':str(Q),'z':str(z),'price':str(price),'Nmax':nmax,'tested':count,'passed':len(fail)==0,'not_excluded':fail,
         'maxroot':maxroot,'seconds':time.monotonic()-start,'interval_replay':certify,'best_by_Nm':best_by_Nm,'unique_states':len(cache),
         'scope':'complete real graph/H measurements at symmetric BB84 source; all key/check splits; arbitrary public-syndrome-adaptive order and stopping within the complete-measurement compiler model; B=0..16 plus Watanabe; X/Z axes. Fixed known z is an upper relaxation, not an executable z-dependent policy.'}
    (ROOT/'data'/f'hashed_blocks_Q{float(Q):.3f}_{"cert" if certify else "float"}.json').write_text(json.dumps(rec,indent=2))
    print('RESULT',{k:v for k,v in rec.items() if k not in ('best_by_Nm','not_excluded')})
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--Q',default='23/100');a.add_argument('--price',default='1/10000');a.add_argument('--z',default='0');a.add_argument('--certify',action='store_true');a.add_argument('--nmax',type=int,default=5);v=a.parse_args()
    mp.mp.dps=50
    run(F(v.Q),F(v.z),F(v.price),v.nmax,v.certify)
