"""Give every two-register restart pump the same enlarged AD family."""
from common import *
import json,time
import numpy as np
from libraries import code_kernels
from rates import best_codes
from streaming import rate as rep
from spc_dynamic import all_rates
from complete_werner_long import spc_upper_array


def main():
    ks=code_kernels();rs=np.arange(1,161);ns=np.arange(8,129);nr=np.arange(1,129);dest=DATA/'pumping_fair_codes.json'
    saved=json.loads(dest.read_text()) if dest.exists() else dict(results=[]);done={r['Q'] for r in saved['results']};started=time.time()
    for q in [.10,.15,.18,.20,.21,.22,.23,.24,.25,.26,.27,.275,.28,.285,.29,.295,.30,.305,.31,.315,.32]:
        if q in done:continue
        A=(1-q)**rs;B=q**rs;V=(1-2*q)**rs;den=A+B
        state=np.stack((A+V,A-V,B,B),-1)/(2*den[:,None]);cost=(1+np.concatenate(([0.],np.cumsum(den[:-1]))))/den
        weight=1/cost;factor=np.ones(len(rs));best=np.zeros(len(rs));choices=[None]*len(rs)
        for k in range(17):
            bound=weight*factor*np.maximum(0,2*state.max(1)-1)**2
            threshold=np.full(len(rs),best[1:].max());threshold[0]=best[0]
            ids=np.flatnonzero(bound>threshold+1e-20)
            if not len(ids):break
            p=state[ids].copy();flip=p[:,1]+p[:,2]>.5;p[flip]=p[flip][:,[1,0,3,2]]
            values=best_codes(p,ks);rr=values['rate']*weight[ids]*factor[ids]
            for j in np.flatnonzero(rr>best[ids]):
                i=ids[j];best[i]=rr[j];choices[i]=dict(kind='linear',code=ks[values['code'][j]].entry['id'],binary_rounds=k)
            a=p[:,0]+p[:,3];b=p[:,1]+p[:,2]
            values=rep(np.broadcast_to((a-b)[:,None],(len(ids),len(nr))),p[:,3,None]/a[:,None],p[:,2,None]/b[:,None],0,nr[None,:])*weight[ids,None]*factor[ids,None]
            at=values.argmax(1);rr=values[np.arange(len(ids)),at]
            for j in np.flatnonzero(rr>best[ids]+1e-20):
                i=ids[j];best[i]=rr[j];choices[i]=dict(kind='repetition',n=int(nr[at[j]]),binary_rounds=k)
            mask=spc_upper_array(p,ns)*weight[ids,None]*factor[ids,None]>best[ids,None]+1e-20
            values=all_rates(p,ns,mask)*weight[ids,None]*factor[ids,None];at=values.argmax(1);rr=values[np.arange(len(ids)),at]
            for j in np.flatnonzero(rr>best[ids]+1e-20):
                i=ids[j];best[i]=rr[j];choices[i]=dict(kind='SPC',n=int(ns[at[j]]),binary_rounds=k)
            a,x,y,z=state.T;raw=np.stack((a*a+z*z,x*x+y*y,2*x*y,2*a*z),-1);mass=raw.sum(1);factor*=mass/2;state=raw/mass[:,None]
        result={}
        for M in (1,2):
            i=0 if M==1 else int(best.argmax());branches=[]
            if choices[i]:branches=[dict(**choices[i],syndrome=0,canonicalize_bit=True,contribution=float(best[i]))]
            result[f'M{M}']=dict(Q=q,ED=f'restart_pump_{rs[i]}',m=int(rs[i]),M=1 if i==0 else 2,expected_raw_cost=float(cost[i]),rate=float(best[i]),branches=branches)
        saved['results'].append(dict(Q=q,results=result));saved['scope']='Every restart raw-pair pump success path r=1..160, final H, same all-code n<=7 plus every repetition/SPC n<=128, pooled B prefixes<=16. No ED shortlist. All failures included through exact analytic expected cost. Numerical search, selected protocols independently certified.'
        dest.write_text(json.dumps(saved,separators=(',',':')))
        print('Fair pump codes',q,result['M2']['rate'],result['M2']['ED'],result['M2']['branches'],'seconds',round(time.time()-started,1),flush=True)
    saved['status']='passed';dest.write_text(json.dumps(saved,separators=(',',':')))


if __name__=='__main__':main()
