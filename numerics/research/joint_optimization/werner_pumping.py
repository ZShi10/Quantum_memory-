"""Raw-pair restart pumping, exact retry cost, and short-code AD."""
from pathlib import Path
import json,time
import numpy as np
from rates import CodeKernel,best_codes
from algebra import code_components

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data'

def scan(q,rmax=160,kmax=240):
    r=np.arange(1,rmax+1);A=(1-q)**r;B=q**r;V=(1-2*q)**r;D=A+B
    eps=V/D;phi0=np.clip(B/(D+V),0.,.5);phi1=np.clip(B/(D-V),0.,.5)
    cost=(1+np.r_[0.,np.cumsum(A[:-1]+B[:-1])])/D
    ks=np.arange(kmax+1);N=2.**ks[None,:]
    alpha=np.arctanh(eps)
    bit=np.exp(-2*alpha[:,None]*N);bit/=1+bit
    with np.errstate(divide='ignore'):
        logc0=np.log1p(-2*phi0[:,None])*N;logc1=np.log1p(-2*phi1[:,None])*N
    c0=np.exp(logc0);c1=np.exp(logc1)
    states=np.stack(((1-bit)*(1+c0)/2,bit*(1+c1)/2,bit*(-np.expm1(logc1))/2,(1-bit)*(-np.expm1(logc0))/2),axis=-1)
    ek=np.tanh(alpha[:,None]*N)
    step=(1+ek**2)/4
    pref=np.c_[np.ones(rmax),np.cumprod(step[:,:-1],axis=1)]/cost[:,None]
    return r,ks,eps,phi0,phi1,cost,states,pref

def main():
    codes=[e for e in json.loads((DATA/'codes.json').read_text())['entries'] if len(code_components(e['n'],e['rows']))==1]
    kernels=[CodeKernel(c) for c in codes];output=[];start=time.time()
    for q in [.10,.15,.18,.20,.22,.24,.25,.26,.27,.275,.28,.285,.29,.295,.30,.305,.31,.315,.32,.325,.33,.332]:
        r,ks,eps,p0,p1,cost,states,pref=scan(q)
        result=best_codes(states.reshape(-1,4),kernels)
        rates=result['rate'].reshape(pref.shape)*pref
        endpoint=result['endpoint_rate'].reshape(pref.shape)*pref
        i,j=np.unravel_index(np.argmax(rates),rates.shape);ci=result['code'].reshape(pref.shape)[i,j]
        rec=dict(Q=q,rate=float(rates[i,j]),r=int(r[i]),M=1 if i==0 else 2,binary_rounds=int(ks[j]),code=codes[int(ci)]['id'] if ci>=0 else None,
                 endpoint_rate=float(endpoint.max()),expected_raw_cost=float(cost[i]),direct_rate=float(rates[0].max()))
        # Stable entropy deficits retain extremely small positive repetition
        # witnesses, which must not be lost to a binary64 1-H cancellation.
        best=(0.,None)
        for n in range(1,9):
            # Reuse closed form per k; cumulative yield is already in pref.
            from streaming import repeated_parameters,binary_entropy
            e,d0,d1=repeated_parameters(eps[:,None],p0[:,None],p1[:,None],2.**ks[None,:]*n)
            gain=np.maximum((1-e)*d0+e*d1-binary_entropy(e),0.)
            ek=np.tanh(np.arctanh(eps)[:,None]*2.**ks[None,:])
            final=(((1+ek)/2)**n+((1-ek)/2)**n)/n
            values=pref*final*gain
            ii,jj=np.unravel_index(np.argmax(values),values.shape)
            if values[ii,jj]>best[0]:best=(float(values[ii,jj]),dict(r=int(r[ii]),binary_rounds=int(ks[jj]),final_n=n))
        rec['stable_repetition_rate']=best[0];rec['stable_repetition_choice']=best[1]
        rec['combined_rate']=max(rec['rate'],best[0]);output.append(rec)
        print(rec,'seconds',round(time.time()-start,1),flush=True)
    (DATA/'werner_pumping_joint.json').write_text(json.dumps(dict(rmax=160,kmax=240,nmax=6,results=output),indent=2))

if __name__=='__main__':main()
