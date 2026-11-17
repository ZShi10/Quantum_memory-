"""Independent maps and interval containment checks for long AD."""
from pathlib import Path
import json,sys
import numpy as np
from streaming import pumping,direct,rate

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/'manuscript'/'code'))
from protocols import bstep,repeat,hashing

def main():
    rng=np.random.default_rng(24791);error=0.;checks=0
    for _ in range(400):
        eb,ep=rng.uniform(.015,.24,2);r=int(rng.integers(2,20));axis=int(rng.integers(2))
        ed=pumping(eb,ep,r,axis);t=float(rng.random());z=t*min(eb,ep)
        p=np.array([1-eb-ep+z,eb-z,z,ep-z])
        if axis:p=p[[0,3,2,1]]
        q=p.copy()
        for j in range(r-1):q,_=bstep(q,p)
        q=q[[0,3,2,1]]
        eps,ph0,ph1,ye=ed.values(t)
        expected=np.array([q[0]+q[3]-q[1]-q[2],q[3]/(q[0]+q[3]),q[2]/(q[1]+q[2])])
        assert np.max(abs(np.array([eps,ph0,ph1])-expected))<3e-13
        k=int(rng.integers(0,9));n=int(rng.integers(1,9));state=q.copy();y=1.
        for j in range(k):state,s=bstep(state);y*=s/2
        state,ls=repeat(state,n)
        standard=y*np.exp(ls)/n*hashing(state)
        stable=float(rate(eps,ph0,ph1,k,n))
        error=max(error,abs(standard-stable));assert abs(standard-stable)<2e-13
        lower=ed.lower(ed.ratios,k,n)
        actual=min(ed.value(t,k,n) for t in np.linspace(0,1,41))
        assert lower<=actual+1e-14,(eb,ep,r,k,n,lower,actual)
        checks+=1
    result=dict(status='passed',random_map_and_interval_checks=checks,max_rate_discrepancy=error)
    (ROOT/'data'/'streaming_verification.json').write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':main()
