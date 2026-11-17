"""Interval witnesses against ALL single-round repetition/SPC lengths.

The input Bell state is fixed. Changing a preceding AD stage changes that
input and is not part of this comparison. Infinite tails have analytic bounds.
"""
from pathlib import Path
from fractions import Fraction as F
from math import factorial
import json,time
from algebra import compositions
from rates import CodeKernel
from interval_certificate import IV,LN2

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data'

def ivpower(x,n):
    out=IV(1)
    while n:
        if n&1:out=out*x
        x=x*x;n//=2
    return out

def xlog(x):return x*x.ln()/LN2 if x.hi>0 else IV(0)
def positive(x):return IV(max(0,x.lo),max(0,x.hi))

def kernel_rate(p,n,rows):
    kernel=CodeKernel(dict(n=n,k=len(rows),rows=rows))
    powers=[]
    for typ in compositions(n):
        v=F(1)
        for x,r in zip(p,typ):v*=x**r
        powers.append(v)
    joint=[IV(sum(int(c)*v for c,v in zip(row,powers) if c)) for row in kernel.polys]
    logs=[xlog(v) for v in joint];rate=IV(0)
    for hist,mult in zip(kernel.hist,kernel.multiplicity):
        ps=sum((int(c)*v for c,v in zip(hist,joint) if c),IV(0))
        if ps.hi==0:continue
        gain=len(rows)*ps+sum((int(c)*v for c,v in zip(hist,logs) if c),IV(0))-xlog(ps)
        rate=rate+int(mult)*positive(gain)
    return rate/IV(n)

def spc_rate(p,n):
    powers=[[IV(x**j) for j in range(n+1)] for x in p]
    facts=[factorial(j) for j in range(n+1)];hmass=[IV(0),IV(0)]
    for typ in compositions(n):
        flip=tuple(reversed(typ))
        if typ>flip:continue
        a=b=IV(1);den=1
        for j,t in enumerate(typ):a=a*powers[j][t];b=b*powers[3-j][t];den*=facts[t]
        w=a+b
        if w.hi==0:continue
        coefficient=F(facts[n]//den,2 if typ==flip else 1)
        s=(typ[1]+typ[2])%2;hmass[s]=hmass[s]-IV(coefficient)*xlog(w)
    d=1-2*(p[1]+p[2]);ps=[IV((1+d**n)/2),IV((1-d**n)/2)]
    return sum((positive((n-1)*s-h-xlog(s)) for s,h in zip(ps,hmass) if s.hi>0),IV(0))/IV(n)

def fixed_input(w,blocks):
    q=F(w['Q']);p=(1-F(3,2)*q,q/2,q/2,q/2);ed=blocks[w['ED']]
    powers=[]
    for typ in compositions(ed['m']):
        v=F(1)
        for x,r in zip(p,typ):v*=x**r
        powers.append(v)
    raw=[sum(int(c)*v for c,v in zip(row,powers)) for row in ed['tensor'][0]]
    p=tuple(v/sum(raw) for v in raw)
    for _ in range(w['binary_rounds']):
        a,x,y,z=p;out=(a*a+z*z,x*x+y*y,2*x*y,2*a*z);p=tuple(v/sum(out) for v in out)
    return p

def main():
    blocks={e['id']:e for e in json.loads((DATA/'blocks.json').read_text())['entries']}
    witnesses=json.loads((DATA/'code_witnesses_130digit.json').read_text());results=[];start=time.time()
    for w in witnesses:
        p=fixed_input(w,blocks);c=w['code'];chosen=kernel_rate(p,c['n'],c['rows'])
        rep=[]
        for n in range(1,8):
            value=kernel_rate(p,n,[(1<<n)-1]);rep.append(dict(n=n,lower=str(value.lo),upper=str(value.hi)))
        a=p[0]+p[3];b=p[1]+p[2];c0=abs((p[0]-p[3])/a);c1=abs((p[1]-p[2])/b)
        lam=a*c0*c0+b*c1*c1;cmax=max(c0,c1);Nrep=8
        repetition_tail=IV(lam**Nrep)/(IV(2*Nrep)*LN2*IV(1-cmax**(2*Nrep)))
        H=-sum((xlog(IV(v)) for v in p),IV(0))
        Hsigned=-xlog(IV(p[0]))+xlog(IV(p[1]))+xlog(IV(p[2]))-xlog(IV(p[3]))
        habs=max(Hsigned.lo.copy_abs(),Hsigned.hi.copy_abs());d=1-2*b;Nspc=64
        physical_entropy_lower=(H-IV(d**(Nspc-1))*IV(habs))/IV(1+d**Nspc)+IV((1-d**Nspc)/2).ln()/(LN2*IV(Nspc))
        assert physical_entropy_lower.lo>1
        spc=[]
        for n in range(2,Nspc):
            value=spc_rate(p,n);spc.append(dict(n=n,lower=str(value.lo),upper=str(value.hi)))
            if n<=6:
                independent=kernel_rate(p,n,[(1<<j)|(1<<(n-1)) for j in range(n-1)])
                assert independent.lo<=value.hi and value.lo<=independent.hi
            if n%8==0:print(c['id'],'SPC n',n,'seconds',round(time.time()-start,1),flush=True)
        endpoint_upper=max([repetition_tail.hi]+[IV(v['upper']).hi for v in rep+spc])
        assert chosen.lo>endpoint_upper
        result=dict(code=c,input_exact=[str(v) for v in p],rate_lower=str(chosen.lo),rate_upper=str(chosen.hi),
                    all_single_round_endpoint_rate_upper=str(endpoint_upper),
                    repetition_finite=rep,repetition_tail_start=Nrep,repetition_tail_rate_upper=str(repetition_tail.hi),
                    SPC_finite=spc,SPC_tail_start=Nspc,SPC_tail_conditional_entropy_per_input_lower=str(physical_entropy_lower.lo),
                    scope='Fixed Bell input, one linear-code AD round, all syndrome-dependent retention allowed. All repetition and SPC lengths are covered. Arbitrary preceding AD transformations are not covered.')
        results.append(result)
        (DATA/'single_round_counterexamples_interval.json').write_text(json.dumps(dict(status='passed',results=results),indent=2))
        print('CERTIFIED',c['id'],chosen.lo,endpoint_upper,flush=True)

if __name__=='__main__':main()
