"""60-digit outward certificates for complete-library Werner winners."""
from common import *
from fractions import Fraction as F
from decimal import Decimal as D
from math import factorial
import json,time
import numpy as np
from interval_certificate import IV,LN2,DOWN
from prove_single_round_counterexample import ivpower,positive
from libraries import code_kernels
from algebra import compositions


def xlog(x):
    if x.hi<=0:return IV(0)
    if x.lo>0:return x*x.ln()/LN2
    top=IV(x.hi);v=top*top.ln()/LN2
    low=v.lo if x.hi<D('0.3678794411714422') else D('-0.530737845423044')
    return IV(low,max(D(0),v.hi))


def kernel_rate(p,kernel):
    powers=[]
    cache=[[ivpower(x,n) for n in range(kernel.n+1)] for x in p]
    for typ in kernel.types:
        value=IV(1)
        for j,n in enumerate(typ):value=value*cache[j][int(n)]
        powers.append(value)
    joint=[sum((int(c)*v for c,v in zip(row,powers) if c),IV(0)) for row in kernel.polys]
    logs=[xlog(v) for v in joint];rate=IV(0)
    for hist,mult in zip(kernel.hist,kernel.multiplicity):
        ps=sum((int(c)*v for c,v in zip(hist,joint) if c),IV(0))
        gain=kernel.k*ps+sum((int(c)*v for c,v in zip(hist,logs) if c),IV(0))-xlog(ps)
        rate=rate+int(mult)*positive(gain)
    return rate/IV(kernel.n)


def spc_rate(p,n):
    powers=[[ivpower(x,j) for j in range(n+1)] for x in p]
    facts=[factorial(j) for j in range(n+1)];ps=[IV(0),IV(0)];logs=[IV(0),IV(0)]
    for typ in compositions(n):
        flip=tuple(reversed(typ))
        if typ>flip:continue
        a=b=IV(1);den=1
        for j,t in enumerate(typ):a=a*powers[j][t];b=b*powers[3-j][t];den*=facts[t]
        w=a+b;coefficient=IV(F(facts[n]//den,2 if typ==flip else 1));s=(typ[1]+typ[2])%2
        ps[s]=ps[s]+coefficient*w;logs[s]=logs[s]+coefficient*xlog(w)
    return sum((positive((n-1)*s+l-xlog(s)) for s,l in zip(ps,logs)),IV(0))/IV(n)


def repetition(p,n):
    a=ivpower(p[0]+p[3],n);b=ivpower(p[1]+p[2],n);u=ivpower(p[0]-p[3],n);v=ivpower(p[1]-p[2],n)
    raw=[(a+u)/IV(2),(b+v)/IV(2),(b-v)/IV(2),(a-u)/IV(2)]
    raw=[IV(max(D(0),x.lo),max(D(0),x.hi)) for x in raw];ps=a+b
    return positive(ps+sum((xlog(x) for x in raw),IV(0))-xlog(ps))/IV(n)


def witness(record,entry,weights,kernels):
    q=F(str(record['Q']));m=entry['m'];powers=[(1-F(3,2)*q)**(m-j)*(q/2)**j for j in range(m+1)]
    total=IV(0);branches=[]
    for b in record['branches']:
        if b['contribution']<=0:continue
        counts=weights[m][b['state']];raw=[sum((int(c)*v for c,v in zip(row,powers)),F(0)) for row in counts]
        ps=sum(raw);state=[IV(v/ps) for v in raw];factor=IV(ps*F(b['factor'],m))
        if b.get('canonicalize_bit') and float(state[1].lo+state[2].lo)>.5:state=[state[j] for j in (1,0,3,2)]
        for _ in range(b['binary_rounds']):
            a,x,y,z=state;out=[a*a+z*z,x*x+y*y,2*x*y,2*a*z];mass=sum(out,IV(0))
            state=[v/mass for v in out];factor=factor*mass/IV(2)
        kind=b.get('kind','linear')
        if kind=='linear':r=kernel_rate(state,kernels[b['code']])
        elif kind=='SPC':r=spc_rate(state,b['n'])
        else:r=repetition(state,b['n'])
        value=factor*r;total=total+value;branches.append({**b,'certified_lower':str(value.lo),'certified_upper':str(value.hi)})
        assert abs(float(value.lo)-b['contribution'])<=max(5e-16,abs(b['contribution'])*2e-6),(record['Q'],record['ED'],b['syndrome'],str(value.lo),b['contribution'])
    return dict(Q=record['Q'],ED=record['ED'],m=m,M=entry['M'],lower=str(total.lo),upper=str(total.hi),branches=branches)


def main():
    data=json.loads((DATA/'werner_complete_library.json').read_text());entries={e['id']:e for e in data['entries']}
    weights={m:np.load(DATA/f'werner_states_m{m}.npz')['weights'] for m in range(1,9)};kernels={k.entry['id']:k for k in code_kernels()}
    dest=DATA/'werner_witness_certificates.json';saved=json.loads(dest.read_text()) if dest.exists() else dict(results={})
    started=time.time()
    for path in [DATA/'werner_complete_k16.json']+sorted(DATA.glob('werner_complete_long*.json')):
        src=json.loads(path.read_text())
        for item in src['results']:
            for mode,r in item['results'].items():
                key=json.dumps([r['Q'],r['ED'],r['branches']],sort_keys=True)
                if key in saved['results'] or r['rate']<=0:continue
                v=witness(r,entries[r['ED']],weights,kernels);v['source']=path.name;saved['results'][key]=v
                saved.update(status='passed',arithmetic='Exact rational raw weights; 60-digit outward-rounded Decimal throughout ED/B/code entropy and all success factors.',seconds=time.time()-started)
                dest.write_text(json.dumps(saved,separators=(',',':')))
                print('Certified Werner witness',r['Q'],r['ED'],str(v['lower'])[:32],'total',len(saved['results']),flush=True)


if __name__=='__main__':main()
