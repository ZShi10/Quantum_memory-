"""Directed-rounding certificate for saved long-repetition pilot rules."""
from common import *
from fractions import Fraction as F
from decimal import Decimal as D
import json,time
from interval_certificate import IV,DOWN,UP,LN2,powpoly
from certify_pilot_policy import restrict


def power(x,n):
    out=IV(1);x=IV(x)
    while n:
        if n&1:out=out*x
        n//=2
        if n:x=x*x
    return out


def hpoint(x):
    if x in (D(0),D(1)):return IV(0)
    assert 0<x<1;x=IV(x);y=IV(1)-x
    return -(x*x.ln()+y*y.ln())/LN2


def exact_ratios(q,rule):
    q=F(str(q));parameters=rule['parameters']
    if parameters['family']=='direct':
        raw=[[1-2*q,1-q],[q,F(0)],[F(0),q],[q,F(0)]]
        order=((0,1,2,3),(0,3,2,1),(0,1,3,2))[parameters['axis']]
        a,x,y,z=[raw[i] for i in order]
        plus=lambda a,b:[x+y for x,y in zip(a,b)]
        total=plus(plus(a,x),plus(y,z));num=[a[i]+z[i]-x[i]-y[i] for i in range(2)]
        ratios=[(num,total),(z,plus(a,z)),(y,plus(x,y)),([F(1)]*2,[F(1)]*2)]
    else:
        r=parameters['r'];A=(1-q)**r;B=q**r;den=A+B
        v=powpoly([1-3*q,1-q],r);w=powpoly([q,-q],r);one=[F(1)]*(r+1)
        cost=(1+sum((1-q)**j+q**j for j in range(1,r)))/den
        ratios=[([v[j]+w[j] for j in range(r+1)],[den]*(r+1)),
                ([B+w[j] for j in range(r+1)],[den+v[j]+w[j] for j in range(r+1)]),
                ([B-w[j] for j in range(r+1)],[den-v[j]-w[j] for j in range(r+1)]),
                ([1/cost]*(r+1),one)]
    return [([IV(v) for v in a],[IV(v) for v in b]) for a,b in ratios]


def bound(ratios,a,b,k,n):
    intervals=[]
    for numerator,denominator in ratios:
        numerator=restrict(numerator,a,b);denominator=restrict(denominator,a,b);values=[]
        for x,y in zip(numerator,denominator):
            if y.lo>0:values.append(x/y)
            else:assert y.hi==0 and x.lo==x.hi==0
        intervals.append((min(v.lo for v in values),max(v.hi for v in values)) if values else (D(0),D(0)))
    elo,ehi=intervals[0];elo=max(D(0),elo);ehi=min(D(1),ehi);N=(1<<k)*n
    def bit(epsilon):
        beta=(IV(1)-IV(epsilon))/(IV(1)+IV(epsilon));p=power(beta,N)
        return p/(IV(1)+p)
    emin=bit(ehi).lo;emax=min(D('.5'),bit(elo).hi);deficits=[]
    for lo,hi in intervals[1:3]:
        lo=max(D(0),lo);hi=min(D(1),hi);bias=IV(1)-IV(2)*IV(lo,hi)
        lowbias=bias.lo if bias.lo>0 else bias.hi.copy_negate() if bias.hi<0 else D(0)
        lowc=power(IV(lowbias),N).lo
        phase=UP.divide(UP.subtract(D(1),lowc),D(2))
        deficits.append(max(D(0),DOWN.subtract(D(1),hpoint(phase).hi)))
    gain=(IV(1)-IV(emax))*IV(deficits[0])+IV(emin)*IV(deficits[1])-hpoint(emax)
    eps=IV(elo);factor=IV(1)
    for _ in range(k):
        square=eps*eps;factor=factor*(IV(1)+square)/IV(4)
        eps=IV(2)*eps/(IV(1)+square);eps=IV(max(D(0),eps.lo),min(D(1),eps.hi))
    success=(power((IV(1)+eps)/IV(2),n)+power((IV(1)-eps)/IV(2),n))/IV(n)
    return max(D(0),(IV(max(D(0),intervals[3][0]))*factor*success*gain).lo)


def main():
    started=time.time()
    for path in sorted(DATA.glob('bb84_long_repetition_M*.json')):
        if path.stem.endswith('_certified'):continue
        out=path.with_name(path.stem+'_certified.json')
        if out.exists():continue
        source=json.loads(path.read_text());results=[]
        for record in source['results']:
            if record['lower']<=0:results.append(dict(Q=record['Q'],lower='0'));continue
            target=D(str(record['lower']))*D('.998');minimum=D('Infinity');cache={};cells=[];end=F(0)
            for rule in record['policy']:
                a,b=map(F,rule['interval']);assert a==end;end=b
                if source.get('permissions')=='real':assert rule['parameters']['family']=='pumping' or rule['parameters']['axis'] in (0,1)
                key=rule['ED']
                if key not in cache:cache[key]=exact_ratios(record['Q'],rule)
                pending=[(a,b,0)]
                while pending:
                    a,b,depth=pending.pop();lo=bound(cache[key],a,b,rule['binary_rounds'],rule['final_n'])
                    if lo<target:
                        assert depth<24,(record['Q'],key,lo,target)
                        mid=(a+b)/2;pending.extend([(mid,b,depth+1),(a,mid,depth+1)]);continue
                    minimum=min(minimum,lo);cells.append({**rule,'interval':[str(a),str(b)],'lower':str(lo)})
            assert end==1
            results.append(dict(Q=record['Q'],lower=str(minimum),policy=cells))
            print('Long AD interval',path.name,record['Q'],str(minimum)[:35],'cells',len(cells),'seconds',round(time.time()-started,1),flush=True)
            out.write_text(json.dumps(dict(results=results),separators=(',',':')))
        out.write_text(json.dumps(dict(status='passed',source=path.name,permissions=source.get('permissions','full'),arithmetic='Exact rational ED polynomials, 60-digit outward rounding, integer powers for every AD length; complete pilot-rule coverage.',results=results),separators=(',',':')))


if __name__=='__main__':main()
