"""Directed-rounding certificates for selected fixed BB84 strategies.

All probability polynomials and time-sharing weights start as exact rational
numbers. Decimal arithmetic is rounded outwards at 60 digits. Decimal.ln is
correctly rounded; adjacent representable values enclose its exact result.
This verifies achievability, not the floating-point search's global upper bound.
"""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal as D,Context,ROUND_FLOOR,ROUND_CEILING
from math import comb
import heapq,json,argparse,time
import numpy as np
from algebra import compositions
from rates import CodeKernel

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data'
DOWN=Context(prec=60,rounding=ROUND_FLOOR);UP=Context(prec=60,rounding=ROUND_CEILING)

class IV:
    __slots__=('lo','hi')
    def __init__(self,x,hi=None):
        if isinstance(x,IV):self.lo,self.hi=x.lo,x.hi;return
        if isinstance(x,F):
            self.lo=DOWN.divide(D(x.numerator),D(x.denominator));self.hi=UP.divide(D(x.numerator),D(x.denominator))
        else:self.lo=D(x);self.hi=D(x) if hi is None else D(hi)
    def __add__(self,o):
        o=IV(o);return IV(DOWN.add(self.lo,o.lo),UP.add(self.hi,o.hi))
    __radd__=__add__
    def __neg__(self):return IV(self.hi.copy_negate(),self.lo.copy_negate())
    def __sub__(self,o):return self+-IV(o)
    def __mul__(self,o):
        o=IV(o)
        if self.lo>=0 and o.lo>=0:return IV(DOWN.multiply(self.lo,o.lo),UP.multiply(self.hi,o.hi))
        if self.hi<=0 and o.hi<=0:return IV(DOWN.multiply(self.hi,o.hi),UP.multiply(self.lo,o.lo))
        if self.lo>=0 and o.hi<=0:return IV(DOWN.multiply(self.hi,o.lo),UP.multiply(self.lo,o.hi))
        if self.hi<=0 and o.lo>=0:return IV(DOWN.multiply(self.lo,o.hi),UP.multiply(self.hi,o.lo))
        lows=[DOWN.multiply(a,b) for a in (self.lo,self.hi) for b in (o.lo,o.hi)]
        highs=[UP.multiply(a,b) for a in (self.lo,self.hi) for b in (o.lo,o.hi)]
        return IV(min(lows),max(highs))
    __rmul__=__mul__
    def __truediv__(self,o):
        o=IV(o);assert o.lo>0
        return self*IV(DOWN.divide(D(1),o.hi),UP.divide(D(1),o.lo))
    def ln(self):
        assert self.lo>0
        return IV(DOWN.next_minus(DOWN.ln(self.lo)),UP.next_plus(UP.ln(self.hi)))

LN2=IV(2).ln()

def mul(a,b):
    n=len(a)-1;m=len(b)-1;out=[F(0)]*(n+m+1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):out[i+j]+=x*y*F(comb(n,i)*comb(m,j),comb(n+m,i+j))
    return out

def powpoly(a,n):
    out=[F(1)]
    for _ in range(n):out=mul(out,a)
    return out

def mono(components,types):
    n=sum(types[0]);cache={(i,r):powpoly(components[i],r) for i in range(4) for r in range(n+1)}
    out=[]
    for typ in types:
        row=[F(1)]
        for i,r in enumerate(typ):row=mul(row,cache[i,int(r)])
        out.append(row)
    return out

def matpoly(matrix,polys):
    return [[sum(int(c)*p[j] for c,p in zip(row,polys) if c) for j in range(len(polys[0]))] for row in matrix]

def midpoint(coeff):
    n=len(coeff)-1
    return sum((v*IV(F(comb(n,i),2**n)) for i,v in enumerate(coeff)),IV(0))

def splitpoly(coeff):
    work=coeff[:];left=[work[0]];right=[work[-1]]
    while len(work)>1:
        work=[(a+b)*IV(F(1,2)) for a,b in zip(work,work[1:])]
        left.append(work[0]);right.append(work[-1])
    return left,list(reversed(right))

class Curve:
    def __init__(self,raw,kernel,keep,scale):
        joint=matpoly(kernel.polys,mono(raw,kernel.types))
        hist=kernel.hist[keep].astype(int);mult=kernel.multiplicity[keep].astype(int)
        used=np.flatnonzero(hist.sum(0));self.hist=hist[:,used].tolist();self.mult=list(map(int,mult))
        self.joint=[[IV(v) for v in joint[j]] for j in used]
        ps=[sum(v) for v in zip(*raw)]
        self.den=[IV(v) for v in mul(powpoly(ps,kernel.n-1),[F(1)]*len(ps))]
        self.scale=IV(scale/kernel.n);self.k=kernel.k

    def bound(self,joint,den):
        mid=[midpoint(v) for v in joint];gradient=[IV(0) for _ in mid]
        value=IV(0)
        for hist,mult in zip(self.hist,self.mult):
            total=sum((mid[j]*c for j,c in enumerate(hist) if c),IV(0))
            if total.hi==0:continue
            logtotal=total.ln()/LN2
            for j,c in enumerate(hist):
                if not c or mid[j].hi==0:continue
                g=(IV(self.k)+mid[j].ln()/LN2-logtotal)*(c*mult)
                gradient[j]=gradient[j]+g;value=value+mid[j]*g
        coeff=[sum((g*row[i] for g,row in zip(gradient,joint)),IV(0))*self.scale for i in range(len(den))]
        # At a zero-probability endpoint both polynomials can have zero
        # Bernstein coefficients. Exact zero/zero terms carry no weight in
        # the ratio bound; a nonzero numerator there must never be ignored.
        for a,b in zip(coeff,den):
            if b.lo<=0:assert b.lo==b.hi==a.lo==a.hi==0
        lower=min((a/b).lo for a,b in zip(coeff,den) if b.lo>0)
        middle=value*self.scale/midpoint(den)
        return lower,middle

def construct(eb,ep,record,blocks,codes):
    entry=blocks[record['ED']];s=eb+ep;t=min(eb,ep)
    components=[[1-s,1-s+t],[eb,eb-t],[F(0),t],[ep,ep-t]]
    physical=mono(components,compositions(entry['m']))
    raw=[matpoly(branch,physical) for branch in entry['tensor']]
    weights=[F(str(v['input_fraction'])) for v in record['strategy']]
    for syndrome in {v['syndrome'] for v in record['strategy']}:
        ids=[j for j,v in enumerate(record['strategy']) if v['syndrome']==syndrome]
        total=sum(weights[j] for j in ids)
        if total>1:
            for j in ids:weights[j]/=total
    curves=[]
    for item,w in zip(record['strategy'],weights):
        kernel=CodeKernel(codes[item['code']])
        curves.append(Curve(raw[item['syndrome']],kernel,item['keep_classes'],w/entry['m']))
    return curves,weights

def certify(curves,atol='3e-8',max_cells=15000):
    counter=0;heap=[];incumbent=D('Infinity');argmin=None
    def add(a,b,arrays):
        nonlocal counter,incumbent,argmin
        lower=D(0);value=IV(0)
        for curve,(joint,den) in zip(curves,arrays):
            lo,mid=curve.bound(joint,den);lower=DOWN.add(lower,lo);value=value+mid
        if value.hi<incumbent:incumbent,argmin=value.hi,(a+b)/2
        heapq.heappush(heap,(lower,counter,a,b,arrays));counter+=1
    add(0.,1.,[(c.joint,c.den) for c in curves])
    while counter<max_cells and UP.subtract(incumbent,heap[0][0])>D(atol):
        _,_,a,b,arrays=heapq.heappop(heap)
        children=[]
        for joint,den in arrays:
            pairs=[splitpoly(row) for row in joint];dl,dr=splitpoly(den)
            children.append((([p[0] for p in pairs],dl),([p[1] for p in pairs],dr)))
        add(a,(a+b)/2,[pair[0] for pair in children]);add((a+b)/2,b,[pair[1] for pair in children])
    return dict(lower=str(heap[0][0]),upper=str(incumbent),gap=str(UP.subtract(incumbent,heap[0][0])),argmin=argmin,cells=counter,converged=UP.subtract(incumbent,heap[0][0])<=D(atol))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--eb',default='0.20');parser.add_argument('--ep')
    parser.add_argument('--modes',nargs='+',default=['real_M2','full_M1','full_M2'])
    parser.add_argument('--tolerance',default='3e-8');args=parser.parse_args()
    ep=args.ep or args.eb;ebq,epq=F(args.eb),F(ep)
    data=json.loads((DATA/f'bb84_{float(ebq):.5f}_{float(epq):.5f}.json').read_text())
    blocks={e['id']:e for e in json.loads((DATA/'blocks.json').read_text())['entries']}
    codes={e['id']:e for e in json.loads((DATA/'codes.json').read_text())['entries']}
    result=dict(eb=args.eb,ep=ep,precision=60,arithmetic='Exact rational coefficients, outward-rounded Decimal intervals; correctly rounded logarithms enclosed by adjacent values.',results={})
    started=time.time()
    for mode in args.modes:
        record=data['results'][mode]
        if record['ED'] is None:continue
        curves,weights=construct(ebq,epq,record,blocks,codes)
        cert=certify(curves,atol=args.tolerance);cert['ED']=record['ED'];cert['exact_weights']=[str(w) for w in weights]
        result['results'][mode]=cert
        print(mode,cert,'seconds',round(time.time()-started,1),flush=True)
        (DATA/f'bb84_interval_{float(ebq):.5f}_{float(epq):.5f}.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
