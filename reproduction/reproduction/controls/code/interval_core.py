"""Continuum lower certificates for TWO fixed BB84 policies, not class optima.

Bisection of all z in [0,11/50], interval arithmetic and interval automatic
differentiation. A finite-grid minimum is never a lower bound. Saved leaf
partitions can be replayed. The implementation uses mpmath.iv; this is not a
machine-checked formal verification of the Python runtime.
"""
from __future__ import annotations
import argparse,json,math,time
from pathlib import Path
from functools import lru_cache
import mpmath as mp
from bell import MAPS
mp.mp.dps=65;mp.iv.dps=35
iv=mp.iv;LN2=iv.log(iv.mpf(2));ZERO=iv.mpf(0)
ROOT=Path(__file__).resolve().parents[1]
def lo(x):return mp.mpf(x._mpi_[0])
def hi(x):return mp.mpf(x._mpi_[1])
def interval(a,b):return iv.mpf([a,b])
def hull(xs):return interval(min(lo(x) for x in xs),max(hi(x) for x in xs))
def clamp_prob(x):
 l=max(mp.mpf(0),lo(x));u=min(mp.mpf(1),hi(x))
 if l>u:raise ArithmeticError('Empty probability range')
 return interval(l,u)
class D:
 __slots__=('v','d')
 def __init__(self,v=0,d=0):self.v=v if hasattr(v,'_mpi_') else iv.mpf(v);self.d=d if hasattr(d,'_mpi_') else iv.mpf(d)
 def __add__(a,b):
  b=asD(b);return D(a.v+b.v,a.d+b.d)
 __radd__=__add__
 def __neg__(a):return D(-a.v,-a.d)
 def __sub__(a,b):return a+-asD(b)
 def __rsub__(a,b):return asD(b)+-a
 def __mul__(a,b):
  b=asD(b);return D(a.v*b.v,a.d*b.v+a.v*b.d)
 __rmul__=__mul__
 def __truediv__(a,b):
  b=asD(b)
  if lo(b.v)<=0<=hi(b.v):raise ArithmeticError('Zero in denominator')
  return D(a.v/b.v,(a.d*b.v-a.v*b.d)/(b.v*b.v))
 def __rtruediv__(a,b):return asD(b)/a
 def __pow__(a,n):
  if n==0:return D(1)
  return D(a.v**n,n*a.v**(n-1)*a.d)
def asD(x):return x if isinstance(x,D) else D(x)
def maximum(*xs):
 xs=[asD(x) for x in xs];l=max(lo(x.v) for x in xs);u=max(hi(x.v) for x in xs)
 return D(interval(l,u),hull([x.d for x in xs if hi(x.v)>=l]))
def xlogx(a):
 a=asD(a);v=clamp_prob(a.v);l=lo(v);u=hi(v)
 if u==0:return D(0,0)
 def point(x):
  if x==0:return ZERO
  xx=iv.mpf(x);return xx*iv.log(xx)/LN2
 ends=[point(l),point(u)];pivot=iv.exp(-iv.mpf(1))
 if l<=hi(pivot) and u>=lo(pivot):ends.append(-pivot/LN2)
 if l==0:dd=interval(-mp.inf,mp.inf) if lo(a.d)!=0 or hi(a.d)!=0 else ZERO
 else:dd=(1+iv.log(v))/LN2*a.d
 return D(hull(ends),dd)
def H(ps):return -sum((xlogx(x) for x in ps),D(0))
def h2(x):return H([x,1-x])
def dec(d,p,r):
 q=[D(0) for _ in range(4)]
 for i,j,k in MAPS[d]:
  if k is not None:q[k]+=p[i]*r[j]
 s=sum(q,D(0));return [x/s for x in q],s
def B(p):
 a,b,c,d=p;q=[a*a+d*d,b*b+c*c,2*b*c,2*a*d];s=sum(q,D(0))
 return [x/s for x in q],s
def W(p):
 a=p[0]+p[3];c=p[1]+p[2]
 first=1-H(p)+a*c*h2((p[0]*p[1]+p[3]*p[2])/(a*c))
 bp,s=B(p);return maximum(0,first,s/2*(1-H(bp)))
@lru_cache(None)
def spc_types(n):
 terms=[]
 for a in range(n+1):
  for b in range(n-a+1):
   for c in range(n-a-b+1):
    t=(a,b,c,n-a-b-c);tt=(t[3],t[2],t[1],t[0])
    if t>tt:continue
    mul=math.factorial(n)//math.prod(math.factorial(v) for v in t)
    terms.append((t,tt,mul,2 if t==tt else 1,(b+c)%2))
 return terms
def spc(p,n):
 powers=[[x**k for k in range(n+1)] for x in p];sums=[D(0),D(0)]
 for t,tt,mul,div,s in spc_types(n):
  a=D(1);b=D(1)
  for j in range(4):a*=powers[j][t[j]];b*=powers[j][tt[j]]
  sums[s]+=(D(mul)/div)*xlogx(a+b)
 e=p[1]+p[2];bias=1-2*e;Ps=[(1+bias**n)/2,(1-bias**n)/2]
 return sum((maximum(0,(n-1)*P+sums[s]-xlogx(P)) for s,P in enumerate(Ps)),D(0))/n
