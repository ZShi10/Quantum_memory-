"""Independent finite checks. Full original search upper bounds are not rerun."""
from __future__ import annotations
import argparse,json,itertools,sys,time
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from stopping import graph_data,branch_weights,zero_flag_cost,public_partitions,stopping_test
from bell_core import decoder,axis,B,entropy
ROOT=Path(__file__).resolve().parents[1]

def direct(adj,v,p):
 n=len(adj);out=[[F(0)]*4 for _ in range(1<<(n-1))]
 k=(v&-v).bit_length()-1;others=[i for i in range(n) if i!=k]
 bits=[(0,0),(1,0),(1,1),(0,1)];idx={(0,0):0,(1,0):1,(1,1):2,(0,1):3}
 for errors in itertools.product(range(4),repeat=n):
  x=sum(bits[e][0]<<i for i,e in enumerate(errors));z=sum(bits[e][1]<<i for i,e in enumerate(errors))
  w=F(1)
  for e in errors:w*=p[e]
  y=z
  for i in range(n):y^=((adj[i]&x).bit_count()%2)<<i
  bit=(y>>k)&1;phase=(v&x).bit_count()%2
  s=sum((((y>>i)&1)^(((v>>i)&1)*bit))<<j for j,i in enumerate(others))
  out[s][idx[bit,phase]]+=w
 return out

def retained_mem(depth):return depth+1

def terminal_mem(word,ax):
 if not word:return 1
 d=word[-1];child=word[:-1]
 if ax=='Z' and d=='Z':return terminal_mem(child,'Z')
 if ax=='Z' and d=='X':return terminal_mem(child,'X')
 if ax=='Z' and d=='C':return max(terminal_mem(child,'X'),terminal_mem(child,'Z'))
 return len(word)+1

def run():
 start=time.monotonic();catalog=json.loads((ROOT/'data/graph_representatives.json').read_text())
 hist={};checked=0
 for g in catalog['graphs'].values():
  n=g['N'];a=[0]*n
  for i,j in g['edges']:a[i]|=1<<j;a[j]|=1<<i
  m=graph_data(a)[-1];assert m==g['original_min_memory'];checked+=1
  hist.setdefault(str(n),{}).setdefault(str(m),0);hist[str(n)][str(m)]+=1
 adj=[18,5,10,4,1];v=5;p=[F(7,10),F(1,10),F(1,10),F(1,10)]
 assert branch_weights(adj,v,p)==direct(adj,v,p)
 costs={m:zero_flag_cost(adj,v,p,m) for m in (2,3)}
 assert costs[2][0]==F(2975,173) and costs[3][0]==F(11425,692)
 mass=[sum(w) for w in branch_weights(adj,v,p)];parts,cuts,_=public_partitions(adj,v)
 surv=[sum(mass[s] for s in next(g for g in ps if 0 in g)) for ps in parts]
 brute={2:None,3:None}
 for order in itertools.permutations(range(5)):
  a=0;d=F(0);width=0
  for i in order:d+=surv[a];width=max(width,cuts[a]);a|=1<<i
  c=d/mass[0]
  for m in (2,3):
   if width<=m-1 and (brute[m] is None or c<brute[m]):brute[m]=c
 assert all(brute[m]==costs[m][0] for m in brute)
 # General full-flag Bell laws, not only zero syndrome or Werner symmetry.
 cases=0
 for a in ([2,1],[2,5,2],[6,5,3],[2,5,10,4],adj):
  n=len(a)
  for dual in (1,(1<<n)-1):
   pp=[F(11,20),F(1,5),F(1,10),F(3,20)]
   w=branch_weights(a,dual,pp);assert w==direct(a,dual,pp)
   assert sum(map(sum,w))==1;cases+=1
 # All 4^8 homogeneous words and both real terminal axes: bounded non-Z prefix.
 words=0
 for tup in itertools.product('ZXRC',repeat=8):
  word=''.join(tup)
  for ax in ('Z','X'):
   m=terminal_mem(word,ax)
   if m<=7:
    prefix=word.rstrip('Z') if ax=='Z' else word
    assert len(prefix)<=m<=7
   words+=1
 # Check conversion of terminal Z layers to pooled classical B, exactly.
 conversion=0;p=[F(13,25),F(6,25),F(0),F(6,25)]
 for word in ('R','RR','X','RC','RRZRX','RRZRR'):
  q=p;c=F(1)
  for d in word:q,s=decoder(d,q,q);c=2*c/s
  for _ in range(4):
   qe,se=decoder('Z',q,q);qb,sb=B(q)
   assert qe==qb and se==sb
   ce=2*c/se;assert F(1)/ce==F(1)/c*sb/2
   q,c=qe,ce;conversion+=1
 result=dict(status='passed',graph_representatives=checked,graph_memory_histogram=hist,
  direct_full_Bell_checks=cases+1,pauli_assignments_five_input=4**5,orders_five_input=120,
  early_stopping_costs={str(m):dict(cost=str(c),order=o) for m,(c,o) in costs.items()},
  homogeneous_memory_cases=words,exact_B_conversion_checks=conversion,
  runtime_seconds=time.monotonic()-start,
  boundary='Exact tests of new arithmetic and structural formulas; supplied class upper bounds remain source evidence, not independently rerun search results.')
 (ROOT/'validation/independent_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--intervals',action='store_true');a=parser.parse_args();run()
 if a.intervals:
  from interval_rates import replay
  for f in sorted((ROOT/'data').glob('*_lower_certificate.json')):replay(f)
