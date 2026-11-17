"""Construct and independently check a signed four-qubit sequential readout.

Gate wire labels refer to graph inputs; `slot` fields show reused physical slots.
All intermediate records stay private. The release rule announces only original
public-check combinations; the full final affine map reconstructs the key.
"""
from fractions import Fraction as F
from pathlib import Path
import itertools,json,math
import numpy as np
import mpmath as mp
from fixed_block_scan import rank_binary
ROOT=Path(__file__).resolve().parents[1]

def multiply(a,b):
 x,z,s,c=a;X,Z,S,C=b
 e=((x&z).bit_count()+(X&Z).bit_count()-((x^X)&(z^Z)).bit_count()+2*(z&X).bit_count())%4
 if e%2:raise ValueError('Noncommuting product')
 return (x^X,z^Z,s^S^(e//2),c^C)

def gate(row,g):
 x,z,s,c=row
 if g[0]=='h':
  i=g[1];xi=(x>>i)&1;zi=(z>>i)&1;s^=xi&zi
  if xi!=zi:x^=1<<i;z^=1<<i
 else:
  i,j=g[1:];xc=(x>>i)&1;xt=(x>>j)&1;zc=(z>>i)&1;zt=(z>>j)&1
  s^=xc&zt&(xt^zc^1)
  if xc:x^=1<<j
  if zt:z^=1<<i
 return x,z,s,c

def compile_order(adj,order):
 n=len(adj);rows=[(1<<i,adj[i],0,1<<i) for i in range(n)];live=[];slots={};events=[];gates=[];measurements=[];peak=0
 for v in order:
  slot=next(i for i in range(n) if i not in slots.values());slots[v]=slot;live.append(v);peak=max(peak,len(live));events.append({'action':'load','input':v,'slot':slot})
  while rows:
   lm=sum(1<<i for i in live);choice=None
   for bits in range(1,1<<len(rows)):
    a=(0,0,0,0)
    for j,row in enumerate(rows):
     if (bits>>j)&1:a=multiply(a,row)
    if not ((a[0]|a[1])&~lm):choice=(bits,a);break
   if choice is None:break
   bits,a=choice;pivot=(bits&-bits).bit_length()-1;rows[pivot]=a
   def apply(g):
    nonlocal rows
    rows=[gate(row,g) for row in rows];gates.append(g)
    events.append({'action':'gate','gate':g,'slots':[slots[i] for i in g[1:]]})
   while (rows[pivot][0]&rows[pivot][1]).bit_count():
    ys=[i for i in live if (rows[pivot][0]&rows[pivot][1])>>i&1]
    assert len(ys)%2==0;apply(('cx',ys[0],ys[1]))
   for i in live:
    if rows[pivot][0]>>i&1:apply(('h',i))
   zs=[i for i in live if rows[pivot][1]>>i&1];q=zs[0]
   for i in zs[1:]:apply(('cx',i,q))
   a=rows[pivot];assert a[0]==0 and a[1]==1<<q
   meas={'wire':q,'slot':slots[q],'original_generator_mask':a[3],'offset':a[2]}
   measurements.append(meas);events.append({'action':'measure_Z',**meas})
   for j in range(len(rows)):
    if j!=pivot and rows[j][1]>>q&1:rows[j]=multiply(rows[j],a)
   rows.pop(pivot);live.remove(q);del slots[q]
 assert not rows and len(measurements)==n
 return {'peak_memory':peak,'input_order':list(order),'gates':gates,'events':events,'measurements':measurements}

def pauli_matrix(row,n):
 x,z,s,_=row;dim=1<<n;out=np.zeros((dim,dim),complex)
 phase=(-1)**s*(1j)**((x&z).bit_count())
 for j in range(dim):out[j^x,j]=phase*(-1)**((z&j).bit_count())
 return out

def dense_check(adj,comp):
 n=len(adj);dim=1<<n;U=np.eye(dim,dtype=complex)
 for g in comp['gates']:
  if g[0]=='h':
   bit=1<<g[1]
   for a in range(dim):
    if not a&bit:
     u=U[a].copy();v=U[a|bit].copy();U[a]=(u+v)/np.sqrt(2);U[a|bit]=(u-v)/np.sqrt(2)
  else:
   i,j=g[1:];perm=[k^(1<<j) if k>>i&1 else k for k in range(dim)];U=U[perm]
 maximum=0.
 for m in comp['measurements']:
  target=(0,0,0,0)
  for i in range(n):
   if m['original_generator_mask']>>i&1:target=multiply(target,(1<<i,adj[i],0,1<<i))
  z=pauli_matrix((0,1<<m['wire'],m['offset'],0),n)
  err=float(np.max(np.abs(U.conj().T@z@U-pauli_matrix(target,n))));maximum=max(maximum,err)
 assert maximum<1e-12;return maximum

def main():
 cert=json.loads((ROOT/'verification/fixed_block_interval_certificate.json').read_text());gcat=json.loads((ROOT/'data/graph_representatives.json').read_text())['graphs']
 outputs=[]
 for key in ['N6_m2','N6_m3','N6_m4']:
  w=cert['winners'][key];n=w['N'];dual=w['dual'];g=gcat[w['graph']];adj=[0]*n
  for i,j in g['edges']:adj[i]|=1<<j;adj[j]|=1<<i
  supports=[]
  for u in range(1<<n):
   z=0
   for i in range(n):
    if u>>i&1:z^=adj[i]
   supports.append(u|z)
  probs=[];cuts=[]
  for a in range(1<<n):
   checks=[u for u in range(1<<n) if (u&dual).bit_count()%2==0 and not(supports[u]&~a)]
   probs.append(sum((F(11,25)**supports[u].bit_count() for u in checks),F(0))/len(checks))
   cuts.append(int(rank_binary(np.array([adj[i]&((1<<n)-1-a) for i in range(n) if a>>i&1],dtype=np.int64))))
  m=int(key[-1]);best=None;bestorder=None;tested=0
  for order in itertools.permutations(range(n)):
   a=0;cost=F(0);width=0
   for v in order:cost+=probs[a];width=max(width,cuts[a]);a|=1<<v
   if width<=m-1:
    tested+=1
    if best is None or cost<best:best=cost;bestorder=order
  assert best==F(w['attempt_cost_numerator'],w['common_denominator'])
  # Entire 4**6 Pauli ensemble, in exact rational arithmetic.
  nums=[0,0,0,0];den=50**n;pivot=(dual&-dual).bit_length()-1
  for x in range(1<<n):
   ax=sum(((adj[i]&x).bit_count()%2)<<i for i in range(n))
   for z in range(1<<n):
    y=z^ax
    if y not in [0,dual]:continue
    b=(y>>pivot)&1;phase=(dual&x).bit_count()%2;index={(0,0):0,(1,0):1,(1,1):2,(0,1):3}[b,phase]
    wt=(x|z).bit_count();nums[index]+=29**(n-wt)*7**wt
  assert [F(v,sum(nums)) for v in nums]==[F(v,sum(w['Bell_counts'])) for v in w['Bell_counts']]
  assert F(sum(nums),den)==F(w['acceptance_numerator'],w['common_denominator'])
  comp=compile_order(adj,bestorder);assert comp['peak_memory']==m
  err=dense_check(adj,comp)
  comp.update({'graph':w['graph'],'edges':g['edges'],'logical_dual':dual,'key_generator':pivot,'public_check_rule':'All generator coefficient masks a with parity(a & logical_dual)=0. Reconstruct only these public combinations after each load; all other intermediate records remain private.',
      'Q':'7/25','mean_attempt_cost':str(best),'acceptance_probability':str(probs[-1]),'mean_raw_cost':str(best/probs[-1]),
      'conditioned_Bell_probabilities':[str(F(v,sum(nums))) for v in nums], 'B_rounds':w['B_rounds'], 'classical_terminal':'Watanabe',
      'dense_signed_observable_residual':err,'legal_orders_bruteforced':tested,'all_orders_examined':math.factorial(n),'all_Pauli_errors_examined':4**n})
  # Explicit inverse map from released-wire outcomes to original basis labels.
  A=[x['original_generator_mask'] | (1<<(n+j)) for j,x in enumerate(comp['measurements'])]
  for col in range(n):
   pivot_row=next(j for j in range(col,n) if A[j]>>col&1)
   A[col],A[pivot_row]=A[pivot_row],A[col]
   for j in range(n):
    if j!=col and A[j]>>col&1:A[j]^=A[col]
  inv=[a>>n for a in A];offsets=sum(x['offset']<<j for j,x in enumerate(comp['measurements']))
  comp['final_reconstruction']=[{'original_generator':i,'measurement_mask':a,'offset':(a&offsets).bit_count()%2} for i,a in enumerate(inv)]
  comp['private_key_reconstruction']=comp['final_reconstruction'][pivot]
  comp['public_check_reconstruction']=[]
  for i in range(n):
   if i==pivot:continue
   original_mask=(1<<i)^((1<<pivot) if dual>>i&1 else 0)
   mask=inv[i]^(inv[pivot] if dual>>i&1 else 0)
   comp['public_check_reconstruction'].append({'original_generator_mask':original_mask,'measurement_mask':mask,'offset':(mask&offsets).bit_count()%2})
  for x in range(1<<n):
   b=sum((((r['original_generator_mask']&x).bit_count()%2)^r['offset'])<<j for j,r in enumerate(comp['measurements']))
   reconstructed=sum((((r['measurement_mask']&b).bit_count()%2)^r['offset'])<<r['original_generator'] for r in comp['final_reconstruction'])
   assert reconstructed==x
  comp['all_affine_outcomes_checked']=1<<n
  outputs.append(comp)
 (ROOT/'data/executable_control_winners.json').write_text(json.dumps(outputs,indent=2))
 print('Exact error/cost and dense signed-circuit checks passed for',len(outputs),'winners')
 for o in outputs:print(o['peak_memory'],o['mean_raw_cost'],o['input_order'],o['dense_signed_observable_residual'])

if __name__=='__main__':main()
