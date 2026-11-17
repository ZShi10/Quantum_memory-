"""Independent high-precision checks, exact partition coverage, and signed resources."""
from pathlib import Path
from fractions import Fraction as F
import json,mpmath as mp,hashlib
ROOT=Path(__file__).resolve().parents[1];mp.mp.dps=100

def h(q):return -sum(x*mp.log(x,2) for x in q if x>0)
def h2(x):return h([x,1-x])
def B(q):
 s=(q[0]+q[3])**2+(q[1]+q[2])**2
 return [v/s for v in [q[0]**2+q[3]**2,q[1]**2+q[2]**2,2*q[1]*q[2],2*q[0]*q[3]]],s/2

def run():
 checks=[]
 for w in json.loads((ROOT/'data/executable_control_winners.json').read_text()):
  q=[mp.mpf(str(F(s).numerator))/F(s).denominator for s in w['conditioned_Bell_probabilities']];c=F(w['mean_raw_cost']);yield_=mp.mpf(1)
  for _ in range(w['B_rounds']):q,y=B(q);yield_*=y
  a=q[0]+q[3];b=q[1]+q[2];qb,y=B(q)
  W=max(0,1-h(q)+a*b*h2((q[0]*q[1]+q[3]*q[2])/(a*b)),y*(1-h(qb)))
  rate=yield_*W*c.denominator/c.numerator
  import csv
  with (ROOT/'data/fixed_block_interval_bounds.csv').open() as f:r=next(z for z in csv.DictReader(f) if int(z['Nmax'])==6 and int(z['memory'])==w['peak_memory'])
  assert mp.mpf(r['lower'])<=rate<=mp.mpf(r['upper'])
  checks.append({'memory':w['peak_memory'],'mp_dps':100,'rate':mp.nstr(rate,90),'inside_new_interval':True})
 covers=[]
 for p in (ROOT/'verification').glob('BB*_certificate.json'):
  rec=json.loads(p.read_text());parts=[];target=F(rec['target_lower'])
  for z in rec['leaves']:
   d,i=z['depth'],z['index'];parts.append((F(i,2**d),F(i+1,2**d)));assert F(z['lower'])>=target
  parts.sort();assert parts[0][0]==0 and parts[-1][1]==1;assert all(a[1]==b[0] for a,b in zip(parts,parts[1:]))
  covers.append({'file':p.name,'leaves':len(parts),'exact_cover_verified':True,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 out={'status':'passed','independent_high_precision_winner_rates':checks,'exact_continuum_partition_checks':covers,
      'clarification':'All leaf interval bounds were computed in certificate generation; this validator independently checks exact coverage and stored targets, not a second recomputation of every interval expression. certify_selected.py --replay performs the latter.'}
 (ROOT/'verification/output_consistency.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':run()
