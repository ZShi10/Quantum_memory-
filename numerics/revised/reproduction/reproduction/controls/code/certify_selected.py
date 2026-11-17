"""Independent uniform-z certificates for the principal large-memory protocols."""
from interval_core import *
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[1]
PROTOCOLS={
 'BB23_m5':{'Q':F(23,100),'word':'RRZRX','axis':'Z','target':'0.00054334','probe_t':'0.000065148'},
 'BB23_m6':{'Q':F(23,100),'word':'RRZRR','axis':'X','target':'0.00095290','probe_t':'0.02332407'},
 'BB24_m6':{'Q':F(24,100),'word':None,'axis':'Z','target':'0.000063','probe_t':None},
}

def load():
 d=json.loads((ROOT/'data/selected_protocols.json').read_text())
 for key,v in PROTOCOLS.items():
  mm=int(key[-1]);sel=next(s for s in d['selections'] if s['model']=='BB84_real' and F(str(s['Q']))==v['Q'] and s['budget']==f'M{mm}')
  de=d['definitions'][sel['protocol_id']];ed=de['ED']
  assert ed['family']=='real_decoder_recurrence' and de['classical_B_rounds']==0
  assert de['AD_strategy'][0]['AD']['kind']=='Watanabe_2007'
  v['word']=ed['path'];v['axis']=ed['final_axis'];v['probe_t']=str(de['saved_rate_certificate']['argmin'])
 return PROTOCOLS
load()

def rate(name,z):
 v=PROTOCOLS[name];Q=D(iv.mpf(v['Q'].numerator)/v['Q'].denominator)
 p=[1-2*Q+z,Q-z,z,Q-z];C=D(1)
 for d in v['word']:p,s=dec(d,p,p);C=2*C/s
 p=[p[i] for i in {'Z':[0,1,2,3],'X':[0,3,2,1]}[v['axis']]]
 return W(p)/C

def bound(name,l,r):
 out=rate(name,D(interval(l,r),1));lower=lo(out.v)
 if mp.isfinite(lo(out.d)) and mp.isfinite(hi(out.d)):
  mid=(l+r)/2;center=rate(name,D(iv.mpf(mid),1));lower=max(lower,lo(center.v+out.d*interval(l-mid,r-mid)))
 return lower

def certify(name):
 v=PROTOCOLS[name];q=v['Q'];target=mp.mpf(v['target']);todo=[(0,0)];leaves=[];start=time.time();processed=0
 while todo:
  dep,i=todo.pop();processed+=1;l=lo(iv.mpf(q.numerator)*i/(q.denominator*2**dep));r=hi(iv.mpf(q.numerator)*(i+1)/(q.denominator*2**dep))
  try:lower=bound(name,l,r)
  except (ArithmeticError,ValueError,ZeroDivisionError):lower=-mp.inf
  if lower>=target:leaves.append({'depth':dep,'index':i,'lower':mp.nstr(lower,40)})
  else:
   if dep>70 or processed>20000:raise RuntimeError('Incomplete interval cover')
   todo.extend([(dep+1,2*i+1),(dep+1,2*i)])
 pz=iv.mpf(v['probe_t'])*q.numerator/q.denominator
 upper=hi(rate(name,D(pz,1)).v)
 rec={'status':'complete_uniform_z_lower_certificate','name':name,'Q':str(q),'word':v['word'],'axis':v['axis'],'classical':'Watanabe; no B prefix',
      'target_lower':v['target'],'proved_lower':mp.nstr(min(mp.mpf(a['lower']) for a in leaves),40),'infimum_upper_at_test_point':mp.nstr(upper,40),
      'probe_t':v['probe_t'],'interval_dps':iv.dps,'leaves':leaves,'seconds':time.time()-start,
      'scope':'One fixed protocol over the complete BB84-compatible interval; not a class-wide upper certificate.'}
 (ROOT/'verification'/f'{name}_certificate.json').write_text(json.dumps(rec,indent=2));print(name,len(leaves),'leaves','seconds',round(time.time()-start,2),'infimum bracket',rec['proved_lower'],rec['infimum_upper_at_test_point'],flush=True)
 return rec

def replay(path):
 rec=json.loads(Path(path).read_text());q=F(rec['Q']);target=mp.mpf(rec['target_lower']);parts=[]
 for leaf in rec['leaves']:
  d,i=leaf['depth'],leaf['index'];l=lo(iv.mpf(q.numerator)*i/(q.denominator*2**d));r=hi(iv.mpf(q.numerator)*(i+1)/(q.denominator*2**d))
  assert bound(rec['name'],l,r)>=target
  parts.append((F(i,2**d),F(i+1,2**d)))
 parts.sort();assert parts[0][0]==0 and parts[-1][1]==1
 assert all(a[1]==b[0] for a,b in zip(parts,parts[1:]));print('REPLAY PASSED',rec['name'],len(parts),flush=True)

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--replay');a=ap.parse_args()
 if a.replay:replay(a.replay)
 else:
  for name in PROTOCOLS:certify(name)
