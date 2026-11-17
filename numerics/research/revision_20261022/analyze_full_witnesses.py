"""Exact cost/state decomposition of two BB84 witnesses at a common source."""
from fractions import Fraction as F
import json
import numpy as np
from full_clifford_model import HERE,bootstrap
core,engine,model,ops=bootstrap(HERE/'diagnostics/full_witness_cost')
from fast_entropy import install
install()
import stable_intervals as iv
from certify_bb84_recursive import watanabe_bound
source=HERE/'export/bb84_policies.json'
if not source.exists():source=HERE.parents[1]/'evidence/revision_20261022/export/bb84_policies.json'
policies=json.loads(source.read_bytes())
q=F(1,5);p=(1-2*q,q,F(0),q)
def evaluate(tree):
    if tree['kind']=='raw':return p,F(1),F(1),1
    left,cl,pl,nl=evaluate(tree['left']);right,cr,pr,nr=evaluate(tree['right'])
    raw=[F(0)]*4
    for i,j,k in model.EVENTS[tree['decoder']]:raw[k]+=left[i]*right[j]
    s=sum(raw);return tuple(v/s for v in raw),(cl+cr)/s,pl*pr*s,nl+nr
rows=[]
for m in (2,3):
    policy=policies[f'full_Q0.2_M{m}'];ed=policy['ED'];law,c,joint,n=evaluate(ed['tree'])
    existing=model.evaluate(model.tree_from_dict(ed['tree']),np.array(list(map(float,p))),ed['final_axis'])
    assert abs(float(c)-existing['expected_raw_cost'])<1e-12
    state=tuple(law[i] for i in model.AXIS_PERMUTATIONS[ed['final_axis']]);factor=F(1)
    for _ in range(policy['B']):
        I,X,Y,Z=state;s=(I+Z)**2+(X+Y)**2
        state=((I*I+Z*Z)/s,(X*X+Y*Y)/s,2*X*Y/s,2*I*Z/s);factor*=s/2
    assert len(policy['strategy'])==1 and policy['strategy'][0]['AD']['kind']=='Watanabe_2007'
    box=[iv.rational(v) for v in state];bounds=tuple(np.array([[v[k] for v in box]]) for k in (0,1))
    rate=watanabe_bound(bounds);per=iv.mul(rate,iv.rational(factor));net=iv.div(per,iv.rational(c))
    restart=F(n)/joint
    rows.append(dict(memory=m,Q=str(q),z='0',N=n,tree=ed['canonical_tree'],axis=ed['final_axis'],B=policy['B'],
        fidelity=str(law[0]),ED_law=list(map(str,law)),cost_retained=str(c),cost_full_restart=str(restart),
        per_ED_output_lower=float(per[0][0]),per_ED_output_upper=float(per[1][0]),
        rate_lower=float(net[0][0]),rate_upper=float(net[1][0]),cost_ratio=float(restart/c)))
result=dict(status='passed',source='Exact symmetric BB84 Q=1/5,z=0',rows=rows,
    scope='A common-source diagnostic. The restart comparison explains cost, not an exclusion of all lower-memory alternatives. Whole hidden-interval conclusions use separate certificates.')
(HERE/'diagnostics/full_witness_cost/result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
