"""Exact state/cost and outward rate profiles of fixed full-Clifford witnesses."""
from fractions import Fraction as F
import csv,hashlib,json
import numpy as np
from full_clifford_model import HERE,PROJECT,bootstrap
core,engine,model,ops=bootstrap(HERE/'diagnostics/full_witness_profiles')
from fast_entropy import install
install()
import stable_intervals as iv
from certify_bb84_recursive import watanabe_bound
source=HERE/'export/bb84_policies.json'
if not source.exists():source=PROJECT/'evidence/revision_20261022/export/bb84_policies.json'
policies=json.loads(source.read_bytes());q=F(1,5)

def evaluate(tree,p):
    if tree['kind']=='raw':return p,F(1),F(1),1
    left,cl,pl,nl=evaluate(tree['left'],p);right,cr,pr,nr=evaluate(tree['right'],p)
    raw=[F(0)]*4
    for i,j,k in model.EVENTS[tree['decoder']]:raw[k]+=left[i]*right[j]
    s=sum(raw)
    return tuple(v/s for v in raw),(cl+cr)/s,pl*pr*s,nl+nr

rows=[];summaries=[]
for m in (2,3):
    policy=policies[f'full_Q0.2_M{m}'];ed=policy['ED'];selected=[]
    assert len(policy['strategy'])==1 and policy['strategy'][0]['AD']['kind']=='Watanabe_2007'
    for i in range(201):
        z=F(i,1000);p=(1-2*q+z,q-z,z,q-z)
        law,c,joint,n=evaluate(ed['tree'],p)
        numeric=model.evaluate(model.tree_from_dict(ed['tree']),np.array(list(map(float,p))),ed['final_axis'])
        assert abs(float(c)-numeric['expected_raw_cost'])<1e-10
        state=tuple(law[j] for j in model.AXIS_PERMUTATIONS[ed['final_axis']]);factor=F(1)
        for _ in range(policy['B']):
            I,X,Y,Z=state;s=(I+Z)**2+(X+Y)**2
            state=((I*I+Z*Z)/s,(X*X+Y*Y)/s,2*X*Y/s,2*I*Z/s);factor*=s/2
        box=[iv.rational(v) for v in state];bounds=tuple(np.array([[v[k] for v in box]]) for k in (0,1))
        terminal=watanabe_bound(bounds);per=iv.mul(terminal,iv.rational(factor));net=iv.div(per,iv.rational(c))
        assert float(net[1][0])>=policy['lower']
        row=dict(policy=f'full_Q0.2_M{m}',memory=m,Q='1/5',z=float(z),z_exact=str(z),N=n,
            axis=ed['final_axis'],B=policy['B'],C=float(c),C_exact=str(c),C_restart_exact=str(F(n)/joint),
            r_cl_lower=float(per[0][0]),r_cl_upper=float(per[1][0]),rate_lower=float(net[0][0]),rate_upper=float(net[1][0]))
        rows.append(row);selected.append(row)
    minimum=min(selected,key=lambda r:r['rate_lower'])
    summaries.append(dict(memory=m,sampled_minimum=minimum,uniform_lower=policy['lower']))
out=HERE/'export';out.mkdir(exist_ok=True)
with (out/'full_witness_profiles.csv').open('w',encoding='utf-8',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
result=dict(status='passed',points_per_policy=201,policies=2,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),summaries=summaries,
    scope='Fixed-policy source profiles only. Every state and cost uses exact rational arithmetic; each entropy rate is outward enclosed. The independently certified whole-interval lower, not the sampled minimum, guarantees the worst case.')
(HERE/'diagnostics/full_witness_profiles/result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
