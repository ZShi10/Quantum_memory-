"""Read-only workload diagnostic; no floating value is used as a certificate."""
from fractions import Fraction as F
import json, time
import numpy as np
from full_clifford_model import HERE,bootstrap
from binary_evaluate import outward

q=F(23,100);job=HERE/'diagnostics/upper_groups';core,engine,model,ops=bootstrap(job)
import stable_intervals as iv
from long_code_bounds import classical_frame_boxes,h
nodes=np.load(HERE/'data/full_binary_catalog.npz')['nodes']
events=np.array([[(i//4,i%4,k) for i,k in enumerate(r['mapping']) if k>=0] for r in ops],np.int32)
feasible=json.loads((HERE/'bb84/0.23/feasible.json').read_bytes())
lower=np.array([0.]+[feasible['results'][f'M{m}']['lower'] for m in range(1,9)])
tests=[F(0),F(1,32),F(1,8),F(1)]
source=[tuple(iv.rational(v) for v in (1-2*q+q*t,q-q*t,q*t,q-q*t)) for t in tests]
sl=np.array([[v[0] for v in row] for row in source]);sh=np.array([[v[1] for v in row] for row in source])
p0,p1,y0,y1=outward(nodes,events,sl,sh)
mx=np.minimum(1,p1.max(-1));entropy=h((mx,mx))[0]+h((np.maximum(0,iv.down(1-mx)),np.minimum(1,iv.up(1-mx))))[0]
er=np.where(mx>.5,iv.up(1-entropy),0.)
rates=iv.up(y1*np.maximum(0,er));cap=rates.min(-1)
report=[]
for ai,axis in enumerate('XYZ'):
    mm=nodes[:,5+ai];active=np.flatnonzero(cap>lower[mm]+np.maximum(2e-9,lower[mm]*1e-4))
    anchor=rates[active].argmin(-1);perm=list(model.AXIS_PERMUTATIONS[axis])
    p=(p0[active,anchor][:,perm],p1[active,anchor][:,perm])
    boxes,inv=classical_frame_boxes(p)
    row=dict(axis=axis,active=len(active),groups=len(boxes[0]));report.append(row);print(row,flush=True)
path=job/'result.json';path.write_text(json.dumps(report,indent=2))
