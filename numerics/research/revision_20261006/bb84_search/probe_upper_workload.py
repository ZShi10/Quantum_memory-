"""Read-only count of candidates entering costly B=0 upper screening."""
from pathlib import Path
import sys,json
import run_point
job=Path(sys.argv[1])
core,_,upper,_=run_point.bootstrap(job)
from fractions import Fraction as F
import numpy as np
from recurrence_interval_library import generate_bounds,records
from recurrence_equivalence import representatives,key
from certify_bb84_recursive import source
import stable_intervals as iv
src=json.loads(next((job/'data').glob('bb84_joint_extension_*_n128.json')).read_text())
b,p=src['b'],src['p']
anchors=[F(0),F(1,64),F(1,8),F(1,2),F(1)]
pp,yy=generate_bounds(source(b,p,[(t,t) for t in anchors]),src['permissions'])
M,rr=records(src['permissions']);rep=representatives(rr,b==p);ids=np.flatnonzero(rep==np.arange(len(rep)))
pp=tuple(v[ids] for v in pp); yy=tuple(v[ids] for v in yy); M=M[ids]
lower=np.array([0.]+[src['results'][f'M{m}']['lower'] for m in range(1,9)])
concurrence=np.maximum(0,iv.up(2*pp[1].max(-1)-1))
bound=np.minimum(yy[1],iv.up(yy[1]*iv.up(concurrence*concurrence))).min(-1)
evidence=json.loads((job/'data/imported_interval_dual_evidence.json').read_text())['duals']
tags=[json.dumps([key(rr[i]),0],sort_keys=True) for i in ids]
for j,tag in enumerate(tags):
    if tag in evidence:bound[j]=min(bound[j],evidence[tag]['upper'])
active=np.flatnonzero(bound>lower[M]+5e-12)
out=dict(total_records=len(rr),representatives=len(ids),active=len(active),
         active_per_memory={str(m):int(np.sum(M[active]==m)) for m in range(1,9)})
print(json.dumps(out))
