from common import *
import json,time
import numpy as np
from complete_werner_long import spc_upper_array
from spc_long import spc_rate

start=time.time();rng=np.random.default_rng(418039)
states=np.r_[rng.dirichlet(np.ones(4),size=80),[[1,0,0,0],[.5,0,0,.5],[.5,.5,0,0],[.25]*4]]
lengths=np.array(list(range(2,13))+[16,32,64])
upper=spc_upper_array(states,lengths);checks=0;minimum=1
for j,p in enumerate(states):
    for i,n in enumerate(lengths):
        r=spc_rate(p,int(n));gap=upper[j,i]-r
        assert gap>=-2e-12,(p,n,r,upper[j,i]);minimum=min(minimum,gap);checks+=1
report=dict(status='passed',comparisons=checks,minimum_upper_minus_rate=minimum,seconds=time.time()-start)
(DATA/'spc_upper_validation.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
