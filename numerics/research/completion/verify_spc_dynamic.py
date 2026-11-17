from common import *
import time,json
import numpy as np
from spc_dynamic import all_rates
from spc_long import spc_rate

start=time.time();rng=np.random.default_rng(726919)
p=np.r_[rng.dirichlet(np.ones(4),size=70),[[1,0,0,0],[.5,0,0,.5],[.51,.49,0,0],[.501,.002,.003,.494]]]
ns=np.array([2,3,7,8,12,25,64,128]);a=all_rates(p,ns)
b=np.array([[spc_rate(s,int(n)) for n in ns] for s in p]);error=float(np.max(np.abs(a-b)))
assert error<2e-12,error
report=dict(status='passed',comparisons=len(p)*len(ns),max_discrepancy=error,seconds=time.time()-start)
(DATA/'spc_dynamic_validation.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
