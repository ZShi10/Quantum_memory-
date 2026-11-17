"""Time the already validated CSR bound on a realistic full-menu batch."""
import json,time
import numpy as np
from full_clifford_model import HERE,bootstrap
core,engine,model,ops=bootstrap(HERE/'diagnostics/sparse_benchmark')
import stable_intervals as iv
from fast_entropy import install as entropy_install
from fast_short_modes import install as short_install
from fast_positive_dot import install as dot_install
from long_code_bounds import best_bounds
from bb84_fixed import physical_kernel
entropy_install();short_install()
kernels=core.load_codes();physical={k.entry['id']:physical_kernel(k) for k in kernels}
rng=np.random.default_rng(221026)
p=rng.dirichlet([5.,.6,.6,.6],256)
p=np.column_stack((np.maximum(p[:,0],.55),p[:,1:]))
p=p/p.sum(axis=1)[:,None]
box=np.maximum(0,np.nextafter(p,-np.inf)),np.nextafter(p,np.inf)
original=iv.positive_dot
# Warm compiled entropy and CSR compilation, then use identical inputs.
best_bounds(tuple(v[:2] for v in box),kernels,physical,16)
dot_install();sparse=iv.positive_dot
best_bounds(tuple(v[:2] for v in box),kernels,physical,16)
times={};results={}
for name,fn in [('dense',original),('sparse',sparse),('dense_repeat',original),('sparse_repeat',sparse)]:
    iv.positive_dot=fn;start=time.perf_counter()
    results[name]=best_bounds(box,kernels,physical,128)
    times[name]=time.perf_counter()-start
    print(name,times[name],flush=True)
for key,value in results.items():
    assert np.all(value[0]<=results['dense'][1]) and np.all(results['dense'][0]<=value[1]),key
report=dict(states=len(p),codes=len(kernels),seconds=times,scope='Performance diagnostic; interval overlap plus prior independent CSR decimal validation.')
(HERE/'diagnostics/sparse_benchmark/result.json').write_text(json.dumps(report,indent=2)+'\n')

