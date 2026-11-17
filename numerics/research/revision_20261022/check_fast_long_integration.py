"""Independent complete-menu integration check for compiled long formulas."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from full_clifford_model import HERE,bootstrap
core,engine,model,ops=bootstrap(HERE/'diagnostics/fast_long_integration')
from fast_entropy import install as install_entropy
from fast_short_modes import install as install_short
from fast_long_menu import install as install_long
from long_code_bounds import best_bounds
from bb84_fixed import physical_kernel
install_entropy();install_short()
kernels=core.load_codes();physical={k.entry['id']:physical_kernel(k) for k in kernels}
rng=np.random.default_rng(231026);p=rng.dirichlet([3.,.4,.4,.4],32)
p=np.vstack((p,[.7,.1,.1,.1],[.52,.16,.16,.16],[.9,.02,.03,.05],[.500001,.166666,.166666,.166667]))
box=np.maximum(0,np.nextafter(p,-np.inf)),np.nextafter(p,np.inf)
start=time.monotonic();prior=best_bounds(box,kernels,physical,128);old=time.monotonic()-start
install_long();start=time.monotonic();new=best_bounds(box,kernels,physical,128);new_time=time.monotonic()-start
assert np.isfinite(new).all()
assert np.all(prior[0]<=new[1]) and np.all(new[0]<=prior[1])
result=dict(status='passed',sources=len(p),short_codes=len(kernels),endpoint_n=128,
    old_seconds=old,new_seconds=new_time,source_sha256=hashlib.sha256((HERE/'fast_long_menu.py').read_bytes()).hexdigest())
(HERE/'diagnostics/fast_long_integration/result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
