"""Local timing diagnostic; no published bounds are obtained from this run."""
import cProfile,pstats
import numpy as np
from full_clifford_model import HERE,bootstrap
core,engine,model,ops=bootstrap(HERE/'diagnostics/terminal_profile')
from fast_entropy import install as install_entropy
from fast_short_modes import install as install_short
from long_code_bounds import best_bounds
from bb84_fixed import physical_kernel
install_entropy();install_short()
k=core.load_codes();physical={v.entry['id']:physical_kernel(v) for v in k}
rng=np.random.default_rng(221026);p=rng.dirichlet([5.,.6,.6,.6],64)
p=np.column_stack((np.maximum(p[:,0],.55),p[:,1:]));p/=p.sum(axis=1)[:,None]
box=np.maximum(0,np.nextafter(p,-np.inf)),np.nextafter(p,np.inf)
best_bounds(tuple(v[:1] for v in box),k,physical,8)
profile=cProfile.Profile();profile.enable();best_bounds(box,k,physical,128);profile.disable()
profile.dump_stats(str(HERE/'diagnostics/terminal_profile/profile.pstats'))
pstats.Stats(profile).sort_stats('tottime').print_stats(18)
