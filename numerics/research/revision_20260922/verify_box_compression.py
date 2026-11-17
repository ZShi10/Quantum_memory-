"""Check containing-box reuse against uncompressed interval evaluations."""
from core import *
from long_code_bounds import best_bounds,classical_frame_boxes
from bb84_fixed import physical_kernel
from known_search import canonical_classical_states
from ad_modes import best_short
from compound_terminals import actions,values

rng=np.random.default_rng(9149);base=rng.dirichlet(np.ones(4),8)
pp=np.concatenate([base,base[:,[3,1,2,0]],base[:,[1,0,3,2]],base+np.array([1,-1,1,-1])*1e-16])
ll=np.maximum(0,np.nextafter(pp,-np.inf));hh=np.minimum(1,np.nextafter(pp,np.inf))
boxes,inverse=classical_frame_boxes((ll,hh));canonical=canonical_classical_states(pp)
assert np.all(canonical>=boxes[0][inverse]) and np.all(canonical<=boxes[1][inverse])
k=load_codes();physical={v.entry['id']:physical_kernel(v) for v in k}
lo,hi=best_bounds((ll,hh),k,physical,16);actual=values(pp,actions(16)).max(-1)
assert np.all(actual>=lo-1e-13) and np.all(actual<=hi+1e-13)
out=dict(status='passed',source_boxes=len(pp),merged_boxes=len(boxes[0]),
         max_width=float(np.max(hi-lo)),scope='Containing hull after a fixed classical frame; no near-equality assumption. Floating terminal comparison is a diagnostic only.')
atomic_json(DATA/'box_compression_checks.json',out);print(json.dumps(out),flush=True)
