"""Export a valid saved whole-domain cover without claiming gap closure."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json
from full_clifford_model import HERE
from full_binary_search import read,save

job=HERE/'bb84/0.23/symmetric_lower/9955f5c97cdd564f4993'
files=list((job/'data/bb84_interval_work').glob('*.json'));assert len(files)==1
path=files[0];data=read(path);cells=sorted((F(v[2]),F(v[3]),float(v[0])) for v in data['heap'])
assert cells[0][0]==0 and cells[-1][1]==1
assert all(a<b for a,b,_ in cells) and all(x[1]==y[0] for x,y in zip(cells,cells[1:]))
lower=min(c[2] for c in cells);upper=data['incumbent'];assert lower<=upper
cert=dict(lower=lower,upper=upper,gap=upper-lower,argmin=data['argmin'],cells=data['counter'],
    converged=False,atol=5e-10,whole_domain_cover=True,cover_cells=len(cells),
    cover_source=str(path),cover_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    arithmetic='Valid outward boxes with exact dyadic endpoints tile [0,1]; first-derivative mean-value bounds. The remaining gap is reported, not declared closed.')
row=dict(complete=True,Q=.23,lower=max(0.,lower),ED=data['ED'],B=data['B'],strategy=data['strategy'],certificate=cert,
    execution='Export of a completed valid domain-cover audit; fixed-policy lower only.',label='real_xyz')
save(job/'cover_lower.json',row)
save(job.parent/'cover_lower_latest.json',dict(path=str(job/'cover_lower.json'),sha256=hashlib.sha256((job/'cover_lower.json').read_bytes()).hexdigest()))
print(json.dumps(dict(lower=lower,upper=upper,gap=upper-lower,cells=len(cells))),flush=True)
