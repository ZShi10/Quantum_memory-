from pathlib import Path
from itertools import permutations
import json
import numpy as np
from rates import CodeKernel
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data'
partition=({0,3},{1,2})
frames=[p for p in permutations(range(4)) if set(p[i] for i in partition[0]) in partition]
codes=json.loads((DATA/'codes.json').read_text())['entries']
rng=np.random.default_rng(260919);states=rng.dirichlet(np.ones(4),size=12)
error=0.
for code in codes:
    kernel=CodeKernel(code);base=kernel.evaluate(states)
    for p in frames:
        values=kernel.evaluate(states[:,p]);error=max(error,float(abs(values-base).max()))
        assert np.max(abs(values-base))<3e-14
result=dict(status='passed',codes=len(codes),frames=len(frames),states=len(states),max_rate_discrepancy=error)
(DATA/'frame_verification.json').write_text(json.dumps(result,indent=2));print(result)
