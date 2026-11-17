"""Compare fidelity-first and final-rate-first ED at the same block size."""
from pathlib import Path
import json
import numpy as np
from search_werner import werner_blocks
from rates import CodeKernel,best_codes
from algebra import code_components
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data'
entries=werner_blocks(json.loads((DATA/'blocks.json').read_text())['entries'])
codes=[e for e in json.loads((DATA/'codes.json').read_text())['entries'] if len(code_components(e['n'],e['rows']))==1]
kernels=[CodeKernel(c) for c in codes];output=[]
for m in (3,4):
    es=[e for e in entries if e['m']==m and e['M']<=2]
    for Q in (.15,.2,.25,.28,.30):
        f=1-1.5*Q;t=Q/2;powers=np.array([f**(m-w)*t**w for w in range(m+1)])
        raw=np.array([np.asarray(e['werner_weights'])[0]@powers for e in es]);ps=raw.sum(1)
        p=raw/ps[:,None];fidelity=p.max(1);state=p.copy();factor=ps/m
        best=np.zeros(len(es));choice=[None]*len(es)
        for k in range(13):
            values=best_codes(state,kernels)
            r=factor*values['rate']
            for i in np.flatnonzero(r>best+1e-16):
                best[i]=r[i];choice[i]=(k,codes[int(values['code'][i])]['id'])
            a,x,y,z=state.T
            out=np.stack((a*a+z*z,x*x+y*y,2*x*y,2*a*z),1);s=out.sum(1)
            state=out/s[:,None];factor*=s/2
        mostF=np.flatnonzero(fidelity>=fidelity.max()-1e-13)
        fi=int(mostF[np.argmax(best[mostF])]);ri=int(np.argmax(best))
        if best[ri]>best[fi]+1e-12 and fidelity[ri]<fidelity[fi]-1e-10:
            record=dict(Q=Q,m=m,M_budget=2,
                        fidelity_first=dict(ED=es[fi]['id'],F=float(fidelity[fi]),success=float(ps[fi]),rate=float(best[fi]),AD=choice[fi]),
                        rate_first=dict(ED=es[ri]['id'],F=float(fidelity[ri]),success=float(ps[ri]),rate=float(best[ri]),AD=choice[ri]))
            output.append(record);print(record,flush=True)
(DATA/'fidelity_ablation.json').write_text(json.dumps(output,indent=2))
