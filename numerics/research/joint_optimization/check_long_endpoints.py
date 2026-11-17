from pathlib import Path
import json,sys,time
import numpy as np
from spc_long import spc_rate
from rates import CodeKernel
from algebra import compositions
ROOT=Path(__file__).resolve().parent;D=ROOT/'data'
sys.path.insert(0,str(ROOT.parents[1]/'manuscript'/'code'))
from protocols import bstep

codes=json.loads((D/'codes.json').read_text())['entries']
rng=np.random.default_rng(1063);error=0.
for c in codes:
    if c['k']!=c['n']-1 or c['d']!=2:continue
    kernel=CodeKernel(c)
    for _ in range(20):
        p=rng.dirichlet(np.ones(4));error=max(error,abs(spc_rate(p,c['n'])-float(kernel.evaluate(p))))
assert error<2e-13
witnesses=json.loads((D/'code_witnesses_130digit.json').read_text());output=[]
start=time.time()
for w in witnesses:
    p=np.array([float(v) for v in w['AD_input']]);best=(0.,None)
    for n in range(2,65):
        r=spc_rate(p,n)
        if r>best[0]:best=(r,n)
    record=dict(code=w['code']['id'],Q=w['Q'],chosen_rate=float(w['chosen_conditional_rate']),
                SPC_n2_to_64_best_rate=best[0],SPC_best_n=best[1])
    blocks={e['id']:e for e in json.loads((D/'blocks.json').read_text())['entries']}
    ed=blocks[w['ED']];q=float(w['Q']);praw=np.array([1-1.5*q,q/2,q/2,q/2])
    powers=np.prod(praw[None,:]**np.array(compositions(ed['m'])),axis=1)
    raw=np.asarray(ed['tensor'])[0]@powers;ps=raw.sum();state=raw/ps;factor=ps/ed['m'];spc_best=(0.,None)
    for k in range(25):
        for n in range(2,65):
            value=factor*spc_rate(state,n)
            if value>spc_best[0]:spc_best=(value,(k,n))
        state,s=bstep(state);factor*=s/2
    record['SPC_optimized_prefix_and_n64_raw_rate']=spc_best[0]
    record['SPC_optimized_prefix_and_n64_choice']=spc_best[1]
    output.append(record);print(record,'seconds',time.time()-start,flush=True)
(D/'long_endpoint_check.json').write_text(json.dumps(dict(short_kernel_verification_error=error,results=output),indent=2))
