"""Independent exact rational checks of the new compiled DAG evaluator."""
from pathlib import Path
from fractions import Fraction as F
from functools import lru_cache
import hashlib,json
import numpy as np
from full_clifford_model import HERE,AXES,install_nodes
from binary_evaluate import floating,outward

def main():
    model,ops=install_nodes();allnodes=np.load(HERE/'data/full_binary_catalog.npz')['nodes']
    rng=np.random.default_rng(221026)
    targets=sorted(set([0,1,5,6,8,10,15,80,1296,len(allnodes)-1]+rng.integers(1,len(allnodes),50).tolist()))
    required={0}
    def add(i):
        if i in required:return
        required.add(i);add(int(allnodes[i,1]));add(int(allnodes[i,2]))
    for i in targets:add(i)
    ordered=sorted(required);translation={v:i for i,v in enumerate(ordered)}
    nodes=allnodes[ordered].copy()
    for row in nodes[1:]:row[1]=translation[int(row[1])];row[2]=translation[int(row[2])]
    events=np.array([[(i//4,i%4,k) for i,k in enumerate(r['mapping']) if k>=0] for r in ops],np.int32)
    sources=[]
    for q,t in [(F(23,100),F(0)),(F(23,100),F(1,4)),(F(23,100),F(1)),(F(1,6),F(7,43))]:
        z=q*t;sources.append((1-2*q+z,q-z,z,q-z))
    fs=np.array(sources,float)
    slo=np.nextafter(fs,-np.inf).clip(0,1);shi=np.nextafter(fs,np.inf).clip(0,1)
    p,y=floating(nodes,events,fs);pl,ph,yl,yh=outward(nodes,events,slo,shi)
    tested=0;max_float_error=0.
    for t,source in enumerate(sources):
        @lru_cache(None)
        def exact(i):
            if i==0:return source,F(1)
            op,l,r=map(int,nodes[i,:3]);a,ca=exact(l);b,cb=exact(r)
            weights=[F(0)]*4
            for j,k,z in events[op]:weights[int(z)]+=a[int(j)]*b[int(k)]
            s=sum(weights);return tuple(w/s for w in weights),(ca+cb)/s
        for target in targets:
            i=translation[target];law,cost=exact(i);values=(*law,1/cost)
            for k,value in enumerate(values):
                lo,hi=(pl[i,t,k],ph[i,t,k]) if k<4 else (yl[i,t],yh[i,t])
                assert F(float(lo))<=value<=F(float(hi)),(target,t,k,lo,float(value),hi)
                floating_value=p[i,t,k] if k<4 else y[i,t]
                max_float_error=max(max_float_error,abs(float(value)-floating_value));tested+=1
    report=dict(status='passed',exact_rational_value_checks=tested,targets=len(targets),
        dependency_closed_nodes=len(nodes),source_cases=len(sources),
        maximum_floating_absolute_error=max_float_error,
        coverage='Boundary and interior hidden sources, homogeneous depth seven, native constant pump through 160 inputs, and tree/variable-pump nodes.',
        evaluator_sha256=hashlib.sha256((HERE/'binary_evaluate.py').read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/'data/binary_evaluator_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
