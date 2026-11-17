"""Independent checks for the exact libraries, compiler, and rate kernels."""
from pathlib import Path
from fractions import Fraction
from collections import Counter
import json,sys,time
import numpy as np
from algebra import symp,quadratic,span,physical_types,code_key,spaces
from circuits import apply_gate
from rates import CodeKernel

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/'manuscript/code'))
from linear_codes import probabilities,metadata,rates,exact_rate


def verify_compiler(entry):
    m=entry['m'];checks=entry['checks'];lx=entry['logical_X'];lz=entry['logical_Z']
    mapping={}
    for p in range(1<<(2*m)):
        original_s=sum(symp(p,g,m)<<j for j,g in enumerate(checks))
        original_x=symp(p,lz,m);original_z=symp(p,lx,m)
        out=p
        for g in entry['gates']:out=apply_gate(out,g,m)
        measured_s=(out&((1<<m)-1))>>1
        frame=(original_s,original_x^(out&1),original_z^(out>>m&1))
        if measured_s in mapping:assert mapping[measured_s]==frame
        mapping[measured_s]=frame
    assert len(mapping)==1<<(m-1)
    assert len({x[0] for x in mapping.values()})==len(mapping)
    if entry['real']:
        assert all(g[0]!='s' for g in entry['gates'])
        assert all(g[0]!='s' for g in entry['terminal_gates'])
        assert all(quadratic(g,m)==0 for g in checks+[lx,lz])
    for p in range(1<<(2*m)):
        out=p
        for g in entry['terminal_gates']:out=apply_gate(out,g,m)
        measured=out&((1<<m)-1)
        recovered=[(measured&v).bit_count()%2 for v in entry['terminal_postprocessing']]
        assert recovered[0]==symp(p,lz,m)
        assert recovered[1:]==[symp(p,g,m) for g in checks]
    assert entry['M']==entry['terminal_cut']['M']
    return {str(s):dict(tensor_syndrome=f[0],bit_flip=f[1],phase_flip=f[2]) for s,f in mapping.items()}


def main():
    start=time.time();data=ROOT/'data'
    blockdata=json.loads((data/'blocks.json').read_text())
    codedata=json.loads((data/'codes.json').read_text())
    for entry in blockdata['entries']:
        entry['compiled_frame']=verify_compiler(entry)
    print('All',len(blockdata['entries']),'compiled instruments checked on every Pauli string.',flush=True)
    rng=np.random.default_rng(204619)
    states=rng.dirichlet([2.,1.,1.,1.],size=23)
    states=np.concatenate((states,np.eye(4),[[.25]*4, [.64,108/7075,108/7075,2331/7075]]))
    probs={n:probabilities(n,states) for n in range(1,7)}
    maxerror=0;shape=[]
    for entry in codedata['entries']:
        kernel=CodeKernel(entry)
        reference=rates(entry['n'],metadata(entry['n'],entry['rows']),probs[entry['n']])
        value=kernel.evaluate(states)
        error=float(np.max(np.abs(reference-value)));maxerror=max(maxerror,error)
        assert error<2e-12,(entry,error)
        ps,hmass=kernel.components(states)
        assert np.max(np.abs(ps@kernel.multiplicity-1))<1e-12
        shape.append((entry['n'],entry['k'],len(kernel.polys),len(kernel.hist)))
    print('All code kernels checked; maximum rate discrepancy',maxerror,'largest polynomial/syndrome shapes',max(x[2] for x in shape),max(x[3] for x in shape),flush=True)
    # Completeness of code orbit counts is independently checked for n<=5
    # through every labeled subspace against its stored representative.
    representatives={code_key(e['n'],e['rows']):e for e in codedata['entries']}
    orbit_checks=0
    for n in range(1,6):
        for k in range(1,n+1):
            for rows in spaces(n,k):
                e=representatives[code_key(n,rows)]
                a=rates(n,metadata(n,rows),probs[n])
                b=rates(n,metadata(n,e['rows']),probs[n])
                assert np.max(np.abs(a-b))<2e-12
                orbit_checks+=1
    result=dict(status='passed',compiled_instruments=len(blockdata['entries']),
                code_orbits=len(codedata['entries']),labeled_code_orbit_checks=orbit_checks,
                max_kernel_discrepancy=maxerror,kernel_shapes=shape,seconds=time.time()-start)
    (data/'blocks.json').write_text(json.dumps(blockdata,separators=(',',':')))
    (data/'foundation_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='kernel_shapes'},indent=2))


if __name__=='__main__':main()
