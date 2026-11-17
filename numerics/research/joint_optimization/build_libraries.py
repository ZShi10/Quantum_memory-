"""Complete short-code orbits and full multivariate Clifford instruments."""
from pathlib import Path
from itertools import permutations
from collections import Counter
import json,time
import numpy as np
from algebra import spaces,span,symp,quadratic,compositions,physical_types,code_library,rref
from circuits import compile_instrument,measurement_cut,apply_gate

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'


def instrument_tensor(m,checks,lx,lz):
    types=compositions(m);lookup={x:i for i,x in enumerate(types)}
    physical=physical_types(m)
    counts=np.zeros((1<<(m-1),4,len(types)),dtype=np.int16)
    for p in range(1<<(2*m)):
        syndrome=sum(symp(p,g,m)<<j for j,g in enumerate(checks))
        x=symp(p,lz,m);z=symp(p,lx,m)
        logical={(0,0):0,(1,0):1,(1,1):2,(0,1):3}[x,z]
        counts[syndrome,logical,lookup[physical[p]]]+=1
    return counts


def enumerate_blocks(max_m=4):
    graph_libraries=json.loads((DATA/'graph_measurements.json').read_text())
    entries=[];counts={};realcounts={}
    for m in range(1,max_m+1):
        start=time.time();seen={};count=0;realcount=0
        for checks in spaces(2*m,m-1):
            if any(symp(a,b,m) for i,a in enumerate(checks) for b in checks[:i]):continue
            count+=1;group=set(span(checks))
            normal=[p for p in range(1<<(2*m)) if all(symp(p,g,m)==0 for g in checks)]
            rest=set(normal)-group
            lx=min(rest);rest-={lx^p for p in group};lz=min(rest)
            cosets=(lx,lx^lz,lz)
            singular=all(quadratic(g,m)==0 for g in checks)
            if singular:realcount+=1
            for logical_z in cosets:
                possible=[v for v in cosets if symp(v,logical_z,m)]
                logical_x=next((v for v in possible if singular and quadratic(v,m)==0),possible[0])
                real=singular and quadratic(logical_z,m)==0 and quadratic(logical_x,m)==0
                tensor=instrument_tensor(m,checks,logical_x,logical_z)
                # Public syndrome relabeling is free; physical types remain
                # four-dimensional. Keep zero-syndrome separate for ablations.
                nonzero=sorted([x.tobytes() for x in tensor[1:]])
                key=tensor[0].tobytes()+b''.join(nonzero)
                gates=compile_instrument(checks,logical_x,logical_z,m,real)
                terminal=measurement_cut(gates,m)
                retained=measurement_cut(gates,m,True)
                lagrangian='_'.join(map(str,rref(list(checks)+[logical_z],2*m)))
                for mode in (('real','full') if real else ('full',)):
                    graph=graph_libraries[f'{mode}_{m}'][lagrangian]
                    graph_cut=graph['cut']
                    use_graph=(graph_cut['M'],len(graph['gates']))<(terminal['M'],len(gates))
                    tg=graph['gates'] if use_graph else gates
                    tc=graph_cut if use_graph else terminal
                    post=[]
                    for observable in [logical_z]+list(checks):
                        out=observable
                        for g in tg:out=apply_gate(out,g,m)
                        assert out&((1<<m)-1)==0
                        post.append(out>>m)
                    record=dict(m=m,checks=list(checks),logical_X=logical_x,logical_Z=logical_z,
                                real=mode=='real',permission=mode,gates=gates,M=tc['M'],M_retained=retained['M'],
                                cut=terminal,retained_cut=retained,tensor=tensor.tolist(),multiplicity=1,
                                terminal_gates=tg,terminal_cut=tc,terminal_postprocessing=post,
                                terminal_implementation='graph_measurement' if use_graph else 'decoder_cut')
                    # Keep each operation permission class separately.
                    typed_key=(mode,key)
                    if typed_key in seen:
                        old=seen[typed_key];mult=old['multiplicity']+1
                        if (record['M'],len(tg))<(old['M'],len(old['terminal_gates'])):seen[typed_key]=record
                        seen[typed_key]['multiplicity']=mult
                    else:seen[typed_key]=record
        for record in seen.values():
            record['id']=f"E{len(entries):05d}";entries.append(record)
        counts[m]=count;realcounts[m]=realcount
        print('blocks',m,'isotropic',count,'real',realcount,'statistics',len(seen),'seconds',round(time.time()-start,2),flush=True)
    return entries,counts,realcounts


def main():
    DATA.mkdir(exist_ok=True)
    codes,counts,orbits=code_library(6)
    assert list(counts.values())==[1,4,15,66,373,2824]
    (DATA/'codes.json').write_text(json.dumps(dict(max_n=6,labeled_counts=counts,orbit_counts=orbits,entries=codes),indent=2))
    print('codes',len(codes),'orbits from',sum(counts.values()),'labeled subspaces',orbits,flush=True)
    blocks,counts,realcounts=enumerate_blocks(4)
    assert list(counts.values())==[1,15,315,11475]
    (DATA/'blocks.json').write_text(json.dumps(dict(max_m=4,isotropic_counts=counts,singular_counts=realcounts,entries=blocks),separators=(',',':')))
    print('Saved',len(blocks),'full multivariate instruments.',flush=True)


if __name__=='__main__':main()
