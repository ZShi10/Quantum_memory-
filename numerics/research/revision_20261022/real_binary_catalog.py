"""Matched real-Clifford intermediate controls, with the same structural caps."""
from itertools import product
import hashlib,json
import numpy as np
from clifford_normal_forms import bfs,error_map,outperm,AXES
from full_clifford_model import HERE,data,install_nodes,compose_input,axis_before

def main():
    model,ops=install_nodes();opid={r['name']:i for i,r in enumerate(ops)}
    bymap={tuple(r['mapping']):r for r in ops[9:]}
    H=(0,3,2,1);frames=[(0,1,2,3),H]
    normal={}
    for name in ('X','R','Z','C'):
        mapping=[-1]*16
        for i,j,k in model.EVENTS[name]:mapping[4*i+j]=k
        row=bymap[tuple(mapping)];normal[row['check']]=row
    c=normal['XZ']['mapping'];normal['ZX']=bymap[tuple(c[4*j+i] for i,j in product(range(4),repeat=2))]
    lookup={outperm(row['mapping'],frame):(name,frame) for name,row in normal.items() for frame in frames}
    real=[r for r in ops[9:] if r['real_implementable']]
    assert len(real)==10 and set(lookup)=={tuple(r['mapping']) for r in real}
    count=0
    for row in real:
        for a,b in product(frames,repeat=2):
            mapped=compose_input(row['mapping'],a,b)
            assert mapped in lookup;count+=1
    for check,row in normal.items():
        assert tuple(row['mapping'])==tuple(normal[check[::-1]]['mapping'][4*j+i] for i,j in product(range(4),repeat=2))
    def raw_tensor(mapping):
        out=[0]*48
        for i,j in product(range(4),repeat=2):
            k=mapping[4*i+j]
            if k>=0:out[12*k+3*i+(0 if j==0 else 2 if j==2 else 1)]+=1
        return tuple(out)
    raw_normal=['XZ','YY','ZZ'];raw_proof={}
    for check,row in normal.items():
        found=[(c,p) for c in raw_normal for p in frames if raw_tensor(row['mapping'])==raw_tensor(outperm(normal[c]['mapping'],p))]
        assert found;raw_proof[check]=found[0]
    # Recover actual H/CNOT-only implementations, not merely a real-map label.
    realwords={}
    for cols,word in bfs(real=True).items():realwords.setdefault(error_map(cols),(cols,word))
    executable=[]
    for row in real:
        cols,word=realwords[tuple(row['mapping'])]
        assert all(g[0]!='s' for g in word)
        executable.append(dict(row,real_columns=cols,real_gates=word))
    nodes=[[-1,-1,-1,1,1,1,1,1,15]];index={};levels={1:[0]}
    normal_by_name={r['name']:c for c,r in normal.items()}
    def add(name,left,right,mask):
        if name in normal_by_name and left>right:
            check=normal_by_name[name];left,right=right,left;name=normal[check[::-1]]['name']
        oid=opid[name];key=(oid,left,right)
        if key in index:
            idx=index[key];nodes[idx][8]|=mask;return idx
        a,b=nodes[left],nodes[right];w=min(max(a[4],b[4]+1),max(b[4],a[4]+1));terminal=[]
        for axis in AXES:
            targets=model.TERMINAL_REDUCTIONS.get((name,axis))
            terminal.append(min(w,max(a[5+AXES.index(targets[0])],b[5+AXES.index(targets[1])])) if targets else w)
        idx=len(nodes);index[key]=idx;nodes.append([oid,left,right,a[3]+b[3],w,*terminal,mask]);return idx
    front=[0];hc=[1]
    for depth in range(1,8):
        front=sorted({add(normal[c]['name'],i,i,1) for i in front for c in ('XX','YY','ZZ','XZ')});hc.append(len(front))
    front=[0];pc=[1]
    for depth in range(1,8):
        front=sorted({add(normal[c]['name'],i,0,2) for i in front for c in raw_normal});pc.append(len(front))
    for n in range(2,6):
        ids=set()
        for k in range(1,n):
            for a in levels[k]:
                for b in levels[n-k]:
                    for row in normal.values():ids.add(add(row['name'],a,b,4))
        levels[n]=sorted(ids)
    for row in real:
        last=0
        for n in range(2,161):last=add(row['name'],last,0,8)
    path=HERE/'data/real_binary_catalog.npz';np.savez_compressed(path,nodes=np.asarray(nodes,np.int32))
    report=dict(status='passed',nodes=len(nodes),quantum_operation_count=10,canonical_real_checks=list(normal),
        homogeneous_depth=7,homogeneous_level_nodes=hc,variable_pump_checks=7,variable_pump_level_nodes=pc,
        ordered_tree_leaves=5,tree_layer_nodes={k:len(v) for k,v in levels.items()},constant_native_pump_inputs=160,
        constant_native_maps=10,real_native_executable=executable,frame_transport_checks=count,
        raw_H_reduction=raw_proof,final_axes_controls=['XZ','XYZ'],
        resource_rules='Identical to full_binary_catalog. All output H frames transported; no non-real intermediate output frame is granted.',
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest())
    (HERE/'data/real_binary_catalog.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','nodes','quantum_operation_count','homogeneous_level_nodes','variable_pump_level_nodes','tree_layer_nodes','frame_transport_checks')},indent=2))
if __name__=='__main__':main()
