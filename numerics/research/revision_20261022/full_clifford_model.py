"""Executable full-Clifford binary nodes and exact output-frame transport."""
from pathlib import Path
from functools import lru_cache
from itertools import product
import hashlib, json, os, shutil, sys
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1]
REV=PROJECT/'research/revision_20261006';OLD=PROJECT/'research/revision_20260922'
sys.dont_write_bytecode=True
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from clifford_normal_forms import AXES,BITS,PERMS,bits,outperm

def data():return json.loads((HERE/'data/clifford_normal_forms.json').read_bytes())

def install_nodes():
    sys.path.insert(0,str(REV))
    from resource_model import model
    rows=data();ops=[]
    for row in rows['normal_forms']:
        ops.append(dict(row,name='F'+row['name']))
    for i,row in enumerate(rows['physical_maps']):
        ops.append(dict(row,name=f'D{i:02d}'))
    for row in ops:
        name=row['name'];check=row['check']
        model.PRIMITIVES[name]={k:row[k] for k in ('check','logical_X','logical_Z','gates')}
        model.EVENTS[name]=tuple((i//4,i%4,k) for i,k in enumerate(row['mapping']) if k>=0)
        for axis in AXES:
            key=(bits(row['logical_X']) if axis=='X' else bits(row['logical_Z']) if axis=='Z'
                 else bits(row['logical_X'])^bits(row['logical_Z']))
            if bits(check[0]+'I') in (key,key^bits(check)):
                model.TERMINAL_REDUCTIONS[name,axis]=tuple(check)
    return model,ops

def bootstrap(job):
    job=Path(job);job.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[str(REV/'bb84_search'),str(OLD),str(REV)]
    import core
    core.ROOT=job;core.DATA=job/'data';core.DATA.mkdir(exist_ok=True)
    shutil.copyfile(OLD/'fixed_policy_lp.py',job/'fixed_policy_lp.py')
    from n8_adapter import install
    install(core)
    model,ops=install_nodes()
    from resource_model import bounds
    import bb84_recursive,interval_jets,certify_bb84_recursive
    fp,jp,ip=bb84_recursive.prepare,interval_jets.prepare,certify_bb84_recursive.prepare
    def prepare(b,p,ts,record,rounds=0):
        if record['family']!='explicit_binary_tree':return fp(b,p,ts,record,rounds)
        z=min(b,p)*core.np.asarray(ts)
        initial=core.np.stack((1-b-p+z,b-z,z,p-z),-1)
        out=model.evaluate(model.tree_from_dict(record['tree']),initial,record['final_axis'])
        state,y=out['state'],out['yield_per_raw_pair']
        for _ in range(rounds):state,s=core.b_step(state);y=y*s/2
        return state,y
    bb84_recursive.prepare=prepare
    interval_jets.prepare=lambda b,p,boxes,r,B=0: bounds.prepare_jets(b,p,boxes,r,B) if r['family']=='explicit_binary_tree' else jp(b,p,boxes,r,B)
    certify_bb84_recursive.prepare=lambda b,p,boxes,r,B=0: bounds.prepare_intervals(certify_bb84_recursive.source(b,p,boxes),r,B) if r['family']=='explicit_binary_tree' else ip(b,p,boxes,r,B)
    import bb84_joint_extension as engine
    return core,engine,model,ops

def normal_lookup():
    rows=data()['normal_forms'];lookup={}
    for row in rows:
        for perm in PERMS:
            key=outperm(row['mapping'],perm)
            assert key not in lookup
            lookup[key]=('F'+row['name'],perm)
    assert len(lookup)==54
    return lookup

def compose_input(mapping,left,right):
    return tuple(mapping[4*left[i]+right[j]] for i,j in product(range(4),repeat=2))

def axis_before(axis,frame):return 'IXYZ'[frame.index('IXYZ'.index(axis))]

def exact_frame_validation():
    model,ops=install_nodes();lookup=normal_lookup();tested=0
    for row in data()['physical_maps']:
        for a,b in product(PERMS,repeat=2):
            mapping=compose_input(row['mapping'],a,b)
            name,frame=lookup[mapping]
            cm=next(r['mapping'] for r in ops if r['name']==name)
            assert tuple(mapping)==outperm(cm,frame)
            # Weight-two parity directions are transported by the same frame.
            native_axis=next(axis for axis in AXES if (next(r['name'] for r in ops if r.get('mapping')==row['mapping']),axis) in model.TERMINAL_REDUCTIONS)
            mapped=axis_before(native_axis,frame)
            assert mapped=='Z'
            targets=model.TERMINAL_REDUCTIONS[name,mapped]
            assert targets==(axis_before(row['check'][0],a),axis_before(row['check'][1],b))
            tested+=1
    # Exact raw-H symmetry for symmetric BB84. Quotient only fresh inputs.
    def tensor(mapping):
        out=[0]*48
        for i,j in product(range(4),repeat=2):
            k=mapping[4*i+j]
            if k>=0:out[12*k+3*i+(0 if j==0 else 2 if j==2 else 1)]+=1
        return tuple(out)
    raw_reductions={}
    for a in AXES:
        mx=next(r['mapping'] for r in ops if r['name']=='F'+a+'X')
        mz=next(r['mapping'] for r in ops if r['name']=='F'+a+'Z')
        perm=next(p for p in PERMS if tensor(mx)==tensor(outperm(mz,p)))
        raw_reductions[a+'X']=dict(canonical=a+'Z',output_frame=perm)
    result=dict(status='passed',transported_maps=tested,full_input_frame_pairs=36,
        fresh_BB84_H_reductions=raw_reductions,
        invariant='Each node has the same success probability and accepted Bell law up to an explicit I-fixing output frame. Independent-child retry cost is unchanged. Frames transport local measurement axes, preserving the registered terminal reductions and register width.',
        scope='Inductive normalization for binary zero-check retained-output trees. Does not identify arbitrary public-adaptive or multiple-output instruments.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/'data/frame_transport_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def make_catalog():
    import numpy as np
    model,ops=install_nodes();opid={r['name']:i for i,r in enumerate(ops)}
    # Columns: operator, left child, right child, leaves, retained slots,
    # terminal X/Y/Z slots, and membership mask (1 homogeneous,2 pump,
    # 4 ordered tree,8 constant native Clifford pump).
    nodes=[[-1,-1,-1,1,1,1,1,1,15]];index={};levels={1:[0]}
    def add(name,left,right,membership):
        if name.startswith('F') and left>right:
            left,right=right,left;name='F'+name[2]+name[1]
        oid=opid[name];key=(oid,left,right)
        if key in index:
            idx=index[key];nodes[idx][8]|=membership;return idx
        a,b=nodes[left],nodes[right]
        w=min(max(a[4],b[4]+1),max(b[4],a[4]+1))
        terminal=[]
        for axis in AXES:
            targets=model.TERMINAL_REDUCTIONS.get((name,axis))
            terminal.append(min(w,max(a[5+AXES.index(targets[0])],b[5+AXES.index(targets[1])])) if targets else w)
        idx=len(nodes);index[key]=idx
        nodes.append([oid,left,right,a[3]+b[3],w,*terminal,membership]);return idx
    # Every arbitrary weight-two two-input Clifford word reduces to these
    # six checks on identical children, with output frames absorbed later.
    hom=['F'+a+b for a in AXES for b in AXES if a<=b];front=[0]
    hom_counts=[1]
    for depth in range(1,8):
        front=sorted({add(name,i,i,1) for i in front for name in hom})
        hom_counts.append(len(front))
    # Fresh raw H symmetry removes X versus Z only on the fresh child.
    pump=['F'+a+b for a in AXES for b in 'YZ'];front=[0];pump_counts=[1]
    for depth in range(1,8):
        front=sorted({add(name,i,0,2) for i in front for name in pump})
        pump_counts.append(len(front))
    for leaves in range(2,6):
        ids=set()
        for k in range(1,leaves):
            for left in levels[k]:
                for right in levels[leaves-k]:
                    for a,b in product(AXES,repeat=2):ids.add(add('F'+a+b,left,right,4))
        levels[leaves]=sorted(ids)
    # A fixed native decoder need not become a fixed canonical check after
    # frame transport; retain all 54 native constant maps through N=160.
    for row in ops[9:]:
        last=0
        for leaves in range(2,161):last=add(row['name'],last,0,8)
    array=np.asarray(nodes,dtype=np.int32)
    path=HERE/'data/full_binary_catalog.npz';np.savez_compressed(path,nodes=array)
    report=dict(status='complete',nodes=len(nodes),quantum_operation_count=54,
        canonical_check_count=9,homogeneous_depth=7,homogeneous_level_nodes=hom_counts,
        variable_pump_checks=7,variable_pump_level_nodes=pump_counts,
        ordered_tree_leaves=5,tree_layer_nodes={str(k):len(v) for k,v in levels.items()},
        constant_native_pump_inputs=160,constant_native_maps=54,
        final_axes=list(AXES),memory_budgets=list(range(1,9)),
        terminal_records=3*len(nodes),source='symmetric BB84; fresh raw X/Z symmetry only',
        normalization='All zero-relative-check two-input Clifford maps with weight-two check. Weight-one maps are dominated independent-child pass-throughs.',
        memory='Registered depth-first child preparation, independent retries, recursively removable terminal product measurements; peak includes incoming ports.',
        exclusions=['arbitrary recursive acceptance flags','public-dependent decoder selection','multiple quantum outputs','other tree/word/ancestry cutoffs','arbitrary retry/recycling'],
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/'data/full_binary_catalog.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def main():
    print(json.dumps(exact_frame_validation(),indent=2),flush=True)
    print(json.dumps(make_catalog(),indent=2),flush=True)
if __name__=='__main__':main()
