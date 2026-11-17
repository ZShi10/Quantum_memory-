"""Exact source-compatible normal form for comparing explicit binary policies.

This identifies accepted law, retry cost and registered terminal width for
symmetric BB84; it is not a quotient of arbitrary quantum instruments.
"""
from functools import lru_cache
from full_clifford_model import install_nodes,normal_lookup,compose_input,axis_before
model,ops=install_nodes();lookup=normal_lookup();maps={r['name']:tuple(r['mapping']) for r in ops}
for name,events in model.EVENTS.items():
    if name not in maps:
        row=[-1]*16
        for i,j,k in events:row[4*i+j]=k
        maps[name]=tuple(row)
IDENT=(0,1,2,3);H=(0,3,2,1)

@lru_cache(None)
def normalize(tree):
    if isinstance(tree,model.Raw):return model.RAW,IDENT
    left,fl=normalize(tree.left);right,fr=normalize(tree.right)
    mapping=compose_input(maps[tree.decoder],fl,fr);name,frame=lookup[mapping]
    # Apply a source-preserving H only to a fresh raw child.
    hl=isinstance(left,model.Raw) and name[1]=='X'
    hr=isinstance(right,model.Raw) and name[2]=='X'
    if hl or hr:
        mapping=compose_input(mapping,H if hl else IDENT,H if hr else IDENT)
        name,frame=lookup[mapping]
    a,b=model.canonical(left),model.canonical(right)
    if a>b or (a==b and name[1]>name[2]):
        left,right=right,left;name='F'+name[2]+name[1]
    return model.Node(name,left,right),frame

def from_record(ed):
    if ed['family']=='explicit_binary_tree':return model.tree_from_dict(ed['tree'])
    if ed['family']=='real_decoder_recurrence':return model.homogeneous_tree(ed['path'])
    if ed['family']=='constant_raw_pumping':return model.pumping_tree(ed['primitive']*(ed['r']-1))
    raise ValueError(ed['family'])

def identity(ed):
    original=from_record(ed);tree,frame=normalize(original)
    axis=axis_before(ed['final_axis'],frame)
    assert model.terminal_memory(tree,axis)==model.terminal_memory(original,ed['final_axis'])
    return model.canonical(tree),axis

def validate():
    from fractions import Fraction as F
    from itertools import product
    from pathlib import Path
    import hashlib,json,numpy as np
    from full_clifford_model import HERE
    from literature_rates import watanabe
    raw=model.RAW;trees=[raw]
    for r in ops[9:]:trees.append(model.Node(r['name'],raw,raw))
    for i,r in enumerate(ops[9:]):trees.append(model.Node(r['name'],trees[(7*i)%54+1],raw))
    sources=np.array([[.66,.17,0.,.17],[.69,.14,.03,.14],[.77,.06,.11,.06],[.83,0.,.17,0.]])
    comparisons=0;error=0.
    for tree in trees:
        canonical,frame=normalize(tree)
        original=model.evaluate(tree,sources,'Z');normal=model.evaluate(canonical,sources,'Z')
        inv=np.argsort(frame)
        delta=float(np.max(np.abs(original['state']-normal['state'][:,inv])))
        error=max(error,delta);assert delta<3e-14
        assert np.max(np.abs(original['yield_per_raw_pair']-normal['yield_per_raw_pair']))<3e-14
        for axis in 'XYZ':
            mapped=axis_before(axis,frame)
            assert model.terminal_memory(tree,axis)==model.terminal_memory(canonical,mapped)
            p=model.evaluate(tree,sources,axis)['state'];pp=model.evaluate(canonical,sources,mapped)['state']
            assert np.max(np.abs(watanabe(p)-watanabe(pp)))<3e-13
            comparisons+=len(sources)
    report=dict(status='passed',objective_comparisons=comparisons,max_state_error=error,
        proof_inputs=['exact 1944 frame identities','exact canonical child-swap identities','pX=pZ fresh-input H symmetry'],
        scope='Source-family identity of accepted state up to a transported frame, cost and registered width. No change to membership of either real or full family.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/'data/recursive_normalization_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':
    from full_clifford_model import bootstrap,HERE
    bootstrap(HERE/'diagnostics/normalization');validate()
