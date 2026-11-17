"""Check real-operation membership and memory of each plotted implementation."""
import json
from full_clifford_model import HERE,PROJECT,bootstrap,data
core,engine,model,ops=bootstrap(HERE/'diagnostics/portfolio_execution')
from normalize_binary import from_record,identity
paths=[parent/folder/'policies.json' for folder in ('curves_final','curves_extended')
       for parent in (HERE,PROJECT/'evidence/revision_20261022')]
path=next(p for p in paths if p.exists())
portfolio=json.loads(path.read_bytes());real={tuple(v['mapping']) for v in data()['physical_maps'] if v['real_implementable']}
implementations=0;real_nodes=0
def check_real(tree):
    global real_nodes
    if isinstance(tree,model.Raw):return
    mapping=[-1]*16
    for i,j,k in model.EVENTS[tree.decoder]:mapping[4*i+j]=k
    assert tuple(mapping) in real,tree.decoder
    real_nodes+=1;check_real(tree.left);check_real(tree.right)
for key,p in portfolio.items():
    for permission in p['permissions']:
        ed=p['implementations'][permission];tree=from_record(ed)
        assert ed['memory_upper_bound']==p['memory']==model.terminal_memory(tree,ed['final_axis'])
        assert identity(ed)==identity(p['ED'])
        if permission.startswith('real'):check_real(tree)
        if permission=='real_xz':assert ed['final_axis'] in ('X','Z')
        implementations+=1
result=dict(status='passed',policies=len(portfolio),implementations=implementations,real_node_occurrences=real_nodes,
    scope='Concrete implementation for every declared operation set; exact frame normal form, registered terminal memory, and native real accepted-map membership.')
(HERE/'diagnostics/portfolio_execution/result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
