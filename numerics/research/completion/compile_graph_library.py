from common import *
from streaming_clifford import *
from verify_streaming import dense_check,solve_combination
from graph_families import families
import json,time


def exact_replay(c,m):
    # Replay the whole Clifford, then compare all independent measured
    # parity equations; no probability data enter this check.
    transformed=[]
    for p in c['generators']:
        row=[p,0,0]
        for gate in c['gates']:row=conjugate(row,gate,m)
        assert row[0]&((1<<m)-1)==0
        transformed.append(row)
    for record in c['measurements']:
        p=s=0
        for j,row in enumerate(transformed):
            if record['original_generator_mask']>>j&1:p^=row[0];s^=row[1]
        assert p==1<<(m+record['wire']) and s==record['offset']


def main():
    old=json.loads((OLD/'data/graph_family_blocks.json').read_text())['entries']
    extended=json.loads((EXT/'data/graphs.json').read_text())['entries']
    allgraphs={};mapping={};extra={};started=time.time();dense_count=0;maxerror=0.
    for e in old+extended:
        key=f"{e['m']}:"+','.join(f'{a}-{b}' for a,b in e['edges'])
        allgraphs.setdefault(key,(e['m'],e['edges']));mapping[e['id']]=key
    # Restore templates previously rejected by the graph-frontier upper
    # bound. All m<=14 graph measurements have cut bound M<=8.
    for m in range(5,15):
        for name,edges in families(m):
            if name.startswith('random_') and int(name[7:])>=6:continue
            key=f'{m}:'+','.join(f'{a}-{b}' for a,b in edges)
            if key not in allgraphs:extra[key]=dict(m=m,graph=name,edges=edges)
            allgraphs.setdefault(key,(m,edges))
    target=DATA/'graph_memory_exact.json';saved=json.loads(target.read_text()) if target.exists() else dict(graphs={})
    for key,(m,edges) in allgraphs.items():
        if key in saved['graphs']:continue
        c=compile_measurement(graph_generators(m,edges),m,real=True);exact_replay(c,m)
        if m<=6 or (m<=8 and dense_count<100):maxerror=max(maxerror,dense_check(c,m));dense_count+=1
        saved['graphs'][key]=c
        target.write_text(json.dumps(saved,separators=(',',':')))
        print('Exact graph memory',m,c['M'],'graphs',len(saved['graphs']),'seconds',round(time.time()-started,1),flush=True)
    saved.update(status='passed',entry_to_graph=mapping,restored_templates=extra,
        dense_checks_this_run=dense_count,max_dense_error_this_run=maxerror,
        scope='Minimum peak local memory for each specified full rank-one graph-basis measurement, including incoming port. Achieved using real H/CNOT operations; arbitrary local input rotations may be applied at load without changing memory.')
    target.write_text(json.dumps(saved,separators=(',',':')))
    print('Completed',len(saved['graphs']),'graphs; restored templates',len(extra),flush=True)


if __name__=='__main__':main()
