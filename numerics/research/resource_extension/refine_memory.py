"""Local-complementation memory refinement for selected graph measurements.

Every compiled change is checked on all m stabilizer generators. The found
memory is an implementation bound; equality with the cut-rank bound is
reported only when achieved. Original numerical search snapshots are kept.
"""
from common import *
import json,time,heapq
from graph_families import graph_widths
from circuits import apply_gate,measurement_cut
from algebra import rref


def complement(m,edges,v):
    edges=set(map(tuple,edges));neighbors=[j for j in range(m) if tuple(sorted((j,v))) in edges]
    for i,a in enumerate(neighbors):
        for b in neighbors[i+1:]:
            pair=tuple(sorted((a,b)))
            if pair in edges:edges.remove(pair)
            else:edges.add(pair)
    return tuple(sorted(edges)),neighbors


def compile_graph(m,edges,order,ops):
    adjacency=[set() for _ in range(m)]
    for a,b in edges:adjacency[a].add(b);adjacency[b].add(a)
    visited=set();live=set();gates=[]
    for v in order:
        gates.extend((op,v) for op in ops[v])
        for u in sorted(adjacency[v]&visited):gates.extend([('h',v),('cx',u,v),('h',v)])
        visited.add(v);live.add(v)
        for u in sorted(q for q in live if not(adjacency[q]-visited)):
            gates.append(('h',u));live.remove(u)
    return gates


def refine(entry,max_graphs=120):
    m=entry['m'];original=tuple(map(tuple,entry['edges']));width=graph_widths(m,original)
    best=(width['M'],original,[],width);heap=[(width['M'],0,original,[])];seen={original};counter=1
    while heap and best[0]>entry['schmidt_memory_lower'] and len(seen)<max_graphs:
        _,_,edges,path=heapq.heappop(heap)
        for v in range(m):
            new,_=complement(m,edges,v)
            if new in seen:continue
            seen.add(new);w=graph_widths(m,new);p=path+[v]
            if w['M']<best[0]:best=(w['M'],new,p,w)
            heapq.heappush(heap,(w['M'],counter,new,p));counter+=1
            if best[0]==entry['schmidt_memory_lower'] or len(seen)>=max_graphs:break
    upper,edges,path,width=best;ops=[[] for _ in range(m)];current=original
    for v in path:
        current,neighbors=complement(m,current,v);ops[v].extend(('h','s','h'))
        for j in neighbors:ops[j].append('s')
    assert current==edges
    gates=compile_graph(m,edges,width['input_order'],ops);cut=measurement_cut(gates,m);masks=[];offsets=[]
    for j in range(m):
        pauli=1<<j
        for a,b in original:
            if a==j:pauli^=1<<(m+b)
            if b==j:pauli^=1<<(m+a)
        sign=0
        for g in gates:
            op,a,*rest=g;x=(pauli>>a)&1;z=(pauli>>(m+a))&1
            if op in ('h','s'):sign^=x&z
            elif op=='cx':
                b=rest[0];xb=(pauli>>b)&1;zb=(pauli>>(m+b))&1
                sign^=x&zb&(xb^z^1)
            pauli=apply_gate(pauli,g,m)
        assert pauli&((1<<m)-1)==0
        masks.append(pauli>>m);offsets.append(sign)
    assert entry['schmidt_memory_lower']<=cut['M']<=upper
    return dict(original_M=entry['M'],M_full=upper,schmidt_lower=entry['schmidt_memory_lower'],
                minimum_proved=upper==entry['schmidt_memory_lower'],graphs_examined=len(seen),
                graph_edges=edges,local_complementation_vertices=path,input_order=width['input_order'],
                per_wire_preoperations=ops,graph_measurement_gates=gates,measurement_cut=cut,
                original_graph_output_masks=masks,original_graph_output_offsets=offsets,
                instruction='At each input load, apply the original input-axis rotation and original local H first, then these preoperations. Bob uses the conjugate Clifford. The output masks recover all original graph measurement bits; use the original dual vector to form the key and ED checks.')


def main():
    graphs={e['id']:e for e in json.loads((DATA/'graphs.json').read_text())['entries']};ids=set()
    six=json.loads((DATA/'six_state.json').read_text())
    ids.update(v['ED'] for p in six['results'] for v in p['results'].values() if v['ED'].startswith('XG'))
    small_ids={v['ED'] for p in six['results'] for v in p['results'].values() if v['ED'].startswith('E')}
    for path in DATA.glob('bb84_[0-9]*.json'):
        data=json.loads(path.read_text())
        ids.update(e['graph_id'] for e in data['protocol_library'].values() if e['source']=='extended_graph')
        small_ids.update(e['id'] for e in data['protocol_library'].values() if e['source']=='complete_short')
    # Include inherited witnesses so every catalog entry has an explicit
    # measurement implementation, not just an existence bound for small m.
    old_graphs={e['id']:e for e in json.loads((OLD/'data/graph_family_blocks.json').read_text())['entries']}
    graphs.update(old_graphs)
    previous=json.loads((OLD/'data/werner_joint_k12f.json').read_text())['results']
    for mode,records in previous.items():
        if not mode.startswith('full'):continue
        ids.update(e['ED'] for e in records if e['ED'] in old_graphs)
        small_ids.update(e['ED'] for e in records if e['ED'].startswith('E'))
    previous=json.loads((OLD/'data/all_policy_interval_certificates.json').read_text())['results']
    for record in previous:
        for policy in record['results'].values():
            small_ids.update(c['ED'] for c in policy.get('cells',[]) if c['ED'].startswith('E'))
    out=DATA/'memory_refinement.json';saved=json.loads(out.read_text()) if out.exists() else dict(results={})
    cache={};started=time.time()
    for gid in sorted(ids):
        if gid in saved['results'] and 'original_graph_output_offsets' in saved['results'][gid]:continue
        e=graphs[gid];key=(e['m'],tuple(map(tuple,e['edges'])))
        if key not in cache:cache[key]=refine(e)
        saved['results'][gid]=cache[key]
        out.write_text(json.dumps(saved,separators=(',',':')))
        print('Memory',gid,e['m'],e['M'],'->',cache[key]['M_full'],'lower',cache[key]['schmidt_lower'],'seconds',round(time.time()-started,1),flush=True)
    small={e['id']:e for e in json.loads((OLD/'data/blocks.json').read_text())['entries']}
    full=json.loads((OLD/'data/graph_measurements.json').read_text());small_results={}
    for sid in sorted(small_ids):
        e=small[sid];m=e['m'];basis=rref(e['checks']+[e['logical_Z']],2*m)
        g=full[f'full_{m}']['_'.join(map(str,basis))];masks=[];offsets=[]
        for observable in [e['logical_Z']]+e['checks']:
            pauli=observable;sign=0
            for gate in g['gates']:
                op,a,*rest=gate;x=(pauli>>a)&1;z=(pauli>>(m+a))&1
                if op in ('h','s'):sign^=x&z
                elif op=='cx':
                    b=rest[0];sign^=x*((pauli>>(m+b))&1)*(((pauli>>b)&1)^z^1)
                pauli=apply_gate(pauli,gate,m)
            assert pauli&((1<<m)-1)==0;masks.append(pauli>>m);offsets.append(sign)
        small_results[sid]=dict(M_full=g['M'],gates=g['gates'],cut=g['cut'],key_then_check_masks=masks,offsets=offsets)
    saved['small_full']=small_results;saved['status']='passed';out.write_text(json.dumps(saved,separators=(',',':')))


if __name__=='__main__':main()
