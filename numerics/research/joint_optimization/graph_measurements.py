"""Terminal stabilizer measurements through local Clifford graph forms.

The classical output labeling is unrestricted and must be stored separately
from the quantum measurement. This can use less memory than decoding a
particular logical qubit before moving the final measurement cut.
"""
from pathlib import Path
from itertools import product,permutations,combinations
import json
from algebra import rref
from circuits import apply_gate,measurement_cut

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data'
LOCAL=((),('h',),('s',),('h','s'),('s','h'),('h','s','h'))


def graph_order(m,edges):
    adjacency=[set() for _ in range(m)]
    for a,b in edges:adjacency[a].add(b);adjacency[b].add(a)
    best=None
    for order in permutations(range(m)):
        visited=set();frontier=set();peak=0
        for v in order:
            peak=max(peak,len(frontier)+1);visited.add(v);frontier.add(v)
            frontier={u for u in frontier if adjacency[u]-visited}
        if best is None or peak<best[0]:best=(peak,order)
    return best


def graph_circuit(m,edges,order,local):
    adjacency=[set() for _ in range(m)]
    for a,b in edges:adjacency[a].add(b);adjacency[b].add(a)
    visited=set();live=set();gates=[]
    for v in order:
        for op in LOCAL[local[v]]:gates.append((op,v))
        for u in sorted(adjacency[v]&visited):
            gates.extend([('h',v),('cx',u,v),('h',v)])
        visited.add(v);live.add(v)
        for u in sorted(q for q in live if not(adjacency[q]-visited)):
            gates.append(('h',u));live.remove(u)
    return gates


def build(max_m=4):
    libraries={}
    for m in range(1,max_m+1):
        possible_edges=list(combinations(range(m),2))
        local_maps=[]
        for local in product(range(6),repeat=m):
            mapping=list(range(1<<(2*m)))
            for v,j in enumerate(local):
                # Inverse symplectic map; S and S^-1 agree modulo phase.
                for op in reversed(LOCAL[j]):mapping=[apply_gate(p,(op,v),m) for p in mapping]
            local_maps.append((local,mapping))
        full={};real={}
        for bits in range(1<<len(possible_edges)):
            edges=[e for j,e in enumerate(possible_edges) if bits>>j&1]
            generators=[1<<j for j in range(m)]
            for a,b in edges:generators[a]^=1<<(m+b);generators[b]^=1<<(m+a)
            width,order=graph_order(m,edges)
            for local,mapping in local_maps:
                basis=rref([mapping[g] for g in generators],2*m)
                real_local=all(j<2 for j in local)
                cost=(width,sum(len(LOCAL[j]) for j in local)+3*len(edges))
                record=dict(m=m,graph_edges=edges,input_order=order,local_choices=local,M=width,cost=cost)
                if basis not in full or cost<tuple(full[basis]['cost']):full[basis]=record
                if real_local and (basis not in real or cost<tuple(real[basis]['cost'])):real[basis]=record
        expected=1
        for j in range(1,m+1):expected*=2**j+1
        assert len(full)==expected,(m,len(full),expected)
        for mode,lib in [('full',full),('real',real)]:
            for basis,record in lib.items():
                gates=graph_circuit(m,record['graph_edges'],record['input_order'],record['local_choices'])
                record['gates']=gates;record['cut']=measurement_cut(gates,m)
                assert record['cut']['M']<=record['M']
                for generator in basis:
                    out=generator
                    for g in gates:out=apply_gate(out,g,m)
                    assert out&((1<<m)-1)==0
            libraries[f'{mode}_{m}']={'_'.join(map(str,key)):value for key,value in lib.items()}
        print('graph measurements',m,'all Lagrangians',len(full),'real',len(real),
              'full widths',sorted({v['M'] for v in full.values()}),'real widths',sorted({v['M'] for v in real.values()}),flush=True)
    return libraries


def main():
    DATA.mkdir(exist_ok=True)
    result=build(4)
    (DATA/'graph_measurements.json').write_text(json.dumps(result,separators=(',',':')))

if __name__=='__main__':main()
