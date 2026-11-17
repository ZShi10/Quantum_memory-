"""Frozen structured/random larger graphs, exact memory, selected duals.

Extends actual implemented memory after discovering old frontier budgets
overcounted it. This is a constructed family, not a complete large-m class.
"""
from common import *
import json,time,random,importlib.util,hashlib
import numpy as np
from streaming_clifford import compile_measurement,graph_generators
from compile_graph_library import exact_replay
from graph_families import families,gf2_rank


def load_character():
    spec=importlib.util.spec_from_file_location('completion_character',EXT/'graph_character.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def entropies(m,edges):
    adj=[0]*m
    for a,b in edges:adj[a]|=1<<b;adj[b]|=1<<a
    allbits=(1<<m)-1;out=[0]*(1<<m)
    for s in range(1,1<<(m-1)):
        t=allbits^s;rows=[adj[j]&t for j in range(m) if s>>j&1];r=gf2_rank(rows)
        out[s]=r;out[t]=r
    return out


def main():
    character=load_character();dest=DATA/'memory_eight_graphs.json'
    saved=json.loads(dest.read_text()) if dest.exists() else dict(graphs={},entries=[])
    started=time.time();count=0
    # The catalogue is fixed before any key-rate evaluations.
    for m in (15,16,17,18,19,20):
        for name,edges in families(m):
            if name not in ('cycle','random_1','random_2'):continue
            gid=f'LARGE_{m}_{name}'
            if gid in saved['graphs']:continue
            ent=entropies(m,edges);compiler=compile_measurement(graph_generators(m,edges),m,real=True,entropies=ent);exact_replay(compiler,m)
            M=compiler['M'];saved['graphs'][gid]=dict(m=m,edges=edges,compilation=compiler)
            if M<=8:
                rng=random.Random(int(hashlib.sha256(gid.encode()).hexdigest()[:12],16))
                vectors=sorted(set([(1<<m)-1,(1<<(m//2))-1,sum(1<<j for j in range(0,m,2))]+rng.sample(range(1,1<<m),3)))
                for v in vectors:
                    counts=character.enumerator(m,edges,v);weights=np.zeros((4,m+1),dtype=np.int64)
                    from algebra import compositions
                    for j,typ in enumerate(compositions(m)):weights[:,m-typ[0]]+=counts[:,j]
                    saved['entries'].append(dict(id=f'{gid}_v{v}',graph=gid,m=m,M=M,logical_dual_vector=v,
                                                character_weights=weights.tolist(),characters=counts.tolist(),real=True,
                                                ED_acceptance='Zero relative syndrome; each complete attempt costs m raw pairs.'))
                    count+=1
            character.base_arrays.cache_clear()
            saved.update(scope='Frozen cycle and two seeded random graph families m=15..20; six noise-independent dual choices per graph. Exact minimum for each full graph measurement, budget M<=8. Achievable constructed family, not complete Clifford enumeration.')
            dest.write_text(json.dumps(saved,separators=(',',':')))
            print('Larger actual memory',gid,'M',M,'entries',len(saved['entries']),'seconds',round(time.time()-started,1),flush=True)
    saved['status']='passed';saved['seconds_this_run']=time.time()-started;dest.write_text(json.dumps(saved,separators=(',',':')))


if __name__=='__main__':main()
