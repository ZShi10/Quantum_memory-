from common import *
import json
import numpy as np
from rates import CodeKernel
from algebra import compositions


def code_kernels(max_n=7,all_codes=False):
    entries=json.loads((EXT/'data/codes.json').read_text())['entries'];result=[]
    for e in entries:
        if e['n']>max_n or (not all_codes and not e['indecomposable']):continue
        cache=EXT/'data/kernels'/f"{e['id']}.npz"
        if cache.exists():
            arrays=np.load(cache);k=CodeKernel.__new__(CodeKernel);k.entry=e;k.n=e['n'];k.k=e['k']
            for key in ('types','polys','hist','multiplicity','marginal'):setattr(k,key,arrays[key])
            k.syndrome_classes=json.loads(str(arrays['syndrome_classes']))
        else:k=CodeKernel(e)
        result.append(k)
    return result


def complete_short_entries(symmetric=True):
    from itertools import permutations
    entries=json.loads((OLD/'data/blocks.json').read_text())['entries']
    memory=json.loads((DATA/'small_memory_exact.json').read_text())['entries'];seen={}
    frames=[p for p in permutations(range(4)) if set(p[i] for i in (0,3)) in ({0,3},{1,2})]
    for original in entries:
        e={**original,'M':memory[original['id']]['M']};source=np.asarray(e['tensor']);collapsed=source
        if symmetric:
            types=compositions(e['m']);pairs=sorted({(v[0],v[2]) for v in types});lookup={v:i for i,v in enumerate(pairs)}
            collapsed=np.zeros(collapsed.shape[:2]+(len(pairs),),dtype=np.int32)
            for j,typ in enumerate(types):collapsed[:,:,lookup[typ[0],typ[2]]]+=source[:,:,j]
        branches=[min(row[list(p)].tobytes() for p in frames) for row in collapsed]
        key=(e['m'],e['real'],b''.join(sorted(branches)))
        if key not in seen or (e['M'],len(e['gates']))<(seen[key]['M'],len(seen[key]['gates'])):seen[key]=e
    return list(seen.values())
