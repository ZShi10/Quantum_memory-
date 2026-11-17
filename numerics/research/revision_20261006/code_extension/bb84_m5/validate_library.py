"""Cross-enumeration and independent physical-error/compiled-POVM checks."""
from build_library import *
from build_libraries import instrument_tensor
from libraries import complete_short_entries
from math import factorial


def original_key(tensor):
    # Independent spelling of the allowed branch-frame invariant.
    result=[]
    for branch in tensor:
        transformations=[]
        for flip_x in range(2):
            for flip_z in range(2):
                for shear in range(2):
                    array=np.zeros_like(branch)
                    encoding={(0,0):0,(1,0):1,(1,1):2,(0,1):3}
                    for (x,z),old in encoding.items():
                        new=encoding[(x^flip_x,z^flip_z^(shear*x))]
                        array[new]=branch[old]
                    transformations.append(array.tobytes())
        result.append(min(transformations))
    return b''.join(sorted(result))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--permission',choices=['real','full'],default='real');parser.add_argument('--m',type=int,default=5);args=parser.parse_args()
    m=args.m;mode=args.permission;start=time.time()
    source=DATA/f'instruments_{mode}_m{m}.json';doc=json.loads(source.read_text());records=doc['entries']
    with np.load(DATA/doc['tensor_file']) as arrays:tensors=arrays['tensors']
    assert doc['status']=='enumeration_complete' and len(records)==len(tensors)
    assert hashlib.sha256((DATA/doc['tensor_file']).read_bytes()).hexdigest()==doc['tensor_sha256']
    hashes=set();max_reference=0.;checked=0
    expected=np.array([factorial(m)//np.prod([factorial(v) for v in typ]) for typ in compositions(m)],dtype=np.int64)
    for i,(e,tensor) in enumerate(zip(records,tensors)):
        assert np.array_equal(tensor.sum(axis=(0,1)),expected)
        key=original_key(tensor);assert key==tensor_key(tensor)
        assert key not in hashes;hashes.add(key)
        # Direct Python propagation is independent of the vectorized
        # construction and checks every retained class, not just winners.
        reference=instrument_tensor(m,e['checks'],e['logical_X'],e['logical_Z'])
        assert np.array_equal(reference,tensor),(e['id'],'physical polynomial mismatch')
        compiled=doc['compilers'][e['measurement_key']]
        assert e['M']==compiled['M']==compiled['M_min']
        assert e['M']==1+max((0,)+tuple(
            __import__('streaming_clifford').cut_entropies(compiled['generators'],m)[sum(1<<j for j in compiled['input_order'][:t])]
            for t in range(m)))
        exact_replay(compiled,m);checked+=1
        if (i+1)%500==0:print('validated',mode,m,i+1,'/',len(records),'seconds',round(time.time()-start,1),flush=True)
    legacy={}
    if m==4:
        old=[e for e in complete_short_entries(False) if e['m']==4 and (mode=='full' or e['real'])]
        for e in old:
            key=original_key(np.asarray(e['tensor'],dtype=np.uint16));legacy[key]=min(legacy.get(key,99),e['M'])
        assert set(legacy)==hashes,(mode,len(legacy),len(hashes),len(set(legacy)-hashes),len(hashes-set(legacy)))
        for e,tensor in zip(records,tensors):assert e['M']==legacy[original_key(tensor)]
    result=dict(status='passed',m=m,permission=mode,instrument_polynomial_checks=checked,
                exact_compiler_replays=checked,all_branch_frames_independently_checked=True,
                legacy_m4_exact_instrument_and_memory_matches=len(legacy),
                complete_basis_count=doc['complete_basis_count'],permutation_orbits=doc['basis_input_permutation_orbits'],
                distinct_instruments=len(records),minimum_memories=doc['minimum_memories'],
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),tensor_sha256=doc['tensor_sha256'],
                scope='Every stored instrument matched independent physical Pauli propagation; every saved compiler replayed exactly. The m4 regression matches the complete frozen instrument set including exact minimum memories. Completeness of m5 uses graph normal forms plus exact Lagrangian cardinality and exhaustive permutation orbits.',seconds=time.time()-start)
    save(DATA/f'validation_{mode}_m{m}.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
