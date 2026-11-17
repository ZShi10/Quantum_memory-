"""Optional future-run adapter; no production driver is patched or imported.

The original candidate set remains the declared class. Each removed record
has a proved objective-equivalent representative with no larger executable
memory. This applies to equal-X/Z source families only.
"""
from pathlib import Path
from functools import lru_cache
import os,sys,json,hashlib
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
HERE=Path(__file__).resolve().parent;REVISION=HERE.parent
sys.path.insert(0,str(REVISION))
sys.path.insert(0,str(REVISION.parent/'revision_20260922'))
import numpy as np
from recurrence_equivalence import representatives
from recurrence_interval_library import records as legacy_records
from objective_equivalence.api import representative_map,fingerprint
from resource_model.catalog import candidate_records


@lru_cache(None)
def full_catalog_aliases():
    old_memory,old=legacy_records('full',7,160)
    old_alias=representatives(old,symmetric=True)
    extension=list(candidate_records(5,7,'full'))
    extension_memory=np.array([record['memory_upper_bound'] for _,record in extension])
    extension_alias=representative_map('full','equal_xz')
    offset=len(old);aliases=np.r_[old_alias,offset+extension_alias]
    memory=np.r_[old_memory,extension_memory];keep=np.unique(aliases)
    assert offset==67455 and len(extension)==76335
    assert np.array_equal(aliases[aliases],aliases)
    assert np.all(memory[aliases]<=memory)
    provenance=dict(source_family='equal_xz',legacy_prefix=offset,extension_count=len(extension),
        legacy_representatives=int(len(np.unique(old_alias))),extension_representatives=len(np.unique(extension_alias)),
        original_candidates=len(aliases),retained_candidates=len(keep),
        extension_map=fingerprint('equal_xz'),
        legacy_symmetry_source_sha256=hashlib.sha256((REVISION.parent/'revision_20260922/recurrence_equivalence.py').read_bytes()).hexdigest(),
        proof='Exact state/yield legacy symmetry plus validated exact polynomial extension map; all B prefixes and registered terminal objectives are preserved; chosen representative memory is no larger.')
    return aliases,keep,memory,provenance


def restrict_full_arrays(probability,yield_per_raw,memory):
    """Accept floating arrays or two endpoint arrays in full original order.

    The caller must establish pX=pZ and record returned provenance. Selection
    of interval representatives preserves a valid enclosure for each exact
    alias objective; it need not contain every alias's *looser computed box*.
    """
    aliases,keep,expected,provenance=full_catalog_aliases()
    assert np.array_equal(memory,expected), 'Full original catalog order required'
    def take(a):return tuple(v[keep] for v in a) if isinstance(a,tuple) else a[keep]
    return take(probability),take(yield_per_raw),memory[keep],keep,provenance


if __name__=='__main__':
    aliases,keep,memory,provenance=full_catalog_aliases()
    with (HERE/'werner_quotient_aliases.npz').open('wb') as stream:
        np.savez_compressed(stream,aliases=aliases,representatives=keep,memory=memory)
    (HERE/'werner_quotient_manifest.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in provenance.items() if k not in ('extension_map','proof')}),flush=True)
