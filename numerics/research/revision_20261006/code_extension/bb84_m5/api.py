"""Load complete m<=5 BB84 Clifford terminal instruments for fresh searches."""
from pathlib import Path
import importlib.util
_spec=importlib.util.spec_from_file_location('bb84_m5_library_builder',Path(__file__).with_name('build_library.py'))
_builder=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_builder)
DATA,PROJECT,tensor_key=_builder.DATA,_builder.PROJECT,_builder.tensor_key
from libraries import complete_short_entries
from algebra import compositions
import json,hashlib
import numpy as np


def load_entries(permission='real',symmetric=False,include_smaller=True):
    """No write side effects. Each entry provides exact four-variable tensor.

    symmetric=True additionally identifies identical pX=pZ laws, as in the
    legacy source. Arbitrary b!=p searches MUST leave symmetric=False.
    """
    path=DATA/f'instruments_{permission}_m5.json'
    validation=DATA/f'validation_{permission}_m5.json'
    source=json.loads(path.read_text());check=json.loads(validation.read_text())
    assert source['status']=='enumeration_complete' and check['status']=='passed'
    assert check['source_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(DATA/source['tensor_file'],allow_pickle=False) as arrays:tensors=arrays['tensors']
    assert check['tensor_sha256']==hashlib.sha256((DATA/source['tensor_file']).read_bytes()).hexdigest()
    records=[]
    if include_smaller:
        records=[e for e in complete_short_entries(symmetric) if permission=='full' or e['real']]
    for e in source['entries']:
        c=source['compilers'][e['measurement_key']]
        records.append({**e,'tensor':tensors[e['tensor_index']],
                        'terminal_compilation':c,'terminal_gates':c['gates'],
                        'M_min':c['M_min'],'minimum_proved':c['minimum_proved']})
    if not symmetric:return records
    seen={}
    for e in records:
        m=e['m'];tensor=np.asarray(e['tensor'],dtype=np.uint16);types=compositions(m)
        terms=sorted({(t[0],t[2]) for t in types});lookup={v:i for i,v in enumerate(terms)}
        collapsed=np.zeros(tensor.shape[:2]+(len(terms),),dtype=np.uint16)
        for i,t in enumerate(types):collapsed[:,:,lookup[t[0],t[2]]]+=tensor[:,:,i]
        key=(m,tensor_key(collapsed))
        if key not in seen or e['M']<seen[key]['M']:seen[key]=e
    return list(seen.values())


def manifest(permission='real'):
    paths=[DATA/f'instruments_{permission}_m5.json',DATA/f'tensors_{permission}_m5.npz',DATA/f'validation_{permission}_m5.json']
    return {str(p.relative_to(PROJECT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
