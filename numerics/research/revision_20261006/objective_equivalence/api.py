"""Consume only independently validated, exact objective-equivalence maps."""
from pathlib import Path
from functools import lru_cache
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent


@lru_cache(None)
def load(source_family='equal_xz'):
    path=HERE/f'{source_family}_map.json';encoded=path.read_bytes();data=json.loads(encoded)
    validation=json.loads((HERE/f'{source_family}_validation.json').read_text())
    assert validation['status']=='passed' and validation['map_sha256']==hashlib.sha256(encoded).hexdigest()
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((HERE.parent/'resource_model').glob('*.py'))}
    assert sources==data['resource_source_sha256'], 'Resource catalog source changed; rebuild and revalidate'
    return data


def indices(permissions='real',source_family='equal_xz'):
    permission='real' if permissions=='real' else 'full'
    return np.array(load(source_family)['permissions'][permission]['representative_indices'],int)


def representative_map(permissions='real',source_family='equal_xz'):
    permission='real' if permissions=='real' else 'full'
    return np.array(load(source_family)['permissions'][permission]['representative_for_record'],int)


def fingerprint(source_family='equal_xz'):
    data=load(source_family)
    return dict(map_file=f'{source_family}_map.json',
                map_sha256=hashlib.sha256((HERE/f'{source_family}_map.json').read_bytes()).hexdigest(),
                validation_sha256=hashlib.sha256((HERE/f'{source_family}_validation.json').read_bytes()).hexdigest(),
                source_family=data['source_family'],max_leaves=data['max_leaves'],max_pump_checks=data['max_pump_checks'])
