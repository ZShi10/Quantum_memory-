"""Install complete length-eight short codes before old engine imports."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

HERE = Path(__file__).resolve().parent
PROVIDER = HERE.parent/'code_extension/api.py'


def fingerprint():
    directory = PROVIDER.parent
    files = [PROVIDER, directory/'data/codes_complete_n8.json', directory/'data/validation.json']
    files.extend((directory/'data/kernels').glob('*.npz'))
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)}


def install(core):
    assert 'compound_terminals' not in sys.modules, 'Install the provider before any terminal/search module import.'
    holder = {}

    def load_codes(max_n=8, all_codes=False):
        return holder['module'].load_codes(max_n=max_n, all_codes=all_codes)

    core.load_codes = load_codes
    spec = importlib.util.spec_from_file_location('qkd_complete_n8_provider', PROVIDER)
    provider = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = provider
    spec.loader.exec_module(provider)
    holder['module'] = provider
    import compound_terminals
    from ad_modes import MODES

    def actions(nmax=128):
        result = [dict(kind='linear', code=k.entry['id'], n=k.n, mode=mode)
                  for k in compound_terminals.kernels().values() for mode in MODES]
        for n in range(9, nmax+1):
            for kind in ('repetition', 'SPC'):
                for mode in MODES:
                    result.append(dict(kind=kind, n=n, mode=mode))
        result.append(dict(kind='Watanabe_2007', n=2, mode='no_otp_hashed'))
        return result

    compound_terminals.actions = actions
    assert len(compound_terminals.kernels()) == 132
    assert len([k for k in compound_terminals.kernels().values() if k.n == 8]) == 76
    expected = {k.entry['id'] for k in provider.load_codes()}
    assert set(compound_terminals.kernels()) == expected
    return provider


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    project = HERE.parents[2]
    sys.path.insert(0, str(project/'research/revision_20260922'))
    import core
    core.ROOT = HERE/'adapter_validation'
    core.DATA = core.ROOT/'data'
    core.DATA.mkdir(parents=True, exist_ok=True)
    provider = install(core)
    import compound_terminals as terminal
    from compound_menu_bounds import menu_bounds
    import numpy as np
    p = np.array([[.61,.17,.08,.14], [.45,.05,.45,.05]])
    menu = terminal.actions(9)
    floating = terminal.values(p, menu)
    lo, hi = menu_bounds((p,p), 9)
    assert floating.shape == lo.shape == hi.shape
    assert np.all(floating >= lo-1e-12) and np.all(floating <= hi+1e-12)
    output = dict(status='passed', actions=len(menu), short_kernels=len(terminal.kernels()),
                  comparison_count=int(floating.size), provider_files=fingerprint(),
                  scope='n8 IDs reach both floating complete menu and independent signed outward upper menu; legacy cache untouched.')
    core.atomic_json(core.DATA/'validation.json', output)
    print(json.dumps({key:value for key,value in output.items() if key != 'provider_files'}))
