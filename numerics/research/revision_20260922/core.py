"""Paths and reproducible numerical runtime for the current revision."""
from pathlib import Path
import os
import sys
import json

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
OLD = ROOT.parent/'joint_optimization'
COMPLETE = ROOT.parent/'completion'
EXT = ROOT.parent/'resource_extension'
DATA = ROOT/'data'
DATA.mkdir(exist_ok=True)
sys.path.insert(0, str(OLD))
sys.path.insert(1, str(COMPLETE))
import numpy as np
from rates import CodeKernel, entropy, monomials, xlog2x
from algebra import compositions, span


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.writing.json')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                              default=lambda v: v.item()), encoding='utf-8')
    temp.replace(path)


def load_codes(max_n=7, all_codes=False):
    entries = json.loads((EXT/'data/codes.json').read_text())['entries']
    selected = [e for e in entries if e['n'] <= max_n and (all_codes or e['indecomposable'])]
    kernels = []
    cache = DATA/'kernels'
    cache.mkdir(exist_ok=True)
    for entry in selected:
        path = EXT/'data/kernels'/f"{entry['id']}.npz"
        if not path.exists():
            path = cache/f"{entry['id']}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as arrays:
                k = CodeKernel.__new__(CodeKernel)
                k.entry, k.n, k.k = entry, entry['n'], entry['k']
                for key in ('types','polys','hist','multiplicity','marginal'):
                    setattr(k, key, arrays[key])
                k.syndrome_classes = json.loads(str(arrays['syndrome_classes']))
        else:
            k = CodeKernel(entry)
            np.savez_compressed(path, types=k.types, polys=k.polys, hist=k.hist,
                                multiplicity=k.multiplicity, marginal=k.marginal,
                                syndrome_classes=json.dumps(k.syndrome_classes))
        kernels.append(k)
    return kernels


def binary_entropy(p):
    p = np.clip(np.asarray(p), 0, 1)
    return -xlog2x(p)-xlog2x(1-p)


def b_step(p):
    a,x,y,z = np.moveaxis(np.asarray(p), -1, 0)
    raw = np.stack((a*a+z*z, x*x+y*y, 2*x*y, 2*a*z), -1)
    success = raw.sum(-1)
    return raw/success[...,None], success


def bell_parameters(p):
    a = p[...,0]+p[...,3]
    b = p[...,1]+p[...,2]
    c0 = np.divide(p[...,0]-p[...,3], a, out=np.zeros_like(a), where=a>0)
    c1 = np.divide(p[...,1]-p[...,2], b, out=np.zeros_like(b), where=b>0)
    return a,b,c0,c1
