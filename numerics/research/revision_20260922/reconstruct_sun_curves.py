"""Independently reproduce Sun's plotted convention on the author's grid.

Check all eight available sizes before reconstructing missing 4-2-1.
This replays the plotted separate-entropy objective including final Pauli
coefficient permutations. It is distinct from a common-objective benchmark.
"""
from pathlib import Path
from itertools import permutations
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
PERMS = [(0,)+p for p in permutations((1,2,3))]


def h(p):
    p=np.clip(p,0,1)
    with np.errstate(divide='ignore',invalid='ignore'):
        return -np.where(p>0,p*np.log2(p),0)-np.where(p<1,(1-p)*np.log2(1-p),0)


def shor_gain(p):
    return np.maximum(0,np.max(np.stack([1-h(p[:,s[1]]+p[:,s[2]])-h(p[:,s[2]]+p[:,s[3]]) for s in PERMS]),axis=0))


def main():
    data=ROOT/'data';source=ROOT/'sources'/'sun'/'data'/'werner'
    library=json.loads((PROJECT/'manuscript/data/small_clifford_library.json').read_text())['entries']
    f=np.linspace(.5,1,500)
    sizes=[(2,1),(2,2),(2,3),(2,4),(3,1),(3,2),(3,3),(4,1),(4,2)]
    curves={}
    for m,n in sizes:
        name=str(m) if n==1 else str(m)+str(n)
        best=np.zeros(len(f))
        for e in library:
            if e['m']!=m:continue
            powers=np.stack([f**(m-w)*((1-f)/3)**w for w in range(m+1)],axis=1)
            raw=powers@np.asarray(e['weights']).T
            edmass=raw.sum(1);p=raw/edmass[:,None]
            for frame in PERMS:
                a,x,y,z=p[:,frame].T
                aa=(a+z)**n;bb=(x+y)**n;cc=(a-z)**n;dd=(x-y)**n
                v=np.stack((aa+cc,bb+dd,bb-dd,aa-cc),axis=1)/2
                admass=v.sum(1);out=v/admass[:,None]
                best=np.maximum(best,edmass*admass/(m*n)*shor_gain(out))
        curves[name]=best
    checks=[]
    for name,curve in curves.items():
        path=source/f'bb84_key_rate_{name}.npy'
        if not path.exists():continue
        a=np.load(path,allow_pickle=False).reshape(len(f),-1).max(1)
        delta=np.abs(curve-a)
        checks.append(dict(size=name,max_absolute_error=float(delta.max()),
                           index=int(delta.argmax()),passed=bool(delta.max()<2e-12)))
    assert len(checks)==8
    passed=all(c['passed'] for c in checks)
    report=dict(status='passed' if passed else 'mismatch_requires_investigation',checks=checks,
                scope='All available author curves on 500 original F points, separate bit/phase entropy, author final coefficient-permutation convention.',
                missing_curve_reconstruction_released=passed,
                source='Author commit 176a9cf7731b93fb2639c59ccc692b8be46915c0 plus independent small Clifford weight enumeration.',
                warning='These are Werner-input curves using the author BB84 rate formula. They are not unknown-correlation BB84 minimax curves or a common-joint-entropy baseline.')
    (data/'sun_reconstruction_audit.json').write_text(json.dumps(report,indent=2))
    if passed:
        names=list(curves)
        with (data/'sun_reconstructed_nine_sizes.csv').open('w',newline='') as file:
            writer=csv.writer(file);writer.writerow(['F','Q']+names+['nine_size_envelope'])
            for i in reversed(range(len(f))):
                values=[curves[name][i] for name in names]
                writer.writerow([f[i],2*(1-f[i])/3]+values+[max(values)])
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
