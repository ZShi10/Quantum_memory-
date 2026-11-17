"""Recompute Sun's nine-size separate-entropy envelope on the author grid.

The eight author envelopes are exact maxima of the original supplied .npy
matrices, with source hashes recorded alongside. The ninth curve is a
reconstruction, never an invented author series.
"""
from pathlib import Path
from itertools import permutations
import numpy as np
import csv,json
ROOT=Path(__file__).resolve().parents[1]
PERMS=[(0,)+p for p in permutations((1,2,3))]
def h(p):
    p=np.clip(p,0,1)
    with np.errstate(divide='ignore',invalid='ignore'):
        return -np.where(p>0,p*np.log2(p),0)-np.where(p<1,(1-p)*np.log2(1-p),0)
def gain(p):return np.maximum(0,np.max(np.stack([1-h(p[:,s[1]]+p[:,s[2]])-h(p[:,s[2]]+p[:,s[3]]) for s in PERMS]),axis=0))
def main():
    inp=ROOT/'data/literature_inputs';library=json.loads((inp/'small_clifford_library.json').read_bytes())['entries']
    f=np.linspace(.5,1,500);curves={}
    for m,n in [(2,1),(2,2),(2,3),(2,4),(3,1),(3,2),(3,3),(4,1),(4,2)]:
        name=str(m) if n==1 else str(m)+str(n);best=np.zeros(len(f))
        for entry in library:
            if entry['m']!=m:continue
            powers=np.stack([f**(m-w)*((1-f)/3)**w for w in range(m+1)],axis=1)
            raw=powers@np.asarray(entry['weights']).T;success=raw.sum(1);p=raw/success[:,None]
            for frame in PERMS:
                a,x,y,z=p[:,frame].T
                aa=(a+z)**n;bb=(x+y)**n;cc=(a-z)**n;dd=(x-y)**n
                v=np.stack((aa+cc,bb+dd,bb-dd,aa-cc),axis=1)/2
                mass=v.sum(1);out=v/mass[:,None]
                best=np.maximum(best,success*mass/(m*n)*gain(out))
        curves[name]=best
    originals=list(csv.DictReader((inp/'sun_author_envelopes.csv').open()))
    checks=[]
    for name in curves:
        if name=='42':continue
        delta=float(np.max(np.abs(curves[name]-np.array([float(r[name]) for r in originals]))))
        assert delta<2e-12,(name,delta)
        checks.append(dict(size=name,max_absolute_error=delta,status='passed'))
    env=np.max(np.stack(list(curves.values())),axis=0)
    baseline=list(csv.DictReader((ROOT/'reproduction/extensions/data/literature/werner_linear_Sun_original.csv').open()))
    assert len(baseline)==500
    plotted=np.array([float(r['rate']) for r in baseline])[::-1]
    grid=np.array([float(r['Q']) for r in baseline])[::-1]
    assert np.max(np.abs(grid-2*(1-f)/3))<1e-14
    baseline_error=float(np.max(np.abs(env-plotted)));assert baseline_error<2e-12
    with (ROOT/'data/sun_reconstructed_nine_sizes.csv').open('w',newline='') as stream:
        w=csv.writer(stream);w.writerow(['F','Q']+list(curves)+['nine_size_envelope'])
        for i in reversed(range(500)):w.writerow([f[i],2*(1-f[i])/3]+[v[i] for v in curves.values()]+[env[i]])
    report=dict(status='passed',author_checks=checks,plotted_envelope_max_error=baseline_error,
        pure_input_rate=float(env[-1]),normalization='per raw pair; minimum ED block has 2 inputs; no extra sifting factor',
        scope='Sun nine-size separate-bit/phase entropy convention; not BB84 compound minimax and not the new joint classical menu')
    (ROOT/'validation/literature_reconstruction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
