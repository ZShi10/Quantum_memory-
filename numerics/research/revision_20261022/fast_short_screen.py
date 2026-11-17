"""Exact zero screening for unmasked code flags using collision entropy.

For a normalized flag spectrum lambda, H(lambda)>=-log2 sum(lambda^2).
Thus sum(joint^2)<=P(flag)^2/2^k proves [k-H(lambda)]_+=0.
Outward upper and lower endpoints prove this inequality before any entropy
term is omitted. Paid-mode signed coefficients retain their original formula.
"""
from fractions import Fraction as F
from pathlib import Path
import hashlib,inspect,json,time
import numpy as np
import fast_short_modes as base

STATS=dict(flags=0,zero_flags=0,entropy_entries=0,computed_entropy_entries=0)

def unmasked(powers,kernel):
    import stable_intervals as iv
    from interval_modes import signed_mul
    joint=iv.positive_dot(powers,kernel.polys.T)
    mass=iv.positive_dot(powers,kernel.marginal.T)
    square=iv.positive_dot(iv.mul(joint,joint),kernel.hist.T)
    threshold=iv.scale(iv.mul(mass,mass),F(1,2**kernel.k))
    zero=square[1]<=threshold[0]
    STATS['flags']+=zero.size;STATS['zero_flags']+=int(zero.sum())
    # Every polynomial entering an unresolved flag must be evaluated. The
    # integer support product is only an exact dependency mask, not a value.
    needed=(~zero).astype(np.int32)@np.asarray(kernel.hist>0,np.int32)>0
    STATS['entropy_entries']+=needed.size;STATS['computed_entropy_entries']+=int(needed.sum())
    hh=iv.entropy_interval((joint[0][needed],joint[1][needed]))
    hj=[np.zeros_like(joint[0]),np.zeros_like(joint[1])]
    for i in (0,1):hj[i][needed]=np.maximum(0.,hh[i])
    hs=iv.entropy_interval(mass);hs=tuple(np.maximum(0.,v) for v in hs)
    total=iv.positive_dot(tuple(hj),kernel.hist.T)
    gain=iv.add(iv.sub(iv.scale(mass,kernel.k),total),hs)
    gain=tuple(np.where(zero,0.,np.maximum(0.,v)) for v in gain)
    gain=signed_mul(gain,iv.rational(F(1,kernel.n)))
    gain=signed_mul(gain,(kernel.multiplicity,kernel.multiplicity))
    return iv.positive_sum(tuple(np.maximum(0.,v) for v in gain))

SOURCE=inspect.getsource(base.best_short_bounds)
OLD='gain,_=components(monomials[k.n],k);no=iv.positive_sum(positive(gain))'
assert SOURCE.count(OLD)==1
NAMESPACE=dict(vars(base),unmasked=unmasked)
exec(compile(SOURCE.replace(OLD,'no=unmasked(monomials[k.n],k)'),str(Path(__file__)),'exec'),NAMESPACE)
best_short_bounds=NAMESPACE['best_short_bounds']

def install():
    import interval_modes
    interval_modes.best_short_bounds=best_short_bounds

def validate():
    from full_clifford_model import HERE,bootstrap
    core,engine,model,ops=bootstrap(HERE/'diagnostics/short_screen')
    from fast_entropy import install as entropy_install
    from fast_long_menu import install as long_install
    entropy_install();long_install()
    import stable_intervals as iv
    from interval_modes import components
    from bb84_fixed import physical_kernel
    from long_code_bounds import best_bounds
    kernels=core.load_codes();physical={k.entry['id']:physical_kernel(k) for k in kernels}
    rng=np.random.default_rng(2210263)
    p=np.vstack((rng.dirichlet([2.,2.,2.,2.],48),rng.dirichlet([9.,.2,.2,.2],24),
        rng.dirichlet([3.,.1,.1,3.],24),np.eye(4),[.25]*4,[.51,.001,.001,.488]))
    box=np.maximum(0,np.nextafter(p-1e-13,-np.inf)),np.nextafter(p+1e-13,np.inf)
    powers={n:iv.monomial_bounds(box,np.asarray(core.compositions(n))) for n in {k.n for k in kernels}}
    for k in kernels:
        g,_=components(powers[k.n],k);old=iv.positive_sum(tuple(np.maximum(0.,v) for v in g));new=unmasked(powers[k.n],k)
        assert np.all(old[0]<=new[1]) and np.all(new[0]<=old[1]),k.entry['id']
    results={}
    for name,installer in [('original',base.install),('screened',install)]:
        installer();start=time.monotonic();value=best_bounds(box,kernels,physical,128)
        results[name]=dict(seconds=time.monotonic()-start,bounds=value)
    a=results['original']['bounds'];b=results['screened']['bounds']
    assert np.all(a[0]<=b[1]) and np.all(b[0]<=a[1])
    assert np.max(np.abs(a[0]-b[0]))<1e-11
    report=dict(status='passed',sources=len(p),codes=len(kernels),nmax=128,
        seconds={k:v['seconds'] for k,v in results.items()},screening=STATS,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        base_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
        scope='Only mathematically zero positive-part unmasked flags are omitted. Every unresolved entropy, every paid coefficient, every code and length remains included.')
    (HERE/'diagnostics/short_screen/result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':validate()
