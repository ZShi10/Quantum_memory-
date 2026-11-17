"""Exact bit-weight formula for paid OTP coefficients of short linear codes.

For a public syndrome s, c_s(w) counts bit-error words of weight w. Conditional
on that word the phase bits are independent, with entropies h0 and h1.
Thus H(X^n,Z^n,s) on this flag is the bit-word entropy plus
sum_w c_s(w) a^(n-w)b^w [(n-w)h0+w h1]. No phase-word enumeration is needed.
Unmasked coded-phase entropy remains the original complete polynomial sum.
"""
from functools import lru_cache
from fractions import Fraction as F
from math import comb
import numpy as np

def best_short_bounds(state,kernels,physical,batch=256,capture=None):
    import stable_intervals as iv
    from core import compositions
    from interval_modes import components,signed_mul
    from long_code_bounds import clip,h,positive,scale,stack,column,powers
    counts={}
    for k in kernels:
        types=compositions(k.n);lookup={tuple(v):i for i,v in enumerate(types)}
        cc=k.marginal[:,[lookup[k.n-w,w,0,0] for w in range(k.n+1)]]
        assert np.array_equal(cc.T@k.multiplicity,np.array([comb(k.n,w) for w in range(k.n+1)]))
        counts[k.entry['id']]=cc
    lower=np.zeros(len(state[0]));upper=lower.copy()
    for start in range(0,len(lower),batch):
        end=min(start+batch,len(lower));p=tuple(v[start:end] for v in state)
        a=clip(iv.add(column(p,0),column(p,3)));b=clip(iv.add(column(p,1),column(p,2)))
        phases=[];entropy=h(p)
        for pair,mass in (((0,3),a),((1,2),b)):
            numerator=positive(iv.sub(iv.add(column(entropy,pair[0]),column(entropy,pair[1])),h(mass)))
            good=mass[0]>0;safe=(np.where(good,mass[0],1.),np.where(good,mass[1],1.))
            conditional=clip(iv.div(numerator,safe))
            phases.append((np.where(good,conditional[0],0.),np.where(good,conditional[1],1.)))
        maximum=max(k.n for k in kernels);ap=powers(a,maximum);bp=powers(b,maximum)
        words={n:stack([iv.mul(ap[n-w],bp[w]) for w in range(n+1)]) for n in {k.n for k in kernels}}
        word_h={n:h(v) for n,v in words.items()}
        monomials={n:iv.monomial_bounds(p,np.asarray(compositions(n))) for n in words}
        for k in kernels:
            gain,_=components(monomials[k.n],k);no=iv.positive_sum(positive(gain))
            cc=counts[k.entry['id']];w=np.arange(k.n+1)
            mass=clip(iv.positive_dot(words[k.n],cc.T));hs=h(mass)
            raw_bit_h=iv.positive_dot(word_h[k.n],cc.T)
            zero_weight=iv.positive_dot(words[k.n],(cc*(k.n-w)).T)
            one_weight=iv.positive_dot(words[k.n],(cc*w).T)
            phase0=iv.mul(zero_weight,tuple(v[:,None] for v in phases[0]))
            phase1=iv.mul(one_weight,tuple(v[:,None] for v in phases[1]))
            raw=iv.add(raw_bit_h,iv.add(phase0,phase1))
            gains=iv.add(iv.sub(scale(mass,k.n),raw),hs)
            gross=scale(iv.positive_dot(positive(gains),k.multiplicity),F(1,k.n))
            cost=scale(iv.positive_dot(hs,k.multiplicity),F(1,k.n))
            plain=iv.sub(gross,iv.rational(F(k.n-k.k,k.n)));compressed=iv.sub(gross,cost)
            if capture is not None:
                assert start==0 and end==len(state[0])
                capture[k.entry['id']]=(no,plain,compressed)
            for lo,hi in (no,plain,compressed):
                lower[start:end]=np.maximum(lower[start:end],lo);upper[start:end]=np.maximum(upper[start:end],hi)
    return lower,upper

def install():
    import interval_modes
    interval_modes.best_short_bounds=best_short_bounds

def validate():
    import hashlib,json,time
    from pathlib import Path
    from full_clifford_model import HERE,bootstrap
    core,engine,model,ops=bootstrap(HERE/'diagnostics/fast_short_modes')
    from bb84_fixed import physical_kernel
    import interval_modes
    from fast_entropy import install as fast_entropy_install
    fast_entropy_install()
    ks=core.load_codes();physical={k.entry['id']:physical_kernel(k) for k in ks}
    rng=np.random.default_rng(20261022);states=rng.dirichlet([.5,.5,.5,.5],48)
    states=np.vstack((states,np.eye(4),[.7,.1,.1,.1],[.51,.16,.16,.17],[.61,.18,.03,.18]))
    lower=np.maximum(0,np.nextafter(states-1e-10,-np.inf));upper=np.nextafter(states+1e-10,np.inf)
    enclosure=lower,upper;start=time.monotonic();old=interval_modes.best_short_bounds(enclosure,ks,physical);oldtime=time.monotonic()-start
    saved={};start=time.monotonic();new=best_short_bounds(enclosure,ks,physical,capture=saved);newtime=time.monotonic()-start
    assert np.all(old[0]<=new[1]) and np.all(new[0]<=old[1]),np.max(new[0]-old[1])
    # Check each paid mode separately, including negative net coefficients.
    import stable_intervals as iv
    from core import compositions
    powers={n:iv.monomial_bounds(enclosure,np.asarray(compositions(n))) for n in {k.n for k in ks}}
    for k in ks:
        gains,cost=interval_modes.components(powers[k.n],physical[k.entry['id']])
        gross=iv.positive_sum((np.maximum(0,gains[0]),np.maximum(0,gains[1])))
        original=(iv.sub(gross,iv.rational(F(k.n-k.k,k.n))),iv.sub(gross,cost))
        for mode,(a,b) in enumerate(zip(original,saved[k.entry['id']][1:])):
            assert np.all(a[0]<=b[1]) and np.all(b[0]<=a[1]),(k.entry['id'],mode)
    result=dict(status='passed',codes=len(ks),sources=len(states),old_seconds=oldtime,new_seconds=newtime,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        identity='Exact syndrome bit-weight enumerators; sum_s multiplicity_s c_s(w)=binomial(n,w). Conditional phase entropy is additive given the entire bit-error word.',
        scope='All unmasked polynomial coefficients unchanged; both paid OTP modes use the exact entropy identity with outward arithmetic.')
    path=HERE/'diagnostics/fast_short_modes/result.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':validate()
