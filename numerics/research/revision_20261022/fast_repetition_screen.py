"""Screen a repetition-code rate by its one-secret-bit-per-block bound.

For an unmasked [n,1] code, the rate per raw pair is at most 1/n.
If an outward lower on the raw hashing rate already exceeds 1/n,
the expensive unmasked entropy sum is unnecessary in a menu maximum.
Both paid modes are still evaluated exactly as before. The returned
individual unmasked interval [0,1/n] remains a valid enclosure, even
outside a maximum. This module does not remove a code or a length.
"""
from pathlib import Path
import hashlib,inspect,json,time
import numpy as np
import fast_long_menu as base

SOURCE=inspect.getsource(base.repetitions.py_func)
OLD='''        h0=(phases[0,0,i],phases[0,1,i]);h1=(phases[1,0,i],phases[1,1,i])
        for w in range(count):'''
NEW='''        h0=(phases[0,0,i],phases[0,1,i]);h1=(phases[1,0,i],phases[1,1,i])
        bit0=interval(ap,1,i);bit1=interval(bp,1,i)
        raw_h=add(add(h(bit0),h(bit1)),add(mul(bit0,h0),mul(bit1,h1)))
        one_bit_bound=up(1./n)
        screen=one_bit_bound<=max(0.,down(1.-raw_h[1]))
        for w in range(count):'''
START='''            cc=smul(interval(upow,n-w,i),interval(vpow,w,i));dd=smul(interval(upow,w,i),interval(vpow,n-w,i))
            mass=clip(add(aa,bb));hm=h(mass)
            raw=(clip(smul(add(aa,cc),(.5,.5))),clip(smul(add(bb,dd),(.5,.5))),
                 clip(smul(sub(bb,dd),(.5,.5))),clip(smul(sub(aa,cc),(.5,.5))))
            el=0.;eh=0.
            for value in raw:
                l,r=h(value);el+=l;eh+=r
            gain=positive(add(sub(mass,sum_adjust(el,eh,4)),hm))
            weight=weights[0,w],weights[1,w]
            l,r=mul(gain,weight);nl+=l;nh+=r'''
REPLACE='''            mass=clip(add(aa,bb));hm=h(mass)
            weight=weights[0,w],weights[1,w]
            if not screen:
                cc=smul(interval(upow,n-w,i),interval(vpow,w,i));dd=smul(interval(upow,w,i),interval(vpow,n-w,i))
                raw=(clip(smul(add(aa,cc),(.5,.5))),clip(smul(add(bb,dd),(.5,.5))),
                     clip(smul(sub(bb,dd),(.5,.5))),clip(smul(sub(aa,cc),(.5,.5))))
                el=0.;eh=0.
                for value in raw:
                    l,r=h(value);el+=l;eh+=r
                gain=positive(add(sub(mass,sum_adjust(el,eh,4)),hm))
                l,r=mul(gain,weight);nl+=l;nh+=r'''
END='''        no=sum_adjust(nl,nh,count);gross=sum_adjust(gl,gh,count);cost=sum_adjust(cl,ch,count)'''
ENDING='''        no=(0.,one_bit_bound) if screen else sum_adjust(nl,nh,count)
        gross=sum_adjust(gl,gh,count);cost=sum_adjust(cl,ch,count)'''
assert SOURCE.count(OLD)==SOURCE.count(START)==SOURCE.count(END)==1
MODIFIED=SOURCE.replace('@njit(cache=True,fastmath=False)','@njit(cache=False,fastmath=False)').replace(OLD,NEW).replace(START,REPLACE).replace(END,ENDING)
NAMESPACE=dict(vars(base));exec(compile(MODIFIED,str(Path(__file__)),'exec'),NAMESPACE)
screened_repetitions=NAMESPACE['repetitions']

def install():
    base.repetitions=screened_repetitions

def validate():
    from full_clifford_model import HERE,bootstrap
    core,engine,model,ops=bootstrap(HERE/'diagnostics/repetition_screen')
    from fast_entropy import install as entropy_install
    from fast_short_modes import install as short_install
    entropy_install();short_install();base.install()
    import long_code_bounds as long
    from bb84_fixed import physical_kernel
    rng=np.random.default_rng(2210262)
    p=rng.dirichlet([6.,.4,.4,.4],48)
    p=np.vstack((p,np.eye(4),[.25]*4,[.6,.2,0,.2],[.97,.01,.01,.01]))
    box=np.maximum(0,np.nextafter(p-1e-13,-np.inf)),np.nextafter(p+1e-13,np.inf)
    kernels=core.load_codes();physical={v.entry['id']:physical_kernel(v) for v in kernels}
    long.best_bounds(tuple(v[:1] for v in box),kernels,physical,8)
    start=time.monotonic();old=long.best_bounds(box,kernels,physical,128);oldtime=time.monotonic()-start
    reference=base.repetitions
    example=long.LongBounds(box,128)
    saved={n:example.repetition(n,False) for n in (2,3,8,16,31,64,128)}
    install()
    for n,before in saved.items():
        after=example.repetition(n,False)
        for mode,(a,b) in enumerate(zip(before,after)):
            assert np.all(a[0]<=b[1]) and np.all(b[0]<=a[1]),(n,mode)
            if mode:assert np.array_equal(a,b),(n,mode,'paid mode changed')
    start=time.monotonic();new=long.best_bounds(box,kernels,physical,128);newtime=time.monotonic()-start
    assert np.all(old[0]<=new[1]) and np.all(new[0]<=old[1])
    assert np.max(np.abs(old[0]-new[0]))<1e-11
    assert np.max(np.abs(old[1]-new[1]))<1e-11
    result=dict(status='passed',sources=len(p),codes=len(kernels),nmax=128,
        original_seconds=oldtime,screened_seconds=newtime,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        base_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
        proof='One secret bit per unmasked repetition block implies rate <=1/n. When this is below an outward hashing lower, the complete menu maximum cannot change. Paid modes are bitwise identical.',
        maximum_lower_change=float(np.max(np.abs(old[0]-new[0]))),maximum_upper_change=float(np.max(np.abs(old[1]-new[1]))))
    (HERE/'diagnostics/repetition_screen/result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':validate()
