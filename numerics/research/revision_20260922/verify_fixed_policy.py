"""Independent checks of fixed BB84 policies and their interval tangents."""
from bb84_fixed import *
from ad_modes import OTPKernel


def at(poly,t):
    n=len(poly)-1
    return sum((v*IV(F(comb(n,i))*t**i*(1-t)**(n-i)) for i,v in enumerate(poly)),IV(0))


def main():
    rng=np.random.default_rng(20260923);kernels=load_codes(4,all_codes=True)
    states=rng.dirichlet(np.ones(4),20)
    physical_error=0.;checks=0
    for k in kernels:
        physical=physical_kernel(k)
        probability,hmass=physical.components_from_monomials(monomials(states,compositions(k.n)))
        gross=np.maximum(k.n*probability-hmass,0).sum(-1)/k.n
        direct=np.stack((np.maximum(0,gross-(k.n-k.k)/k.n),
                         np.maximum(0,gross+(xlog2x(probability).sum(-1))/k.n)),-1)
        target=OTPKernel(k.entry).evaluate(states)
        physical_error=max(physical_error,float(np.max(np.abs(direct-target))))
        checks+=len(states)*2
    assert physical_error<4e-13,physical_error
    entries=[e for e in complete_short_entries(False) if e['m']<=2]
    selected=[entries[0],next(e for e in entries if e['m']==2 and e['real']),
              next(e for e in entries if e['m']==2 and not e['real'])]
    point_error=0.;tangent_checks=0;feasibility_checks=0
    for entry in selected:
        b,p=.13,.19;raw=raw_polynomials(entry,b,p);exact=exact_raw(entry,b,p)
        menu=Menu(raw,entry['m'],kernels)
        for trial in range(3):
            weights=np.zeros(menu.count)
            for branch in range(len(raw)):
                for mode,alpha in zip(MODES,(.3,.2,.4)):
                    opts=[o for o in menu.options if o['branch']==branch and o['mode']==mode]
                    o=opts[int(rng.integers(len(opts)))];weights[o['alpha']]=alpha
                    weights[o['beta']]=alpha*rng.random(len(o['beta']))
            curves,strategy=exact_strategy(menu,weights,exact,entry['m'])
            for branch in range(len(raw)):
                s=[v for v in strategy if v['branch']==branch]
                assert sum(F(v['allocation_exact']) for v in s)<=1
                for v in s:assert all(0<=F(beta)<=F(v['allocation_exact']) for beta in v['retained_allocations_exact'])
                feasibility_checks+=1
            bound=sum(c.bound(c.joint,c.den)[0] for c in curves)
            for numerator in (0,1,13,37,63,91,99,100):
                t=F(numerator,100);expected=float(menu.values([float(t)])[0]@weights)
                value=IV(0)
                for c in curves:
                    _,middle=c.bound([[at(row,t)] for row in c.joint],[at(c.den,t)])
                    value=value+middle
                error=abs(expected-float(value.lo));point_error=max(point_error,error)
                assert error<4e-13,(entry['id'],trial,t,expected,value.lo)
                assert float(bound)<=expected+4e-13
                tangent_checks+=1
    out=dict(physical_entropy_comparisons=checks,max_physical_error=physical_error,
             exact_point_and_tangent_checks=tangent_checks,max_point_error=point_error,
             exact_feasibility_checks=feasibility_checks,
             note='Direct numerical entropy versus exact-rational lifted interval curves, including all OTP modes, endpoint zeros and mixed policies.')
    atomic_json(DATA/'fixed_policy_checks.json',out);print(json.dumps(out),flush=True)


if __name__=='__main__':main()
