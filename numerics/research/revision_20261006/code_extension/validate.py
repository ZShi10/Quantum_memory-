"""Independent coverage, direct physical probabilities, and interval checks."""
from api import *
from enumerate_codes import canonical,save
from itertools import permutations
from math import factorial
import hashlib,time
from otp_reference import reference


def gaussian(n,k):
    a=b=1
    for i in range(k):a*=2**(n-i)-1;b*=2**(k-i)-1
    return a//b


def orbit_count_check():
    """Independent exhaustive S8 automorphisms; orbit masses cover ALL rows."""
    n=8;perms=np.asarray(list(permutations(range(n))),dtype=np.uint16)
    words=np.arange(1<<n,dtype=np.uint16);lut=np.zeros((len(perms),len(words)),dtype=np.uint16)
    for i in range(n):lut|=((words>>i)&1)[None,:]*(np.uint16(1)<<perms[:,i,None])
    totals={k:0 for k in range(1,n+1)};records=[]
    for e in entries(all_codes=True):
        if e['n']!=n:continue
        member=np.zeros(1<<n,dtype=bool);member[span(e['rows'])]=True
        automorphisms=int(member[lut[:,e['rows']]].all(axis=1).sum())
        assert factorial(n)%automorphisms==0
        orbit=factorial(n)//automorphisms;totals[e['k']]+=orbit
        records.append(dict(code=e['id'],automorphisms=automorphisms,orbit_size=orbit))
    expected={k:gaussian(n,k) for k in range(1,n+1)}
    assert totals==expected,(totals,expected)
    save(DATA/'n8_orbit_coverage.json',dict(status='passed',method='All 8! coordinate permutations, automorphism membership tested on generator words; orbit sizes sum to Gaussian binomial counts.',
                                          labeled_totals=totals,expected_gaussian_counts=expected,classes=records))
    return dict(classes=len(records),labeled_subspaces=sum(totals.values()),by_dimension=totals)


def signed(k,states):
    g,c=OTPKernel(k.entry).components(states);gross=np.maximum(g,0).sum(-1)
    return np.column_stack((k.evaluate(states),gross-(k.n-k.k)/k.n,gross-c))


def main():
    start=time.time();coverage=orbit_count_check();print('Orbit coverage passed',coverage,flush=True)
    kernels=load_codes();allkernels=load_codes(all_codes=True)
    bycanonical={canonical(k.n,k.entry['rows']):k for k in allkernels}
    rng=np.random.default_rng(20261006)
    states=np.r_[rng.dirichlet([3,1,1,1],5),np.eye(4),[[.7,.1,.1,.1],[.5,0,0,.5],[.4,.1,.4,.1]]]
    max_reference=0.;checks=0
    for k in kernels:
        if k.n!=8:continue
        actual=signed(k,states)
        for i,p in enumerate(states):
            ref=reference(p,k.n,k.entry['rows'])
            expected=[ref['no_otp'],ref['otp_gross']-ref['otp_plain_cost'],ref['otp_gross']-ref['otp_compressed_cost']]
            error=float(np.max(abs(actual[i]-expected)));max_reference=max(error,max_reference)
            assert error<5e-12,(k.entry['id'],i,error);checks+=3
    print('Direct physical-mode checks',checks,max_reference,flush=True)
    # The new vectorized builder also reproduces frozen exact polynomial
    # tensors on every old indecomposable representative (not just rates).
    exact=0
    for k in kernels:
        if k.n>7:continue
        fresh=build_kernel(k.entry)
        for field in ('types','polys','hist','multiplicity','marginal'):
            assert np.array_equal(getattr(k,field),getattr(fresh,field)),(k.entry['id'],field)
        assert k.syndrome_classes==fresh.syndrome_classes
        exact+=1
    # Every n8 decomposable code is dominated by the SAME component
    # allocation at all states, retaining signed costs; zero factors use
    # the one-bit no-OTP action independently of the unknown state.
    directsum_max=-np.inf;directsum_checks=0
    identity=bycanonical[canonical(1,[1])]
    for k in allkernels:
        if k.n!=8 or k.entry['indecomposable']:continue
        rhs=np.zeros((len(states),3))
        for part in k.entry['components']:
            rows=[]
            for row in k.entry['rows']:
                restriction=sum(((row>>j)&1)<<i for i,j in enumerate(part))
                if restriction:rows.append(restriction)
            if rows:
                kk=bycanonical[canonical(len(part),rows)];value=signed(kk,states)
            else:value=np.repeat(signed(identity,states)[:,0,None],3,axis=1)
            rhs+=len(part)/k.n*value
        excess=float((signed(k,states)-rhs).max());directsum_max=max(excess,directsum_max)
        assert excess<5e-12,(k.entry['id'],excess);directsum_checks+=len(states)*3
    print('Fixed-mixture domination checks',directsum_checks,directsum_max,flush=True)
    vals=signed_values(states);lo,hi=signed_bounds((states,states))
    excess=max(float((lo-vals).max()),float((vals-hi).max()))
    # Outward enclosures refer to exact binary64 input values; direct
    # floating evaluation may round several ulps outside. A few-e-12
    # diagnostic tolerance is not substituted for interval proof objects.
    assert excess<5e-12,excess
    report=dict(status='passed',enumeration=coverage,complete_classes=len(allkernels),
                reduced_kernels=len(kernels),new_n8_kernels=sum(k.n==8 for k in kernels),
                direct_physical_mode_comparisons=checks,maximum_reference_error=max_reference,
                frozen_exact_polynomial_matches=exact,directsum_signed_comparisons=directsum_checks,
                maximum_directsum_domination_excess=directsum_max,
                interval_mode_comparisons=vals.size,maximum_float_enclosure_diagnostic_excess=excess,
                scope='Complete n<=8 code enumeration is proved by puncturing induction and independently checked by exact S8 orbit masses. Formula checks are diagnostics. Fixed-mixture reduction has the pointwise analytic proof in README; signed_bounds uses the existing outward engine.',
                seconds=time.time()-start,
                catalog_sha256=hashlib.sha256((DATA/'codes_complete_n8.json').read_bytes()).hexdigest(),
                sources_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')})
    save(DATA/'validation.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
