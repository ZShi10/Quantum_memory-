"""Independent signed-rate checks for the zero-column reduction edge case."""
from core import *
from verify_compound import signed_rates


def main():
    rng=np.random.default_rng(26926)
    states=np.r_[rng.dirichlet(np.ones(4),64),np.eye(4),[[.25,.25,.25,.25]]]
    zero=CodeKernel(dict(id='zero',n=1,k=0,rows=[]))
    identity=CodeKernel(dict(id='one',n=1,k=1,rows=[1]))
    zr=signed_rates(zero,states);ir=signed_rates(identity,states)
    expected=1-entropy(states)
    assert np.max(abs(zr[:,2]-expected))<2e-12
    assert np.max(zr-ir[:,0,None])<2e-12
    comparisons=zr.size;maximum=-np.inf
    for kernel in load_codes(3,all_codes=True):
        combined=CodeKernel(dict(id='zero_plus',n=kernel.n+1,k=kernel.k,rows=kernel.entry['rows']))
        left=signed_rates(combined,states)
        right=(kernel.n*signed_rates(kernel,states)+ir[:,0,None])/(kernel.n+1)
        maximum=max(maximum,float((left-right).max()));comparisons+=left.size
    assert maximum<3e-12
    atomic_json(DATA/'zero_coordinate_checks.json',dict(status='passed',comparisons=comparisons,
           maximum_domination_excess=maximum,zero_compressed_identity_error=float(np.max(abs(zr[:,2]-expected))),
           proof='The zero factor has rates 0, <=0, and 1-H(p) in the three modes, all dominated by the fixed one-bit no-OTP action.'))
    print('Zero-column checks passed:',comparisons,'maximum excess',maximum,flush=True)


if __name__=='__main__':main()
