"""Cross-check accelerated kernels against independent 4**n sums."""
from core import *
from ad_modes import *
from otp_reference import reference


def main():
    rng=np.random.default_rng(230923)
    states=np.concatenate((rng.dirichlet(np.ones(4),size=15),
                           np.eye(4),np.array([[.55,.15,.15,.15],[.5,0,0,.5],[.8,0,0,.2]])))
    kernels=load_codes(5,all_codes=True)
    maximum=0.;checks=0
    for k in kernels:
        otp=OTPKernel(k.entry)
        actual=np.column_stack((k.evaluate(states),otp.evaluate(states)))
        for i,p in enumerate(states):
            r=reference(p,k.n,k.entry['rows'])
            expected=np.array([r['no_otp'],r['otp_uncompressed_net'],r['otp_compressed_net']])
            delta=float(np.max(np.abs(actual[i]-expected)))
            maximum=max(maximum,delta);checks+=1
            assert delta<2e-12,(k.entry['id'],p,actual[i],expected)
        if k.k==1 and k.entry['d']==k.n:
            delta=np.max(np.abs(repetition_rates(states,k.n)-actual))
            maximum=max(maximum,float(delta));assert delta<2e-12
        if k.k==k.n-1 and k.entry['d']==2:
            delta=np.max(np.abs(spc_otp_rates(states,k.n)-actual[:,1:]))
            maximum=max(maximum,float(delta));assert delta<2e-12
    # The known-input direct-sum proof is tested on all coordinate classes.
    allk=load_codes(7,all_codes=True)
    small=[k for k in allk if k.entry['indecomposable']]
    full=best_short(states,allk)
    reduced=best_short(states,small)
    assert np.max(full['rate']-reduced['rate'])<2e-12
    result=dict(status='passed',direct_reference_checks=checks,
                max_difference=maximum,all_short_code_classes=len(allk),
                indecomposable_classes=len(small),
                maximum_direct_sum_envelope_difference=float(np.max(np.abs(full['rate']-reduced['rate']))),
                scope='Floating implementation cross-checks. The direct-sum reduction has a separate analytic proof and applies to known-input maximization, not unidentified BB84 minimax.')
    atomic_json(DATA/'ad_modes_checks.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
