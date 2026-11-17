"""Verify the exact relabeling used only to share known-state evaluations."""
from core import *
from known_search import canonical_classical_states
from ad_modes import OTPKernel,repetition_rates,spc_otp_rates
from literature_rates import watanabe


def main():
    rng=np.random.default_rng(9371)
    p=np.r_[rng.dirichlet([12,1,1,1],12),rng.dirichlet(np.ones(4),12),np.eye(4)]
    # The group is deliberately generated independently as affine bit/phase maps.
    labels=((0,0),(1,0),(1,1),(0,1));states=[]
    for c in (0,1):
        for a in (0,1):
            for b in (0,1):
                perm=[labels.index((x^c,z^(a*x)^b)) for x,z in labels]
                states.append(p[:,perm])
    states=np.concatenate(states);canonical=canonical_classical_states(states)
    error=0.;count=0
    for k in load_codes():
        otp=OTPKernel(k.entry)
        for aa,bb in ((k.evaluate(states),k.evaluate(canonical)),(otp.evaluate(states),otp.evaluate(canonical))):
            error=max(error,float(np.max(np.abs(aa-bb))));count+=aa.size
    for n in (8,17,64,128):
        for function in (repetition_rates,spc_otp_rates):
            a,b=function(states,n),function(canonical,n)
            error=max(error,float(np.max(np.abs(a-b))));count+=a.size
    error=max(error,float(np.max(np.abs(watanabe(states)-watanabe(canonical)))))
    assert error<2e-12,error
    out=dict(status='passed',comparisons=count,max_difference=error,
             scope='Known-input optimized flags: x -> x+c and z -> z+a*x+b; not used for unknown-state fixed policies.')
    atomic_json(DATA/'classical_frame_checks.json',out);print(json.dumps(out),flush=True)


if __name__=='__main__':main()
