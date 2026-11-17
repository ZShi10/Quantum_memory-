"""Diagnostic for the analytic plain-OTP dominance proposition."""
from core import *
from compound_terminals import values,actions
from ad_modes import OTPKernel


def main():
    rng=np.random.default_rng(20260926)
    p=np.vstack((rng.dirichlet([.4]*4,64),np.eye(4),np.full((1,4),.25)))
    menu=actions(20);vv=values(p,menu);lookup={json.dumps({k:v for k,v in a.items() if k!='mode'},sort_keys=True):{} for a in menu}
    for i,a in enumerate(menu):lookup[json.dumps({k:v for k,v in a.items() if k!='mode'},sort_keys=True)][a['mode']]=i
    checks=0;largest=-np.inf
    for modes in lookup.values():
        if 'otp_plain' not in modes:continue
        gap=vv[:,modes['otp_plain']]-vv[:,modes['no_otp']]
        assert gap.max()<2e-12,gap.max();largest=max(largest,float(gap.max()));checks+=len(p)
    out=dict(status='passed',comparisons=checks,largest_plain_minus_no=largest,
             proof='Entropy under deterministic (X,Z)->(X,GZ), [x+c]_+-c<=[x]_+ for c>=0. Diagnostic supports the implementation; the inequality is analytic for all codes and inputs.')
    atomic_json(DATA/'plain_otp_dominance_checks.json',out);print(json.dumps(out),flush=True)


if __name__=='__main__':main()
