"""Upgrade the fixed-input endpoint-code counterexample to all three modes.

No preceding B prefix or ED reoptimization is included in this statement.
Finite outward evaluation is combined with analytic infinite-length tails.
"""
from core import *
from fractions import Fraction as F
from compound_menu_bounds import menu_bounds
from compound_terminals import actions
import stable_intervals as iv
import hashlib


def main():
    original=OLD/'data/single_round_counterexamples_interval.json'
    old=json.loads(original.read_text())['results'][0]
    p=[F(v) for v in old['input_exact']]
    boxes=[iv.rational(v) for v in p]
    state=tuple(np.array([[v[j] for v in boxes]]) for j in (0,1))
    lo,hi=menu_bounds(state)
    menu=actions(128)
    codes={e['id']:e for e in json.loads((EXT/'data/codes.json').read_text())['entries']}
    endpoints=[]
    for i,a in enumerate(menu):
        if a['kind']=='linear':
            k=codes[a['code']]
            take=k['k'] in (1,k['n']-1)
        else:take=a['kind'] in ('SPC','repetition','Watanabe_2007')
        if take:endpoints.append(i)
    finite=max(endpoints,key=lambda i:hi[0,i])
    H=iv.positive_sum(iv.entropy_interval(state))
    # For repetition k=1: gross <= 1-H(Z|X), H(S)>=n H(X)-1.
    # Therefore compressed OTP <=1-H(p)+1/n, decreasing with n.
    N=9
    tail=iv.add(iv.sub(iv.rational(F(1)),(H[0][0],H[1][0])),iv.rational(F(1,N)))
    assert tail[1]<0
    assert float(old['SPC_tail_conditional_entropy_per_input_lower'])>1
    endpoint_upper=max(float(hi[0,finite]),float(old['repetition_tail_rate_upper']))
    lower=float(iv.down(float(old['rate_lower'])))
    assert lower>endpoint_upper
    out=dict(status='passed',input_exact=old['input_exact'],code=old['code'],
             chosen_rate_lower=lower,endpoint_all_lengths_all_modes_and_W_upper=endpoint_upper,
             largest_finite_endpoint_action=menu[finite],
             repetition_compressed_tail_start=N,repetition_compressed_tail_upper=float(tail[1]),
             repetition_no_OTP_tail_start=old['repetition_tail_start'],
             repetition_no_OTP_tail_upper=old['repetition_tail_rate_upper'],
             SPC_all_modes_tail_start=old['SPC_tail_start'],
             SPC_physical_entropy_per_input_lower=old['SPC_tail_conditional_entropy_per_input_lower'],
             original_certificate_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
             finite_menu_maximum_length=128,plain_OTP_excluded_by='pointwise domination by same-code noOTP',
             scope='Fixed Bell input and key basis; one Du code round or the Watanabe terminal. All repetition/SPC lengths and all three Du modes, including fixed time sharing. Does not include preceding pooled B rounds or changing ED/basis.')
    atomic_json(DATA/'single_round_all_modes_certified.json',out)
    print(json.dumps(out,indent=2),flush=True)


if __name__=='__main__':main()
