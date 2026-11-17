"""Check entropy enclosure identities independently on random states."""
from pathlib import Path
import json
import numpy as np
from rates import CodeKernel,entropy,monomials
from bernstein import monomial_coefficients,evaluate,split,GainCurve,certify_minimum
from algebra import compositions

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data'


def main():
    rng=np.random.default_rng(2700919)
    entries=json.loads((DATA/'codes.json').read_text())['entries']
    checks=0;worst=0.
    for entry in entries:
        kernel=CodeKernel(entry)
        ends=rng.dirichlet([3,1,1,1],size=2)
        raw=ends.T
        curve=GainCurve(raw,kernel)
        lower,upper=curve.interval()
        for t in np.linspace(0,1,17):
            p=(1-t)*ends[0]+t*ends[1]
            ps,hmass=kernel.components(p)
            value=np.dot(kernel.multiplicity,kernel.k*ps-hmass)/kernel.n
            got=curve.value(t)
            worst=max(worst,abs(value-got))
            assert abs(value-got)<2e-12,(entry,value,got)
            assert lower-2e-12<=got<=upper+2e-12
            checks+=1
        for side,(joint,den) in enumerate(zip(split(curve.joint),split(curve.denominator))):
            lo,hi=curve.interval(joint,den)
            for t in np.linspace(side*.5,(side+1)*.5,9):
                assert lo-2e-12<=curve.value(t)<=hi+2e-12
                checks+=1
    record=dict(status='passed',enclosure_checks=checks,maximum_direct_discrepancy=worst,
                note='Algebraic enclosure checked; directed-rounding final witness verification remains separate.')
    (DATA/'bernstein_verification.json').write_text(json.dumps(record,indent=2))
    print(record)


if __name__=='__main__':main()
