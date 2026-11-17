"""Direct, small-code reference for Du et al. Eqs. (29), (30), and (40).

This is an independent floating-point formula check, not an optimizer,
outward interval certificate, or unknown-parameter adaptive policy.
Bell order: I,X,Y,Z. Binary generator matrices are represented by rows.
"""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'joint_optimization'))
from algebra import rref, span, parity
from rates import CodeKernel


def xlogx(a):
    a = np.asarray(a)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(a > 0, a*np.log2(a), 0.)


def reference(p, n, rows):
    p = np.asarray(p, dtype=float)
    assert np.all(p >= 0) and abs(p.sum()-1) < 1e-12
    rows = rref(rows, n)
    k = len(rows)
    dual = rref([a for a in range(1 << n) if all(parity(a & g) == 0 for g in rows)], n)
    sx = np.array([sum(parity(a & h) << j for j,h in enumerate(dual)) for a in range(1 << n)])
    gz = np.array([sum(parity(a & g) << j for j,g in enumerate(rows)) for a in range(1 << n)])
    # joint[x,z] computed by direct multiplication, independent of the
    # compressed polynomial enumerator used by the production code.
    x = np.arange(1 << n)[:, None]
    z = np.arange(1 << n)[None, :]
    joint = np.ones((1 << n, 1 << n))
    labels = np.array([[0, 3], [1, 2]])
    for j in range(n):
        joint *= p[labels[(x >> j) & 1, (z >> j) & 1]]
    probs, full_mass, logical_mass = [], [], []
    for s in range(1 << (n-k)):
        block = joint[sx == s]
        ps = float(block.sum())
        probs.append(ps)
        full_mass.append(float(-xlogx(block).sum()+xlogx(ps)))
        logical = np.stack([block[:, gz == g].sum(axis=1) for g in range(1 << k)], axis=1)
        logical_mass.append(float(-xlogx(logical).sum()+xlogx(ps)))
    probs, full_mass, logical_mass = map(np.asarray, (probs, full_mass, logical_mass))
    assert abs(probs.sum()-1) < 1e-12
    gross = np.maximum(n*probs-full_mass, 0).sum()/n
    plain_cost = (n-k)/n
    compressed_cost = -float(xlogx(probs).sum())/n
    no_otp = np.maximum(k*probs-logical_mass, 0).sum()/n
    return dict(no_otp=float(no_otp), otp_uncompressed_net=float(max(0,gross-plain_cost)),
                otp_compressed_net=float(max(0,gross-compressed_cost)),
                otp_gross=float(gross), otp_plain_cost=plain_cost,
                otp_compressed_cost=compressed_cost,
                signed_all_flags_compressed=float((n-full_mass.sum())/n-compressed_cost))


def main():
    out = ROOT / 'data'
    out.mkdir(exist_ok=True)
    library=json.loads((ROOT.parent/'resource_extension'/'data'/'codes.json').read_text())['entries']
    small = [e for e in library if e['n'] <= 5]
    states = [[1-1.5*q,q/2,q/2,q/2] for q in (.01,.10,.15,.20,.25,.30)]
    states += [[1.,0,0,0],[.5,0,0,.5],[.55,.02,.4,.03]]
    maximum_error=0.; comparisons=0
    for e in small:
        kernel=CodeKernel(e)
        for p in states:
            r=reference(p,e['n'],e['rows'])
            error=abs(r['no_otp']-float(kernel.evaluate(p)))
            maximum_error=max(maximum_error,error)
            assert error < 3e-12,(e,p,error)
            assert r['no_otp']+3e-12 >= r['otp_uncompressed_net'],(e,p,r)
            assert r['otp_compressed_net']+3e-12 >= r['otp_uncompressed_net']
            assert abs(r['signed_all_flags_compressed']-(1+float(xlogx(p).sum()))) < 3e-12
            comparisons+=1
    examples=[]
    for q in (.05,.10,.12,.15,.20,.25):
        p=[1-1.5*q,q/2,q/2,q/2]
        examples.append(dict(Q=q,code='[2,1,2]',**reference(p,2,[3])))
    result=dict(status='passed',code_classes=len(small),state_code_checks=comparisons,
                maximum_no_OTP_discrepancy=maximum_error,examples=examples,
                checks=['Direct entropy sums agree with current no-OTP kernel.',
                        'No OTP dominates uncompressed OTP at the checked inputs.',
                        'Compression does not decrease the OTP rate.',
                        'Without branch rejection, compressed OTP reduces to 1-H(p).'],
                scope='Reference arithmetic only. OTP has not yet been integrated into the joint ED search or certified BB84 fixed-policy optimization. Checks are numerical, not general proofs.')
    (out/'otp_reference_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
