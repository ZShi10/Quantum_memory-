"""Independent diagnostics for the accompanying analytical pumping theorems.

The analytical closure proof, not these finite/random checks, covers arbitrary
pumping length. All arithmetic in normal-form and YY identities is exact.
A selected finite positive-rate example is enclosed with interval arithmetic.
Dependencies: numpy, mpmath. Run: python code/pumping_checks.py
"""
from __future__ import annotations
from collections import deque
from fractions import Fraction as F
from itertools import permutations
from pathlib import Path
import json
import numpy as np
from mpmath import mp, iv

ROOT = Path(__file__).resolve().parent
BITS = [(0,0),(1,0),(1,1),(0,1)]
INDEX = {v:i for i,v in enumerate(BITS)}
PERMS = list(permutations(range(4)))


def transform(v, gate):
    x0,x1,z0,z1=v
    if gate=='H0': x0,z0=z0,x0
    elif gate=='H1': x1,z1=z1,x1
    elif gate=='S0': z0^=x0
    elif gate=='S1': z1^=x1
    elif gate=='C01': x1^=x0; z0^=z1
    elif gate=='C10': x0^=x1; z1^=z0
    else: raise ValueError(gate)
    return (x0,x1,z0,z1)


def clifford_group():
    # Matrices are represented by their columns, i.e. images of Pauli generators.
    ident=tuple(tuple(int(i==j) for i in range(4)) for j in range(4))
    seen={ident}; queue=deque([ident])
    while queue:
        m=queue.popleft()
        for g in ['H0','H1','S0','S1','C01','C10']:
            nxt=tuple(transform(c,g) for c in m)
            if nxt not in seen: seen.add(nxt); queue.append(nxt)
    assert len(seen)==720
    return sorted(seen)


def error_map(matrix, flag):
    result=[]
    for i in range(4):
        for j in range(4):
            v=(BITS[i][0],BITS[j][0],BITS[i][1],BITS[j][1])
            w=tuple(sum(v[k]*matrix[k][a] for k in range(4))%2 for a in range(4))
            result.append(INDEX[(w[0],w[2])] if w[1]==flag else -1)
    return tuple(result)


def canonical_map():
    out=[]
    for x,z in BITS:
        for y,w in BITS:
            out.append(INDEX[(x,z^w)] if x==y else -1)
    return tuple(out)


def normal_form_check():
    can=canonical_map(); equivalents=set()
    for a in PERMS:
        for b in PERMS:
            v=[can[4*a[i]+b[j]] for i in range(4) for j in range(4)]
            for c in PERMS:
                equivalents.add(tuple(c[k] if k>=0 else -1 for k in v))
    matrices=clifford_group(); tested=0; trivial=0
    for m in matrices:
        check=[m[i][1] for i in range(4)]
        wt=int(bool(check[0] or check[2]))+int(bool(check[1] or check[3]))
        for f in (0,1):
            mapping=error_map(m,f)
            if wt==2:
                assert mapping in equivalents
                tested+=1
            else:
                assert wt==1
                # For a weight-one check, the unmeasured input is passed through
                # up to a Pauli permutation, independently of the discarded error.
                measured=0 if check[0] or check[2] else 1
                grids=np.array(mapping).reshape(4,4)
                if measured==1: grids=grids.T
                accepted=[tuple(row) for row in grids if row[0]>=0]
                assert len(accepted)==2 and len(set(accepted))==1
                assert sorted(accepted[0])==list(range(4))
                trivial+=1
    return {'symplectic_matrices':720,'weight_two_flag_maps':tested,
            'weight_one_flag_maps':trivial,'normal_form_equivalents':len(equivalents),
            'status':'passed; exact integer map comparison'}


def se_defect(p):
    p=np.asarray(p)
    return np.sum(p*p,axis=-1)-4*np.sqrt(np.maximum(0,np.prod(p,axis=-1)))-0.5


def blocker(p):
    return all((p[i]-p[j])**2 <= (p[i]+p[j])*(1-p[i]-p[j])+1e-14
               for i in range(4) for j in range(i+1,4))


def z_step(q,p):
    I,X,Y,Z=q; i,x,y,z=p
    v=np.array([I*i+Z*z,X*x+Y*y,X*y+Y*x,I*z+Z*i])
    return v/v.sum()


def random_invariant_check():
    rng=np.random.default_rng(2601008)
    survivors=[]; raws=[]
    while len(survivors)<2000 or len(raws)<2000:
        p=rng.dirichlet([.8]*4)
        if len(survivors)<2000 and se_defect(p)<=0: survivors.append(p)
        if len(raws)<2000 and blocker(p): raws.append(p)
    largest=-1.; cases=0
    for p,q in zip(raws,survivors):
        for perm in PERMS:
            pp=p[list(perm)]; qq=q[rng.permutation(4)]
            out=z_step(qq,pp)
            d=float(se_defect(out)); largest=max(largest,d); cases+=1
            assert d<=2e-12
    # Test the primitive analytical inequality also directly at its boundary.
    residual=1.
    for _ in range(10000):
        k=rng.random(); X=rng.random(); u2=k*rng.random()
        lhs=np.sqrt(1-u2+u2*X*X);rhs=1-k+k*X
        residual=min(residual,float(lhs-rhs));assert lhs+2e-15>=rhs
    return {'random_bell_map_tests':cases,'largest_SE_defect':largest,
            'primitive_inequality_tests':10000,'smallest_primitive_slack':residual,
            'status':'passed; floating diagnostics only, not proof'}


def gate_step(p,q,word):
    v=[F(0)]*4
    for i,(x,z) in enumerate(BITS):
        for j,(y,w) in enumerate(BITS):
            e=(x,y,z,w)
            for g in word: e=transform(e,g)
            if e[1]==0: v[INDEX[(e[0],e[2])]] +=p[i]*q[j]
    s=sum(v)
    return [a/s for a in v],s


def fixed_axis_pump(p,r):
    I,X,Y,Z=p;a=I+Z;b=X+Y;d=I-Z;e=X-Y
    den=a**r+b**r
    out=[(a**r+d**r)/(2*den),(b**r+e**r)/(2*den),
         (b**r-e**r)/(2*den),(a**r-d**r)/(2*den)]
    cost=(1+sum(a**j+b**j for j in range(1,r)))/den
    return out,cost


def exact_yy_check():
    rng=np.random.default_rng(52);nchecks=0
    for _ in range(40):
        ints=rng.integers(1,50,size=4);tot=sum(map(int,ints))
        p=[F(int(v),tot) for v in ints];q=p[:];cost=F(1)
        py=[p[i] for i in [0,1,3,2]]
        for r in range(2,13):
            q,s=gate_step(q,p,['C01','H0','C01','C10']);cost=(cost+1)/s
            b,c=fixed_axis_pump(py,r)
            lhs=[q[i] for i in [0,1,3,2]]
            rhs=b if r%2 else [b[i] for i in [0,2,1,3]]
            assert lhs==rhs and cost==c;nchecks+=1
    return {'exact_rational_state_and_cost_checks':nchecks,
            'status':'passed; YY pumping identity for nonsymmetric general Bell laws'}


def as_iv(f):return iv.mpf(f.numerator)/iv.mpf(f.denominator)

def interval_example():
    iv.dps=80;mp.dps=100
    Q=F(7,25);p=[1-3*Q/2,Q/2,Q/2,Q/2]
    inner,c1=fixed_axis_pump(p,3);inner=[inner[i] for i in [0,3,2,1]]
    outer,c2=fixed_axis_pump(inner,11);cost=c1*c2
    h=-sum(as_iv(a)*iv.ln(as_iv(a))/iv.ln(2) for a in outer if a)
    rate=(1-h)/as_iv(cost)
    # Conservative decimal claims, checked as interval inequalities.
    lo=iv.mpf('0.0000043419');hi=iv.mpf('0.0000043421')
    assert (rate>lo) is True and (rate<hi) is True
    def mpv(f): return mp.mpf(f.numerator)/f.denominator
    return {'Q':'7/25','inner_Z_pump_ancestry':3,'Hadamard_between_levels':True,
            'outer_Z_pump_ancestry':11,'live_memory':3,
            'success_ancestry_raw_leaves':33,
            'mean_raw_cost':mp.nstr(mpv(cost),60),
            'output_Bell_probabilities':[mp.nstr(mpv(a),60) for a in outer],
            'entropy_interval':str(h),'rate_interval':str(rate),
            'conservative_rate_bounds':['0.0000043419','0.0000043421'],
            'status':'exact rational probabilities/cost; mpmath interval logarithms at 80 dps'}


def main():
    results={'scope':'Diagnostics supplement the analytical proof; not a finite-class capacity search.',
             'normal_form':normal_form_check(),
             'SE_invariance':random_invariant_check(),
             'YY_pumping':exact_yy_check(),
             'positive_three_slot_example':interval_example()}
    (ROOT.parent/'validation/pumping_checks_results.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
