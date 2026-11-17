from common import *
import json,random
import numpy as np
from graph_character import evaluate,bb84_polynomials
from graph_families import graph_error_arrays
from bernstein import evaluate as polyvalue


def main():
    entries=json.loads((DATA/'graphs.json').read_text())['entries'];rng=random.Random(9920)
    chosen=[]
    for m in range(5,9):chosen+=rng.sample([e for e in entries if e['m']==m],12)
    generator=np.random.default_rng(9920);maximum=0
    for e in chosen:
        m=e['m'];v=e['logical_dual_vector'];pivot=v.bit_length()-1;mask=(1<<m)-1
        ids=np.arange(4**m,dtype=np.int64);x=ids&mask;z=ids>>m
        # Independent forward application of optional H, all CZ edges, H.
        xx=x.copy();zz=z.copy()
        if e['input_rotations']:
            for j,rot in enumerate(e['input_rotations']):
                if rot==[0,3,2,1]:
                    change=((xx^zz)>>j)&1;xx^=change<<j;zz^=change<<j
        for a,b in e['edges']:zz^=((xx>>a)&1)<<b;zz^=((xx>>b)&1)<<a
        bit=zz;phase=np.array([(int(u)&v).bit_count()%2 for u in xx])
        key=(bit>>pivot)&1;accept=(bit^(key*v))==0
        label=np.where(phase,3-key,key)
        p=generator.dirichlet([2,1,1,1]);weights=np.ones(len(x))
        for j in range(m):weights*=p[np.where((z>>j)&1,3-((x>>j)&1),(x>>j)&1)]
        raw=np.bincount(label[accept],weights=weights[accept],minlength=4)
        new=evaluate(e,p);err=float(abs(raw-new).max());maximum=max(maximum,err);assert err<2e-13
    for m in range(5,15):
        e=rng.choice([e for e in entries if e['m']==m]);exact=bb84_polynomials(e,.18,.25,exact=True)
        for t in (0,.13,.5,.87,1):
            p=np.array([.57+.18*t,.18-.18*t,.18*t,.25-.18*t])
            a=evaluate(e,p);b=polyvalue(np.array(exact,dtype=float),t)
            assert abs(a-b).max()<3e-13
    result=dict(status='passed',independent_Pauli_propagation_examples=len(chosen),exact_BB84_polynomial_examples=10,max_discrepancy=maximum)
    (DATA/'character_verification.json').write_text(json.dumps(result,indent=2));print(result)


if __name__=='__main__':main()
