from common import *
import json
import numpy as np
from algebra import span
from rates import xlog2x
from kernels import load
from spc_long import spc_rate
from search_six_state import spc_upper


def independent_rate(p,entry):
    n=entry['n'];k=entry['k'];words=span(entry['rows']);xs=np.arange(1<<n)
    pop=np.array([i.bit_count() for i in range(1<<n)])
    a=p[0]+p[3];b=p[1]+p[2];c0=(p[0]-p[3])/a;c1=(p[1]-p[2])/b
    overlap=pop[xs[:,None]&np.array(words)[None,:]]
    chars=c0**(pop[words][None,:]-overlap)*c1**overlap
    for bit in range(k):
        step=1<<bit;v=chars.reshape(len(xs),-1,2*step);left=v[:,:,:step].copy();right=v[:,:,step:].copy()
        v[:,:,:step]=left+right;v[:,:,step:]=left-right
    joint=chars/(1<<k)*(a**(n-pop)*b**pop)[:,None]
    assert joint.min()>-1e-13;joint=np.maximum(joint,0.)
    unused=set(map(int,xs));total=0
    while unused:
        x=min(unused);group=[x^w for w in words];unused.difference_update(group)
        block=joint[group];ps=block.sum()
        total+=max(0.,k*ps+float(xlog2x(block).sum())-float(xlog2x(ps)))
    return total/n


def main():
    rng=np.random.default_rng(92021);ks=load();maximum=0;checks=0
    for kernel in ks:
        if kernel.n<7:continue
        for _ in range(3):
            p=rng.dirichlet([5,1,1,1]);a=independent_rate(p,kernel.entry);b=float(kernel.evaluate(p))
            err=abs(a-b);maximum=max(maximum,err);assert err<5e-13;checks+=1
    for _ in range(12):
        p=rng.dirichlet([6,1,1,1])
        for n in (7,16,32,64,128):assert spc_rate(p,n)<=spc_upper(p,n)+2e-12
    result=dict(status='passed',independent_Fourier_rate_checks=checks,SPC_upper_bound_checks=60,max_rate_discrepancy=maximum)
    (DATA/'rate_verification.json').write_text(json.dumps(result,indent=2));print(result)


if __name__=='__main__':main()
