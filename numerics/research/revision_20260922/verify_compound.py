"""Independent cq-register, signed direct-sum and interval checks."""
from bb84_compound import *
from ad_modes import OTPKernel
from literature_rates import quantum_entropy
from verify_fixed_policy import at
from math import comb


def signed_rates(k,p):
    no=k.evaluate(p);g,c=OTPKernel(k.entry).components(p);gross=np.maximum(g,0).sum(-1)
    return np.stack((no,gross-(k.n-k.k)/k.n,gross-c),-1)


def cq_gross(k,p,otp):
    """Eve density matrices after an actual tag/syndrome transcript.

    No virtual phase error identity is used in this reference. The public
    ciphertext for OTP is uniform and independent; its paid key is dealt
    with separately. Alice's unencrypted tag is retained only for no OTP.
    """
    n=k.n;size=1<<n;words=span(k.entry['rows']);index={w:i for i,w in enumerate(words)}
    reps=[min(a^w for w in words) for a in range(size)]
    labels={(0,0):0,(1,0):1,(1,1):2,(0,1):3}
    law=np.array([[np.prod([p[labels[(x>>j&1,z>>j&1)]] for j in range(n)])
                   for z in range(size)] for x in range(size)])
    blocks={}
    for a in range(size):
        for x in range(size):
            v=np.zeros(size*size)
            for z in range(size):v[x+size*z]=np.sqrt(law[x,z]/size)*(-1)**((a&z).bit_count()%2)
            tag=0 if otp else reps[a];key=a if otp else index[a^reps[a]]
            label=(reps[x],tag,key)
            blocks[label]=blocks.get(label,np.zeros((size*size,size*size)))+np.outer(v,v)
    total=0.;signed=[]
    for s in sorted(set(reps)):
        selected={q:v for q,v in blocks.items() if q[0]==s};merged={}
        for (ss,tag,key),v in selected.items():merged[tag]=merged.get(tag,np.zeros_like(v))+v
        conditional=sum(quantum_entropy(v) for v in selected.values())-sum(quantum_entropy(v) for v in merged.values())
        px=law[[x for x in range(size) if reps[x]==s]].sum(-1);ps=px.sum()
        bit_entropy=float(-xlog2x(px).sum()+xlog2x(ps))
        gain=conditional-bit_entropy;signed.append(gain);total+=max(0,gain)/n
    return total,np.array(signed)/n


def main():
    rng=np.random.default_rng(25092026);small=load_codes(3,all_codes=True)
    states=np.r_[rng.dirichlet(np.ones(4),4),[[.91,.03,.02,.04],[.6,.18,.02,.2]],np.eye(4)]
    cq_error=0.;cq_checks=0
    for k in small:
        for p in states:
            no,_=cq_gross(k,p,False);otp,gains=cq_gross(k,p,True)
            target=OTPKernel(k.entry).components(p)[0]
            err=max(abs(no-k.evaluate(p)),abs(otp-np.maximum(target,0).sum()))
            cq_error=max(cq_error,float(err));cq_checks+=2
    assert cq_error<2e-12,cq_error

    states=np.r_[rng.dirichlet(np.ones(4),20),np.eye(4)]
    direct_checks=0;max_excess=-np.inf
    for a in small:
        for b in small:
            if a.n+b.n>5:continue
            n=a.n+b.n;entry=dict(n=n,k=a.k+b.k,rows=a.entry['rows']+[g<<a.n for g in b.entry['rows']],id='direct')
            k=CodeKernel(entry);left=signed_rates(k,states)
            right=(a.n*signed_rates(a,states)+b.n*signed_rates(b,states))/n
            max_excess=max(max_excess,float((left-right).max()));direct_checks+=left.size
    assert max_excess<3e-13,max_excess

    kernels=load_codes(4);entries=[e for e in complete_short_entries(False) if e['m']<=2]
    selected=[entries[0],next(e for e in entries if e['id']=='E00029'),entries[-1]]
    max_point_error=0.;tangent_checks=0;negative_nets=0;allocations=0
    for e in selected:
        b,p=.13,.19;raw=raw_polynomials(e,b,p);exact=exact_raw(e,b,p);menu=CompoundMenu(raw,e['m'],kernels)
        for trial in range(3):
            weights=np.zeros(menu.count)
            for branch in range(len(raw)):
                for mode,alpha in zip(MODES,(.2,.3,.4)):
                    opts=[o for o in menu.options if o['branch']==branch and o['mode']==mode]
                    o=opts[int(rng.integers(len(opts)))];weights[o['alpha']]=alpha
            curves,strategy=exact_strategy(menu,weights,exact,e['m'])
            for branch in range(len(raw)):
                assert sum(F(s['allocation_exact']) for s in strategy if s['branch']==branch)<=1;allocations+=1
            lower=sum(c.bound(c.joint,c.den)[0] for c in curves)
            for j in (0,1,13,37,50,63,91,99,100):
                t=F(j,100);values=menu.values([float(t)])[0];expected=float(values@weights)
                negative_nets+=int(np.sum(values<0));actual=IV(0)
                for c in curves:
                    _,val=c.bound([[at(row,t)] for row in c.joint],[at(c.den,t)]);actual=actual+val
                error=abs(expected-float(actual.lo));max_point_error=max(max_point_error,error)
                assert error<5e-13,(e['id'],t,expected,actual.lo)
                assert float(lower)<=expected+5e-13;tangent_checks+=1
    assert negative_nets>0
    out=dict(status='passed',cq_register_comparisons=cq_checks,maximum_cq_error=cq_error,
             signed_direct_sum_comparisons=direct_checks,maximum_direct_sum_excess=max_excess,
             rational_tangent_checks=tangent_checks,maximum_tangent_point_error=max_point_error,
             exact_feasibility_checks=allocations,negative_OTP_coefficients_retained=negative_nets,
             scope='Independent density matrices for Du raw registers and transcript; signed direct sums; fixed-allocation post-IR interval curves. Uniform asymptotic security is an analytic argument, not established by numerical tests.')
    atomic_json(DATA/'compound_checks.json',out);print(json.dumps(out),flush=True)


if __name__=='__main__':main()
