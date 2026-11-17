from spc_interval_dynamic import *
from decimal import Decimal as D,localcontext
from spc_dynamic import all_rates
from certify_werner_witnesses import spc_rate
from interval_certificate import IV


def main():
    scalar_checks=0
    with localcontext() as context:
        context.prec=200
        for x in np.r_[np.linspace(-1000,0,101),[-1100.,-2048.,-1e-15]]:
            ref=(D.from_float(float(x))*D(2).ln()).exp();lo,hi=exp2_point(np.array([x]))
            assert D.from_float(float(lo[0]))<=ref<=D.from_float(float(hi[0]));scalar_checks+=1
        for x in np.r_[np.linspace(0,500,101),[1e-10,1e-5,1024.]]:
            context.prec=400
            L=D.from_float(float(x));t=(-L*D(2).ln()).exp()
            ref=(-L*D(2).ln()/2).exp()*((1+t)*(1+t).ln()/t/D(2).ln()+L)/2
            lo,hi=ambiguity_point(np.array([x]));assert D.from_float(float(lo[0]))<=ref<=D.from_float(float(hi[0])),(x,lo,hi,ref);scalar_checks+=1
    rational=[[F(v,100) for v in row] for row in ((79,3,7,11),(52,24,1,23),(5,19,71,5))]
    p=np.array([list(map(float,row)) for row in rational]+[[1,0,0,0],[.25,.25,.25,.25],[.5,.5,0,0]])
    pbound=[]
    for row in rational:pbound.append([iv.rational(v) for v in row])
    for row in p[3:]:pbound.append([iv.rational(F(float(v))) for v in row])
    bounds=tuple(np.array([[v[i] for v in row] for row in pbound]) for i in (0,1));ns=np.arange(8,129)
    lo,hi=all_bounds(bounds,ns,np.ones((len(p),len(ns)),bool));floating=all_rates(p,ns)
    assert np.all(lo<=floating+2e-14) and np.all(hi>=floating-2e-14),(lo,hi,floating)
    exact_checks=0
    for i,row in enumerate(rational):
        pp=[IV(v) for v in row]
        for n in (8,13):
            jj=int(np.flatnonzero(ns==n)[0])
            ref=spc_rate(pp,int(n));assert lo[i,jj]<=float(ref.lo) and hi[i,jj]>=float(ref.hi),(row,n,lo[i,jj],hi[i,jj],ref.lo);exact_checks+=1
    result=dict(status='passed',scalar_decimal_comparisons=scalar_checks,complete_type_decimal_comparisons=exact_checks,
                floating_dynamic_comparisons=int(floating.size),max_interval_width=float((hi-lo).max()),
                scope='Certified Taylor exponentials, atanh log1p ratios and decreasing ambiguity; independent 60-digit type enumeration for n=8,13; floating O(N^3) cross-check up to 128, including zeros.')
    atomic_json(DATA/'spc_interval_dynamic_checks.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
