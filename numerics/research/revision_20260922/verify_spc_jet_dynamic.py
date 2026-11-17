from compound_terminals import *
from spc_jet_dynamic import spc_jet
from certify_werner_witnesses import spc_rate
from interval_certificate import IV


def main():
    checks=0;derivative_checks=0;mean_checks=0
    for row in ((79,3,7,11),(85,6,4,5),(96,1,2,1)):
        rational=[F(v,100) for v in row];direction=[F(v,10) for v in (1,-1,1,-1)]
        for n in (8,13):
            points=[]
            for t in (F(-1,1000),F(0),F(1,1000)):
                p=[a+t*d for a,d in zip(rational,direction)];bounds=[iv.rational(a) for a in p];deriv=[iv.rational(d) for d in direction]
                jet=j.Jet(tuple(np.array([[v[i] for v in bounds]]) for i in (0,1)),tuple(np.array([[v[i] for v in deriv]]) for i in (0,1)))
                got=spc_jet(jet,n);ref=spc_rate([IV(a) for a in p],n)
                assert got.v[0][0]<=float(ref.lo) and got.v[1][0]>=float(ref.hi),(row,n,t,got.v,ref.lo);checks+=1
                delta=F(1,10**15)
                rplus=spc_rate([IV(a+delta*d) for a,d in zip(p,direction)],n)
                rminus=spc_rate([IV(a-delta*d) for a,d in zip(p,direction)],n)
                derivative=(rplus-rminus)/IV(2*delta)
                assert got.d[0][0]-1e-19<=float(derivative.lo) and got.d[1][0]+1e-19>=float(derivative.hi),(row,n,t,got.d,derivative.lo)
                derivative_checks+=1;points.append(float(ref.lo))
            low=[a-abs(d)*F(1,1000) for a,d in zip(rational,direction)];high=[a+abs(d)*F(1,1000) for a,d in zip(rational,direction)]
            bounds=(np.array([[iv.rational(x)[0] for x in low]]),np.array([[iv.rational(x)[1] for x in high]]))
            jet=j.Jet(bounds,tuple(np.array([[iv.rational(d)[i] for d in direction]]) for i in (0,1)));box=spc_jet(jet,n)
            radius=max(abs(box.d[0][0]),abs(box.d[1][0]))
            lower=float(points[1])-radius*.001-1e-13
            assert lower<=min(points);mean_checks+=1
    out=dict(status='passed',type_reference_comparisons=checks,decimal_derivative_diagnostics=derivative_checks,
             mean_value_checks=mean_checks,scope='O(N^3) interval derivative versus independently enumerated 60-digit complete error types; finite-difference derivatives are diagnostics, not a replacement for the analytic derivative bounds.')
    atomic_json(DATA/'spc_jet_dynamic_checks.json',out);print(json.dumps(out),flush=True)


if __name__=='__main__':main()
