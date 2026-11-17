"""Compare known-source incumbent screening with an exhaustive AD oracle."""
from core import *
from known_search import optimize
from compound_terminals import values,actions

rng=np.random.default_rng(3026);checks=[]
for count,nmax in ((20,16),(5,128)):
    p=rng.dirichlet([8,1,1,1],count);y=rng.uniform(.05,1,count);M=np.arange(count)%4+1
    records=[dict(test_index=i) for i in range(count)];best=np.zeros(count);pp=p.copy();yy=y.copy();menu=actions(nmax)
    for B in range(4):
        best=np.maximum(best,yy*np.maximum(0,values(pp,menu).max(-1)));pp,s=b_step(pp);yy*=s/2
    result=optimize(p,y,M,records,load_codes(),K=3,long_max=nmax,with_watanabe=True,
                    seeds=[dict(index=0,AD=dict(kind='Watanabe_2007',n=2,mode='no_otp_hashed',classical_B_rounds=2))])
    for budget in range(1,5):
        exact=float(best[M<=budget].max());obtained=result['results'][f'M{budget}']['rate']
        assert abs(exact-obtained)<1e-12,(nmax,budget,exact,obtained)
        checks.append(dict(nmax=nmax,budget=budget,difference=abs(exact-obtained)))
out=dict(status='passed',checks=checks,scope='Exhaustive floating AD oracle, all three signed net modes and Watanabe, every B<=3; diagnostic for global-incumbent SPC pruning and replayed seeds. Outward certificates remain independent.')
atomic_json(DATA/'registered_screening_checks.json',out);print(json.dumps(out),flush=True)
