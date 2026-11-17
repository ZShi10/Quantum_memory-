"""Recompute the Pauli combined-protocol formulas of arXiv:2410.22295v2.

This is an entanglement-yield reference, NOT a certified finite-memory QKD
implementation. The output includes that limitation explicitly.
"""
from core import *
from recursive_library import PRE,greedy
from literature_rates import vollbrecht_verstraete
from itertools import product,permutations
import argparse,time

LABELS=((0,0),(1,0),(1,1),(0,1))
ERRORS=np.array(list(product(range(4),repeat=4)))
TENSOR=np.zeros((16,256));ODD=np.zeros(256);REJECT=np.zeros(256)
for j,error in enumerate(ERRORS):
    x,z=np.array([LABELS[i] for i in error]).T
    if sum(x)%2:ODD[j]=1
    elif sum(z)%2:REJECT[j]=1
    else:
        a=LABELS.index((int(x[0]^x[2]),int(z[0]^z[3])))
        b=LABELS.index((int(x[1]^x[2]),int(z[1]^z[3])))
        TENSOR[4*a+b,j]=1


def vv_correlated(p):
    """Eq.(14), normalized per input Bell pair, including breeding cost."""
    p=np.asarray(p).reshape(4,4)
    even=np.array([[LABELS[i][0]==LABELS[j][0] for j in range(4)] for i in range(4)])
    mass=float(p[even].sum())
    h_even=float(entropy(p[even]/mass)) if mass else 0.
    # The first pair is measured in Z on odd parity; its phase is forgotten.
    g0=np.array([p[0,1]+p[3,1],p[0,2]+p[3,2]])
    g1=np.array([p[1,0]+p[2,0],p[1,3]+p[2,3]])
    value=mass*(1-h_even/2)-float(binary_entropy(mass))/2
    for g in (g0,g1):
        if g.sum():value+=float(g.sum())/2*(1-float(entropy(g/g.sum())))
    return max(0.,value)


def vv_correlated_best(p):
    p=np.asarray(p).reshape(4,4)
    return max(vv_correlated(p[np.ix_(PRE[a],PRE[b])]) for a in PRE for b in PRE)


def aepp4(p):
    probabilities=np.prod(np.asarray(p)[ERRORS],axis=-1)
    raw=TENSOR@probabilities;mass=float(raw.sum());odd=float(ODD@probabilities)
    accepted,_=b_step(np.asarray(p))
    return (raw/mass if mass else np.ones(16)/16),mass,accepted,odd


def best_greedy_vv(p,depth):
    best=(0.,0,'Z')
    for row in greedy(np.asarray(p),depth):
        for axis in PRE:
            value=row['yield_per_raw_pair']*float(vollbrecht_verstraete(np.array(row['p'])[list(PRE[axis])]))
            if value>best[0]:best=(value,row['depth'],axis)
    return best


def combined(p,depth=16):
    direct=best_greedy_vv(p,depth);best=direct[0];chosen=dict(branch='direct_VV',N1=direct[1],axis=direct[2])
    for row in greedy(np.asarray(p),depth):
        # Optimize a common bilateral frame before AEPP4. This can only
        # strengthen the reference; it is explicitly a formula recomputation.
        for perm in permutations((1,2,3)):
            frame=(0,)+perm;state=np.array(row['p'])[list(frame)]
            correlated,mass,failed,odd=aepp4(state)
            suffix=best_greedy_vv(failed,depth)
            value=row['yield_per_raw_pair']*(mass/2*vv_correlated_best(correlated)+odd/4*suffix[0])
            if value>best:
                best=value;chosen=dict(branch='AEPP4_VV',N1=row['depth'],N2=suffix[1],
                                       N3=suffix[1],frame=list(frame),odd_final_axis=suffix[2])
    return dict(rate=best,choice=chosen,greedy_VV_rate=direct[0])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--q',nargs='+',type=float);parser.add_argument('--dense',action='store_true');args=parser.parse_args()
    rng=np.random.default_rng(619)
    for p in rng.dirichlet(np.ones(4),20):
        assert abs(vv_correlated(np.outer(p,p))-float(vollbrecht_verstraete(p)))<1e-13
        raw=TENSOR@np.prod(p[ERRORS],axis=-1)
        assert np.allclose(raw.reshape(4,4),raw.reshape(4,4).T,atol=1e-14)
        assert abs(raw.sum()+ODD@np.prod(p[ERRORS],axis=-1)+REJECT@np.prod(p[ERRORS],axis=-1)-1)<1e-13
    path=DATA/'adaptive_combined_reference.json'
    output=json.loads(path.read_text()) if path.exists() else dict(results=[])
    done={r['Q'] for r in output['results']}
    for q in np.linspace(0,1/3,500) if args.dense else args.q or [.1,.2,.25,.28,.3,.32]:
        if float(q) in done:continue
        start=time.time();result=combined(np.array([1-1.5*q,q/2,q/2,q/2]));result['Q']=float(q)
        output['results'].append(result)
        output['scope']='Formula recomputation, not author source data: Greedy N1,N2,N3<=16, AEPP*(4) Eq.(27) via 256 Pauli events, correlated VV Eq.(14); bilateral frames optimized. Entanglement yield with asymptotic breeding/hashing; finite-memory QKD conversion NOT yet certified.'
        atomic_json(path,output);print('Combined reference',q,result,'seconds',round(time.time()-start,1),flush=True)


if __name__=='__main__':main()
