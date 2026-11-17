"""Primary-source formulas for strong classical terminal baselines.

Watanabe et al. arXiv:0705.2904v2 Theorem 3 Eq.(4), Bell order I,X,Y,Z.
VV: arXiv:2410.22295v2 Eq.(15), independently of its AEPP(4) extension.
"""
from core import *


def watanabe_components(p):
    p=np.asarray(p);a,b,c0,c1=bell_parameters(p)
    u=p[...,0]*p[...,1]+p[...,3]*p[...,2]
    v=a*b
    conditional=np.divide(u,v,out=np.zeros_like(v),where=v>0)
    first=1-entropy(p)+v*binary_entropy(conditional)
    out,success=b_step(p)
    second=success/2*(1-entropy(out))
    return np.stack((first,second),-1)


def watanabe(p):
    return np.maximum(0,watanabe_components(p).max(-1))


def vollbrecht_verstraete(p):
    p=np.asarray(p);a,b,c0,c1=bell_parameters(p)
    return np.maximum(0,1-entropy(p)+a*b/2*(binary_entropy((1-c0)/2)+binary_entropy((1-c1)/2)))


def quantum_entropy(matrix):
    eigen=np.linalg.eigvalsh(matrix)
    assert eigen.min()>-3e-13
    return float(-xlog2x(np.maximum(eigen,0)).sum())


def watanabe_cq_reference(p):
    """Construct Eve's conditional 16-dimensional states from purification.

    Direct density-matrix entropy check, independent of the closed Eq.(4).
    """
    labels=((0,0),(1,0),(1,1),(0,1))
    blocks={}
    for a in range(4):
        aa=(a&1,a>>1)
        for b in range(4):
            bb=(b&1,b>>1)
            errors=(aa[0]^bb[0],aa[1]^bb[1])
            parity=errors[0]^errors[1]
            u1=aa[0]^aa[1];u2=aa[1] if parity==0 else 0
            vector=np.zeros(16)
            for i,(x,z) in enumerate(labels):
                for j,(xx,zz) in enumerate(labels):
                    if (x,xx)==errors:
                        vector[4*i+j]=np.sqrt(p[i]*p[j])/2*(-1)**(aa[0]*z+aa[1]*zz)
            key=(parity,u1,u2)
            blocks[key]=blocks.get(key,np.zeros((16,16)))+np.outer(vector,vector)
    def merged_entropy(keep):
        merged={}
        for key,mat in blocks.items():
            target=tuple(key[i] for i in keep)
            merged[target]=merged.get(target,np.zeros((16,16)))+mat
        return sum(quantum_entropy(mat) for mat in merged.values())
    joint=merged_entropy((0,1,2));eve_w=merged_entropy((0,));eve_wu=merged_entropy((0,1))
    bit=p[1]+p[2];s=(1-bit)**2+bit**2
    leakage_kept=s*float(binary_entropy(bit**2/s))
    return np.array([(joint-eve_w-float(binary_entropy(1-s))-leakage_kept)/2,
                     (joint-eve_wu-leakage_kept)/2])


if __name__=='__main__':
    rng=np.random.default_rng(702903)
    states=np.r_[rng.dirichlet(np.ones(4),size=24),np.eye(4)]
    errors=[]
    for p in states:
        errors.append(float(np.max(np.abs(watanabe_components(p)-watanabe_cq_reference(p)))))
    assert max(errors)<3e-12,errors
    assert np.all(watanabe(states)+3e-12>=vollbrecht_verstraete(states))
    result=dict(status='passed',states=len(states),maximum_cq_entropy_error=max(errors),
                source='Watanabe et al., arXiv:0705.2904v2, Theorem 3 Eq.(4)',
                scope='Independent purification/density-matrix check of both signed expressions. VV check covers Eq.(15), not the whole AEPP combined protocol.')
    atomic_json(DATA/'watanabe_reference_checks.json',result)
    print(json.dumps(result),flush=True)
