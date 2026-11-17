"""Independent dense three-input instrument and exact retry-cost audit."""
from itertools import product
from fractions import Fraction as F
import hashlib,json
import numpy as np
from full_clifford_model import HERE,bootstrap

def main():
    job=HERE/'diagnostics/nonreal_witness';core,engine,model,ops=bootstrap(job)
    row=ops[9+33];assert row['name']=='D33' and not row['real_implementable']
    I=np.eye(2,dtype=complex);X=np.array([[0,1],[1,0]],complex);Z=np.diag([1,-1]);Y=np.array([[0,-1j],[1j,0]])
    paulis=[I,X,Y,Z];mats={'h':(X+Z)/np.sqrt(2),'s':np.diag([1,1j])}
    def kron3(ms):return np.kron(np.kron(ms[0],ms[1]),ms[2])
    def decoder(wires):
        U=np.eye(8,dtype=complex)
        for gate in row['gates']:
            if gate[0] in mats:
                ms=[I,I,I];ms[wires[gate[1]]]=mats[gate[0]];G=kron3(ms)
            else:
                G=np.zeros((8,8),complex);a,b=[wires[j] for j in gate[1:]]
                for k in range(8):
                    bits=[(k>>j)&1 for j in (2,1,0)];bits[b]^=bits[a];G[4*bits[0]+2*bits[1]+bits[2],k]=1
            U=G@U
        return U
    U1=decoder((0,1));U2=decoder((0,2));U=U2@U1
    labels=list(product(range(4),repeat=3));P=np.array([kron3([paulis[j] for j in labs]) for labs in labels])
    accepted=[];first_accepted=[];maxerror=0.
    for labs,E in zip(labels,P):
        e1=U1@E@U1.conj().T;v1=np.einsum('aij,ji->a',P.conj(),e1.T)/8
        j1=int(np.argmax(np.abs(v1)));l1=labels[j1]
        et=U@E@U.conj().T;vt=np.array([np.trace(A.conj().T@et)/8 for A in P])
        jt=int(np.argmax(np.abs(vt)));lt=labels[jt]
        err=float(np.max(np.abs(et-vt[jt]*P[jt])));maxerror=max(maxerror,err)
        assert abs(abs(vt[jt])-1)<2e-13 and err<2e-13
        a1=l1[1] in (0,3);at=lt[1] in (0,3) and lt[2] in (0,3)
        # Explicit sequential accepted map, independent of the dense product.
        k1=row['mapping'][4*labs[0]+labs[1]]
        k2=-1 if k1<0 else row['mapping'][4*k1+labs[2]]
        assert a1==(k1>=0) and at==(k2>=0)
        if at:assert lt[0]==k2
        first_accepted.append(a1);accepted.append(-1 if not at else lt[0])
    tests=[]
    for z in (F(0),F(17,1000),F(17,200),F(17,100)):
        q=F(17,100);p=[1-2*q+z,q-z,z,q-z];out=[F(0)]*4;s1=F(0)
        for labs,a1,k in zip(labels,first_accepted,accepted):
            prob=p[labs[0]]*p[labs[1]]*p[labs[2]]
            if a1:s1+=prob
            if k>=0:out[k]+=prob
        success=sum(out);yield_exact=success/(2+s1)
        tree=model.Node('D33',model.Node('D33',model.RAW,model.RAW),model.RAW)
        result=model.evaluate(tree,np.array([float(x) for x in p]),'Y')
        expected=np.array([float(x/success) for x in out])[list(model.AXIS_PERMUTATIONS['Y'])]
        assert np.max(np.abs(expected-result['state']))<2e-14
        assert abs(float(yield_exact)-result['yield_per_raw_pair'])<2e-14
        assert model.terminal_memory(tree,'Y')==2
        tests.append(dict(z=str(z),success_three_input=str(success),success_first_check=str(s1),
            yield_per_consumed_raw_pair=str(yield_exact),normalized_terminal_state=expected.tolist()))
    strategy=[dict(AD=dict(kind='Watanabe_2007',n=2,mode='no_otp_hashed'),allocation=1.,allocation_exact='1')]
    ed=model.structural_record(tree,'Y','enlarged_axes')
    certificate=engine.certify_policy(.17,.17,ed,0,strategy,atol=2e-11,max_cells=16000)
    assert certificate['converged'] and certificate['lower']>.03092
    result=dict(status='passed',dense_input_Paulis=64,dense_maximum_error=maxerror,
        independent_cost='C=(2+s_first)/p_three_input; rejected partial attempts are included.',
        exact_source_checks=tests,ED=ed,executable_schedule=model.executable_schedule(tree,'Y'),
        certificate=certificate,strategy=strategy,source_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest())
    (job/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status='passed',certificate=certificate),indent=2))
if __name__=='__main__':main()
