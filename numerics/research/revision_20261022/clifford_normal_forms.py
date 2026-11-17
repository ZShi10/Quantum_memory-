"""Exact two-input bilateral Clifford normal forms, preserving output frames.

Zero relative syndrome, one retained Bell pair. Pauli phases cancel in the
U tensor U* relative-error map; gate words are retained for executable use.
This is an exact finite coverage check, not an arbitrary-protocol converse.
"""
from pathlib import Path
from collections import deque
from itertools import permutations, product
import hashlib, json

HERE=Path(__file__).resolve().parent
AXES='XYZ';BITS=((0,0),(1,0),(1,1),(0,1))
PB=dict(zip('IXYZ',BITS));PERMS=[(0,)+p for p in permutations((1,2,3))]
GATES=(('h',0),('h',1),('s',0),('s',1),('cx',0,1),('cx',1,0))

def gate(v,g):
    x=v&3;z=v>>2
    if g[0]=='h':
        j=g[1];t=(x>>j^z>>j)&1;x^=t<<j;z^=t<<j
    elif g[0]=='s':z^=((x>>g[1])&1)<<g[1]
    elif g[0]=='cx':
        a,b=g[1:];x^=((x>>a)&1)<<b;z^=((z>>b)&1)<<a
    else:raise ValueError(g)
    return x|z<<2

def transform(v,cols):
    w=0
    for j in range(4):
        if v>>j&1:w^=cols[j]
    return w

def symp(a,b):return (((a&3)*(1))&(b>>2)).bit_count()%2 ^ (((b&3)&(a>>2)).bit_count()%2)

def pauli(v):return ''.join('IXYZ'[BITS.index(((v>>j)&1,(v>>(2+j))&1))] for j in range(2))
def bits(word):return sum(PB[c][0]<<j|PB[c][1]<<(j+2) for j,c in enumerate(word))

def bfs(real=False):
    identity=(1,2,4,8);seen={identity:()};queue=deque([identity])
    while queue:
        cols=queue.popleft()
        for g in GATES:
            if real and g[0]=='s':continue
            nxt=tuple(gate(v,g) for v in cols)
            if nxt not in seen:seen[nxt]=seen[cols]+(g,);queue.append(nxt)
    assert len(seen)==(72 if real else 720)
    return seen

def error_map(cols):
    mapping=[]
    for a,b in product(BITS,repeat=2):
        v=a[0]|b[0]<<1|a[1]<<2|b[1]<<3;w=transform(v,cols)
        mapping.append(-1 if w&2 else BITS.index((w&1,(w>>2)&1)))
    return tuple(mapping)

def pullback(v,cols):return next(w for w in range(16) if transform(w,cols)==v)

def standard_map(a,b):
    c=next(x for x in AXES if x!=a);d=next(x for x in AXES if x!=b)
    check=bits(a+b);lz=bits(a+'I');lx=bits(c+d);mapping=[]
    for i,j in product(BITS,repeat=2):
        v=i[0]|j[0]<<1|i[1]<<2|j[1]<<3
        mapping.append(-1 if symp(v,check) else BITS.index((symp(v,lz),symp(v,lx))))
    return tuple(mapping),dict(check=a+b,logical_Z=a+'I',logical_X=c+d)

def outperm(mapping,p):return tuple(-1 if k<0 else p[k] for k in mapping)

def dense_check(records):
    import numpy as np
    I=np.eye(2,dtype=complex);X=np.array([[0,1],[1,0]],complex)
    Z=np.diag([1,-1]);Y=np.array([[0,-1j],[1j,0]],complex)
    H=(X+Z)/np.sqrt(2);S=np.diag([1,1j]);mats={'h':H,'s':S}
    P=[I,X,Y,Z];count=0;worst=0.
    for row in records:
        U=np.eye(4,dtype=complex)
        for g in row['gates']:
            if g[0] in mats:
                G=np.kron(mats[g[0]],I) if g[1]==0 else np.kron(I,mats[g[0]])
            else:
                G=np.zeros((4,4),complex)
                for k in range(4):
                    q=[k//2,k%2];q[g[2]]^=q[g[1]];G[2*q[0]+q[1],k]=1
            U=G@U
        cols=tuple(row['columns'])
        for i,j in product(range(4),repeat=2):
            v=BITS[i][0]|BITS[j][0]<<1|BITS[i][1]<<2|BITS[j][1]<<3
            w=transform(v,cols);a=BITS.index((w&1,(w>>2)&1));b=BITS.index(((w>>1)&1,(w>>3)&1))
            actual=U@np.kron(P[i],P[j])@U.conj().T;expected=np.kron(P[a],P[b])
            overlap=np.trace(expected.conj().T@actual)/4
            error=float(np.max(np.abs(actual-overlap*expected)));worst=max(worst,error)
            assert error<2e-13 and abs(abs(overlap)-1)<2e-13
            count+=1
    return dict(tested=count,maximum_error=worst,method='Independent dense complex 4x4 unitary conjugation')

def main():
    groups={name:bfs(real) for name,real in [('full',False),('real',True)]}
    unique={};members={};real_maps=set()
    for name,group in groups.items():
        for cols,gates in group.items():
            mapping=error_map(cols);check=pauli(pullback(8,cols))
            row=dict(columns=cols,gates=gates,check=check,
                     logical_X=pauli(pullback(1,cols)),logical_Z=pauli(pullback(4,cols)),mapping=mapping)
            unique.setdefault(mapping,row);members.setdefault(mapping,set()).add(name)
            if name=='real':real_maps.add(mapping)
    canonical={a+b:standard_map(a,b) for a,b in product(AXES,repeat=2)}
    output=[];weight_one=[];normal=[]
    for mapping,row in sorted(unique.items()):
        row=dict(row,real_implementable=mapping in real_maps)
        if 'I' in row['check']:
            measured=0 if row['check'][1]=='I' else 1
            rows=[tuple(mapping[4*i+j] for j in range(4)) for i in range(4)]
            if measured==1:rows=list(zip(*rows))
            kept=[r for r in rows if r[0]>=0]
            assert len(kept)==2 and kept[0]==kept[1] and sorted(kept[0])==list(range(4))
            weight_one.append(row);continue
        can,_=canonical[row['check']]
        perm=next(p for p in PERMS if outperm(can,p)==mapping)
        row['canonical_check']=row['check'];row['output_permutation']=perm
        output.append(row)
    for name,(mapping,item) in canonical.items():
        row=unique[mapping]
        normal.append(dict(name=name,**item,mapping=mapping,gates=row['gates'],columns=row['columns'],
                           terminal_reduction={'Z':list(name)},
                           events=[(i//4,i%4,k) for i,k in enumerate(mapping) if k>=0]))
    assert len(unique)==90 and len(output)==54 and len(weight_one)==36
    # Swapping both children and their check axes preserves the accepted map.
    for a,b in product(AXES,repeat=2):
        left=canonical[a+b][0];right=canonical[b+a][0]
        assert left==tuple(right[4*j+i] for i,j in product(range(4),repeat=2))
    validation=dense_check(output+weight_one)
    result=dict(status='passed',symplectic_full=720,symplectic_real=72,
        distinct_zero_flag_maps=90,weight_two_maps=54,weight_one_maps=36,
        real_distinct_maps=len(real_maps),normal_forms=normal,physical_maps=output,
        weight_one_instruments=weight_one,dense_validation=validation,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Complete two-input zero-relative-check bilateral Clifford maps on independent Bell-diagonal children. Output permutations retained explicitly.',
        reduction='All 54 weight-two maps are nine canonical checks followed by one of six I-fixing Bell permutations. Such frames must be transported to parent input axes and final readout. Weight-one maps pass the independent unmeasured child up to a frame and consume extra inputs.',
        not_proved_here='Whole recursive-family coverage/cost/memory transport requires the separate recursive normalization verification.')
    path=HERE/'data/clifford_normal_forms.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','symplectic_full','symplectic_real','distinct_zero_flag_maps','weight_two_maps','weight_one_maps','real_distinct_maps','dense_validation']},indent=2))

if __name__=='__main__':main()
