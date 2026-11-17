"""Four explicit real two-input decoders, including YY and XZ checks.

The full class also permits a final Y measurement. This is a registered
generator, not an enumeration of arbitrary memory-bounded protocols.
"""
from core import *
from recursive_library import PRE,pumping_outputs
from functools import lru_cache

PRIMITIVES={
    'Z':dict(ED='E00045',check='ZZ',logical_X='XX',logical_Z='ZI',label='bilateral CNOT'),
    'X':dict(ED='E00021',check='XX',logical_X='ZZ',logical_Z='XI',label='H on both inputs, then bilateral CNOT'),
    'R':dict(ED='E00029',check='YY',logical_X='ZX',logical_Z='XX',label='real YY decoder'),
    'C':dict(ED='E00010',check='XZ',logical_X='ZX',logical_Z='XI',label='H on the first input, then bilateral CNOT'),
}
LABELS=((0,0),(1,0),(1,1),(0,1))
PAULI_BITS={'I':(0,0),'X':(1,0),'Y':(1,1),'Z':(0,1)}


def parity(error,operator):
    return sum(x*PAULI_BITS[q][1]+z*PAULI_BITS[q][0] for (x,z),q in zip(error,operator))%2


MAPS={}
for name,item in PRIMITIVES.items():
    accepted=[]
    for i,a in enumerate(LABELS):
        for j,b in enumerate(LABELS):
            if parity((a,b),item['check'])==0:
                accepted.append((i,j,LABELS.index((parity((a,b),item['logical_Z']),parity((a,b),item['logical_X'])))))
    MAPS[name]=accepted


def step(p,q,primitive):
    raw=np.zeros_like(np.broadcast_arrays(p,q)[0])
    for i,j,k in MAPS[primitive]:raw[...,k]+=p[...,i]*q[...,j]
    s=raw.sum(-1)
    return raw/s[...,None],s


def reduction(primitive,axis):
    if primitive=='Z' and axis=='Z':return 'Z','Z'
    if primitive=='X' and axis=='Z':return 'X','X'
    if primitive=='R' and axis=='Y':return 'Y','Y'
    if primitive=='C' and axis=='Z':return 'X','Z'
    return None


@lru_cache(None)
def memory(path,axis):
    if not path:return 1
    targets=reduction(path[-1],axis)
    if targets is None:return len(path)+1
    return max(memory(path[:-1],a) for a in targets)


def generate(p,depth=7,permissions='real',rmax=160):
    p=np.asarray(p);bases=('Z','X') if permissions=='real' else ('Z','X','Y')
    ys=[];ps=[];Ms=[];records=[];frontier=[('',p,np.ones(p.shape[:-1]))]
    for d in range(depth+1):
        nxt=[]
        for path,state,y in frontier:
            for axis in bases:
                M=memory(path,axis)
                ps.append(state[...,list(PRE[axis])]);ys.append(y);Ms.append(M)
                records.append(dict(family='real_decoder_recurrence',path=path,final_axis=axis,depth=d,
                                    memory_upper_bound=M,minimal_memory_proved=False,
                                    successful_ancestry_inputs=2**d,decoder_register_model='v2_four_real_primitives'))
            if d<depth:
                for primitive in PRIMITIVES:
                    state2,s=step(state,state,primitive);nxt.append((path+primitive,state2,y*s/2))
        frontier=nxt
    for primitive in PRIMITIVES:
        state=p.copy();cost=np.ones(p.shape[:-1])
        for r in range(1,rmax+1):
            if r>1:state,s=step(state,p,primitive);cost=(cost+1)/s
            for axis in bases:
                # This explicit schedule stores the survivor and receives one
                # raw input. Only a completely removable chain is assigned M1.
                removable=True;current=axis
                for _ in range(r-1):
                    targets=reduction(primitive,current)
                    if targets is None:removable=False;break
                    current=targets[0]
                M=1 if removable else 2
                ps.append(state[...,list(PRE[axis])]);ys.append(1/cost);Ms.append(M)
                records.append(dict(family='real_decoder_pumping',primitive=primitive,r=r,final_axis=axis,
                                    memory_upper_bound=M,minimal_memory_proved=False))
    return np.array(ps),np.array(ys),np.array(Ms),records


def prepare(p,record):
    p=np.asarray(p);raw=p.copy();y=np.ones(p.shape[:-1])
    if record['family']=='real_decoder_recurrence':
        for primitive in record['path']:p,s=step(p,p,primitive);y=y*s/2
    else:
        cost=np.ones(p.shape[:-1])
        for _ in range(record['r']-1):p,s=step(p,raw,record['primitive']);cost=(cost+1)/s
        y=1/cost
    return p[...,list(PRE[record['final_axis']])],y


def verify():
    entries={e['id']:e for e in json.loads((OLD/'data/blocks.json').read_text())['entries']}
    from circuits import apply_gate
    rng=np.random.default_rng(7723);error=0.
    for name,item in PRIMITIVES.items():
        e=entries[item['ED']]
        assert e['real'] and all(g[0] in ('cx','h') for g in e['gates'])
        item['retained_output_decoder_gates']=e['gates']
        for i,j,k in MAPS[name]:
            x0,z0=LABELS[i];x1,z1=LABELS[j]
            word=x0+2*x1+4*z0+8*z1
            for gate in e['gates']:word=apply_gate(word,gate,2)
            assert (word>>1)&1==0
            frame=e['compiled_frame']['0']
            observed=(word&1^frame['bit_flip'],(word>>2)&1^frame['phase_flip'])
            assert observed==LABELS[k]
        for p in rng.dirichlet(np.ones(4),30):
            value,s=step(p,p,name)
            raw=np.array(e['tensor'][0])@monomials(p,compositions(2))
            error=max(error,float(np.max(np.abs(value*s-raw))))
            assert np.max(np.abs(value*s-raw))<1e-14,(name,value*s,raw)
    result=dict(primitives=PRIMITIVES,checks=120,max_error=error,
                note='Independent 16-event symplectic propagation agrees with the previously dense-unitary-verified real Clifford decoders.')
    atomic_json(DATA/'recurrence_v2_checks.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':verify()
