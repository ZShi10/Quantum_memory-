"""Independent local POVM comparison for quantum/classical measurement cuts.

Comparing every POVM effect checks the conditional Eve state as well as local
outcome probabilities, since Eve is untouched and the local input may be
entangled with her. This audit uses dense operators only for small examples.
"""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data'


def unitary(g,m):
    dim=1<<m;U=np.zeros((dim,dim),dtype=complex)
    for x in range(dim):
        if g[0]=='h':
            q=g[1];b=x>>q&1
            U[x&~(1<<q),x]=1/np.sqrt(2)
            U[x|(1<<q),x]=(-1)**b/np.sqrt(2)
        elif g[0]=='s':U[x,x]=1j**(x>>g[1]&1)
        elif g[0]=='cx':
            c,t=g[1:];U[x^(((x>>c)&1)<<t),x]=1
        elif g[0]=='x':U[x^(1<<g[1]),x]=1
    return U


def check(entry):
    m=entry['m'];dim=1<<m;gates=entry['terminal_gates'];cut=entry['terminal_cut']
    U=np.eye(dim,dtype=complex)
    for g in gates:U=unitary(g,m)@U
    states=[({},np.eye(dim,dtype=complex))]
    def measure(wire):
        nonlocal states
        result=[]
        for record,K in states:
            assert wire not in record
            for b in (0,1):
                projected=K.copy()
                projected[[i for i in range(dim) if (i>>wire&1)!=b]]=0
                if np.max(np.abs(projected))>1e-14:result.append(({**record,wire:b},projected))
        states=result
    for step in range(len(gates)+1):
        for event in cut['events']:
            if event['step']==step and event['action']=='load_measure_Z':measure(event['wire'])
        if step<len(gates):
            g=gates[step];updated=[]
            for record,K in states:
                if g[0] in ('h','s'):
                    if g[1] in record:
                        assert g[0]=='s'
                    else:K=unitary(g,m)@K
                else:
                    c,t=g[1:]
                    if c in record:
                        if t in record:record={**record,t:record[t]^record[c]}
                        elif record[c]:K=unitary(('x',t),m)@K
                    else:
                        assert t not in record
                        K=unitary(g,m)@K
                updated.append((record,K))
            states=updated
        for event in cut['events']:
            if event['step']==step and event['action']=='measure_Z':measure(event['wire'])
    effects={}
    for record,K in states:
        assert len(record)==m
        y=sum(b<<j for j,b in record.items())
        effects[y]=effects.get(y,0)+K.conj().T@K
    discrepancy=0.
    for y in range(dim):
        expected=np.outer(U[y].conj(),U[y])
        discrepancy=max(discrepancy,float(np.max(np.abs(effects.get(y,0)-expected))))
    assert discrepancy<2e-12,(entry['id'],discrepancy)
    # The new measurement and the original decoder differ by a monomial
    # matrix, hence a classical outcome permutation (plus irrelevant phases).
    original=np.eye(dim,dtype=complex)
    for g in entry['gates']:original=unitary(g,m)@original
    transition=original@U.conj().T
    assert np.max(np.abs(np.sort(np.abs(transition),axis=1)[:,:-1]))<2e-12
    assert np.max(np.abs(np.max(np.abs(transition),axis=1)-1))<2e-12
    return discrepancy


def main():
    entries=json.loads((DATA/'blocks.json').read_text())['entries']
    # Every implementation through m=3, plus 180 reproducibly sampled m=4
    # examples stratified across real/full and all implemented widths.
    chosen=[e for e in entries if e['m']<=3]
    rng=np.random.default_rng(18260919)
    for real in (False,True):
        for memory in range(1,5):
            pool=[e for e in entries if e['m']==4 and e['real']==real and e['M']==memory]
            chosen += [pool[j] for j in rng.choice(len(pool),min(24,len(pool)),replace=False)]
    maximum=max(check(entry) for entry in chosen)
    record=dict(status='passed',instruments=len(chosen),maximum_POVM_discrepancy=maximum,
                scope='All local POVM effects for every m<=3 circuit and a stratified m=4 sample; algebraic cut rules apply generally.')
    (DATA/'measurement_cut_verification.json').write_text(json.dumps(record,indent=2))
    print(record)


if __name__=='__main__':main()
