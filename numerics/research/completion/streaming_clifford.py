"""Exact-memory streaming measurement of a complete stabilizer basis.

All binary Pauli and phase operations are integer-exact. The real option
uses only H and CNOT, including diagonalization of even-Y Paulis.
"""
from common import *
from algebra import rref,symp,quadratic
from circuits import apply_gate


def multiply(a,b,m):
    p,s,c=a;q,t,d=b;mask=(1<<m)-1
    x=p&mask;z=p>>m;u=q&mask;v=q>>m
    exponent=((x&z).bit_count()+(u&v).bit_count()-((x^u)&(z^v)).bit_count()+2*(z&u).bit_count())%4
    assert exponent%2==0
    return [p^q,s^t^(exponent//2),c^d]


def conjugate(row,gate,m):
    p,s,c=row;op,a,*rest=gate;x=(p>>a)&1;z=(p>>(m+a))&1
    if op in ('h','s'):s^=x&z
    else:
        b=rest[0];s^=x*((p>>(m+b))&1)*(((p>>b)&1)^z^1)
    return [apply_gate(p,gate,m),s,c]


def local_combination(rows,future,m):
    """Return one nonzero combination with no support on future inputs."""
    pivots={};mask=future|(future<<m)
    for i,row in enumerate(rows):
        p=row[0]&mask;comb=1<<i
        while p:
            j=p.bit_length()-1
            if j in pivots:p^=pivots[j][0];comb^=pivots[j][1]
            else:pivots[j]=(p,comb);break
        if p==0:return comb
    return None


def cut_entropies(generators,m):
    """S(A)=rank(restriction to complement)-|complement|, in bits."""
    allbits=(1<<m)-1;ent=[]
    for prefix in range(1<<m):
        other=allbits^prefix;mask=other|(other<<m)
        ent.append(len(rref([p&mask for p in generators],2*m))-other.bit_count())
    assert ent[0]==ent[-1]==0 and min(ent)>=0
    return ent


def optimal_order(ent,m):
    count=1<<m;dp=[m+1]*count;parent=[None]*count;dp[0]=0
    for prefix in range(count-1):
        value=max(dp[prefix],ent[prefix]+1)
        for v in range(m):
            if prefix>>v&1:continue
            nxt=prefix|(1<<v)
            if value<dp[nxt]:dp[nxt]=value;parent[nxt]=(prefix,v)
    prefix=count-1;order=[]
    while prefix:prefix,v=parent[prefix];order.append(v)
    return dp[-1],list(reversed(order))


def compile_measurement(generators,m,real=False,order=None,entropies=None):
    generators=list(map(int,generators));assert len(rref(generators,2*m))==m
    assert all(symp(a,b,m)==0 for a in generators for b in generators)
    if real:assert all(quadratic(p,m)==0 for p in generators)
    ent=cut_entropies(generators,m) if entropies is None else entropies
    lower,optorder=optimal_order(ent,m)
    if order is None:order=optorder
    assert sorted(order)==list(range(m))
    rows=[[p,0,1<<i] for i,p in enumerate(generators)]
    live=set();future=(1<<m)-1;loaded=0;peak=0;events=[];gates=[];measurements=[]
    def gate(g):
        nonlocal rows
        assert set(g[1:])<=live
        rows=[conjugate(row,g,m) for row in rows]
        gates.append(g);events.append(dict(action='gate',gate=g))
    for incoming in order:
        live.add(incoming);loaded|=1<<incoming;future^=1<<incoming;peak=max(peak,len(live))
        events.append(dict(action='load',wire=incoming,live=sorted(live)))
        while rows:
            comb=local_combination(rows,future,m)
            if comb is None:break
            pivotrow=(comb&-comb).bit_length()-1;selected=[0,0,0]
            for j,row in enumerate(rows):
                if comb>>j&1:selected=multiply(selected,row,m)
            rows[pivotrow]=selected;p=selected[0]
            ys=[j for j in live if p>>j&1 and p>>(m+j)&1]
            if real:
                assert len(ys)%2==0
                for a,b in zip(ys[::2],ys[1::2]):gate(('cx',a,b))
            else:
                for j in ys:gate(('s',j))
            for j in sorted(live):
                if rows[pivotrow][0]>>j&1:gate(('h',j))
            supp=[j for j in sorted(live) if rows[pivotrow][0]>>(m+j)&1]
            target=supp[-1]
            for j in supp[:-1]:gate(('cx',j,target))
            selected=rows[pivotrow];assert selected[0]==1<<(m+target)
            for j,row in enumerate(rows):
                if j==pivotrow:continue
                assert not(row[0]>>target&1)
                if row[0]>>(m+target)&1:rows[j]=multiply(row,selected,m)
            item=dict(wire=target,original_generator_mask=selected[2],offset=selected[1])
            measurements.append(item);events.append(dict(action='measure_Z',**item))
            rows.pop(pivotrow);live.remove(target)
        assert len(live)==ent[loaded],(order,incoming,len(live),ent[loaded])
    assert not live and not rows and len(measurements)==m
    assert len(rref([v['original_generator_mask'] for v in measurements],m))==m
    if real:assert all(g[0]!='s' for g in gates)
    assert peak==max(1+ent[sum(1<<j for j in order[:i])] for i in range(m))
    return dict(M=peak,M_min=lower,minimum_proved=peak==lower,real=real,input_order=order,
                generators=generators,gates=gates,events=events,measurements=measurements,
                meaning='For each measured Z bit b, parity(original generator outcomes, original_generator_mask)=b XOR offset. Invert this binary map to reconstruct all terminal outputs. Bob implements the conjugate circuit.')


def graph_generators(m,edges):
    rows=[1<<j for j in range(m)]
    for a,b in edges:rows[a]^=1<<(m+b);rows[b]^=1<<(m+a)
    return rows
