"""Construct explicit Clifford implementations and valid measurement cuts."""
from algebra import symp,quadratic


def apply_gate(p,g,m):
    mask=(1<<m)-1;x=p&mask;z=p>>m
    if g[0]=='h':
        j=g[1];v=(x>>j^z>>j)&1;x^=v<<j;z^=v<<j
    elif g[0]=='s':z^=((x>>g[1])&1)<<g[1]
    elif g[0]=='cx':
        c,t=g[1:];x^=((x>>c)&1)<<t;z^=((z>>t)&1)<<c
    else:raise ValueError(g)
    return x|(z<<m)


def compile_instrument(checks,lx,lz,m,real=False):
    """Diagonalize checks to Z_(m-1),...,Z_1, and logical Z to Z_0.

    Row operations among checks only relabel public syndromes. The logical
    representative may acquire check factors; this shifts phase/bit labels
    within a public branch and does not change its entropy-based AD yield.
    """
    rows=list(checks)+[lz];gates=[];track_x=lx
    def gate(g):
        nonlocal rows,track_x
        gates.append(g);rows=[apply_gate(p,g,m) for p in rows]
        track_x=apply_gate(track_x,g,m)
    for a in range(m):
        target=m-1-a
        p=rows[a]
        ys=[j for j in range(target+1) if p>>j&1 and p>>(m+j)&1]
        if real:
            assert len(ys)%2==0
            for j in range(0,len(ys),2):gate(('cx',ys[j],ys[j+1]))
        else:
            for j in ys:gate(('s',j))
        p=rows[a]
        for j in range(target+1):
            if p>>j&1:gate(('h',j))
        p=rows[a]
        piv=next(j for j in range(target+1) if p>>(m+j)&1)
        if piv!=target:
            gate(('cx',piv,target));gate(('cx',target,piv));gate(('cx',piv,target))
        p=rows[a]
        for j in range(target):
            if p>>(m+j)&1:gate(('cx',j,target))
        assert rows[a]==1<<(m+target),(rows,a,target)
        for b in range(a+1,m):
            assert not rows[b]>>target&1
            if rows[b]>>(m+target)&1:rows[b]^=rows[a]
    if track_x>>m&1:
        assert not real
        gate(('s',0))
    assert track_x&((1<<m)-1)==1
    assert not track_x>>m&1
    if real:assert all(g[0]!='s' for g in gates)
    return gates


def measurement_cut(gates,m,retain_output=False):
    """Backward propagation of quantum dependence; explicit feasible schedule.

    CNOT with a classical control becomes conditional X; with a quantum
    control and measured target it must retain that target through the gate.
    The returned width is an implementation upper bound, not a minimum.
    """
    needed={0} if retain_output else set()
    coherent=[set() for _ in gates]
    for i in range(len(gates)-1,-1,-1):
        g=gates[i]
        if g[0]=='h':needed.add(g[1]);coherent[i]={g[1]}
        elif g[0]=='s':coherent[i]={g[1]} if g[1] in needed else set()
        else:
            c,t=g[1:]
            if c in needed:needed.add(t);coherent[i]={c,t}
            elif t in needed:coherent[i]={t}
    first={};last={};uses={}
    for i,g in enumerate(gates):
        for q in g[1:]:uses.setdefault(q,i)
        for q in coherent[i]:
            first.setdefault(q,i);last[q]=i
    if retain_output:
        first.setdefault(0,len(gates));last[0]=len(gates)
    # Any operation earlier than a future coherent use must act before its
    # measurement; load a coherent wire at its first (possibly diagonal) use.
    for q in first:first[q]=min(first[q],uses.get(q,first[q]))
    for q in range(m):uses.setdefault(q,len(gates))
    live=set();loaded=set();peak=0;events=[]
    for i in range(len(gates)+1):
        classical=[q for q in range(m) if q not in first and uses[q]==i]
        for q in classical:
            peak=max(peak,len(live)+1);loaded.add(q)
            events.append(dict(step=i,action='load_measure_Z',wire=q,held=len(live)))
        for q in sorted(q for q in first if first[q]==i):
            events.append(dict(step=i,action='load_quantum',wire=q,held=len(live)))
            live.add(q);loaded.add(q);peak=max(peak,len(live))
        for q in sorted(q for q in live if last[q]==i and not(retain_output and q==0)):
            live.remove(q);events.append(dict(step=i,action='measure_Z',wire=q,held=len(live)))
    assert len(loaded)==m
    return dict(M=max(1,peak),events=events,quantum_wires_by_gate=[sorted(x) for x in coherent])
