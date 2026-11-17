"""Executable homogeneous binary recurrences with explicit terminal reduction."""
from core import *

PRE = {'Z':(0,1,2,3), 'X':(0,3,2,1), 'Y':(0,1,3,2)}
AXES = ('Z','X','Y')


def terminal_schedule(path, final_axis):
    """Move trailing Z-key B checks to classical pools, including input basis.

    Q means pre-rotate Q into Z, do a Z bilateral CNOT check, and do not
    undo the rotation. Measuring the output in Z can therefore be moved
    before this last check by measuring BOTH children in Q.
    """
    path,axis=str(path),str(final_axis)
    removed=[]
    while path and axis=='Z':
        axis=path[-1]
        removed.append(path[-1])
        path=path[:-1]
    return dict(quantum_path=path,quantum_final_axis=axis,
                classicalized_levels=len(removed),
                memory_upper_bound=max(1,len(path)+1),
                minimal_memory_proved=False)


def generate(p,max_depth=7,permissions='real'):
    p=np.asarray(p,dtype=float)
    records=[];states=[];yields=[];memories=[]
    frontier=[('',p,1.)]
    bases=AXES if permissions=='full' else AXES[:2]
    for depth in range(max_depth+1):
        following=[]
        for path,state,y in frontier:
            for final in bases:
                schedule=terminal_schedule(path,final)
                records.append(dict(family='homogeneous_recurrence',path=path,
                                    final_axis=final,depth=depth,
                                    successful_ancestry_inputs=2**depth,
                                    ED_expected_raw_cost=1/y,**schedule))
                states.append(state[list(PRE[final])]);yields.append(y)
                memories.append(schedule['memory_upper_bound'])
            if depth<max_depth:
                for basis in bases:
                    out,s=b_step(state[list(PRE[basis])])
                    following.append((path+basis,out,y*s/2))
        frontier=following
    return np.asarray(states),np.asarray(yields),np.asarray(memories),records


def greedy(p,max_depth=7):
    """Smallest Pauli component rule; known input only, local frames recorded."""
    out=[dict(depth=0,path='',p=list(map(float,p)),yield_per_raw_pair=1.)]
    p=np.asarray(p);y=1.;path=''
    for depth in range(1,max_depth+1):
        basis={1:'X',2:'Y',3:'Z'}[int(np.argmin(p[1:]))+1]
        p,s=b_step(p[list(PRE[basis])]);y*=s/2;path+=basis
        out.append(dict(depth=depth,path=path,p=list(map(float,p)),yield_per_raw_pair=float(y)))
    return out


def pumping_outputs(p,rmax=160,permissions='real'):
    """Retained pair plus one raw pair; failure restarts the whole pump."""
    p=np.asarray(p);bases=AXES if permissions=='full' else AXES[:2]
    states=[];costs=[];memories=[];records=[]
    for preparation in bases:
        raw=p[list(PRE[preparation])]
        a,b,c0,c1=bell_parameters(raw)
        A=B=1.;v=w=1.;prefix=1.
        for r in range(1,rmax+1):
            A*=a;B*=b;v*=a*c0;w*=b*c1
            mass=A+B
            state=np.array([A+v,B+w,B-w,A-v])/(2*mass)
            cost=prefix/mass
            # One input plus the success probability of each nonfinal prefix.
            prefix+=mass
            for final in bases:
                M=1 if r==1 or final=='Z' else 2
                states.append(state[list(PRE[final])]);costs.append(cost);memories.append(M)
                records.append(dict(family='raw_restart_pumping',r=r,
                                    preparation_axis=preparation,final_axis=final,
                                    memory_upper_bound=M,ED_expected_raw_cost=float(cost),
                                    classicalized=bool(final=='Z')))
    return np.array(states),1/np.array(costs),np.array(memories),records
