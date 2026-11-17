"""State-independent recurrence words on the entire BB84 Bell family.

Discovery uses a parameter grid. Independent continuous enclosures are
required before these records are called guaranteed rates.
"""
from core import *
from recursive_library import PRE,terminal_schedule
from literature_rates import watanabe
import argparse,time


def states(b,p,ts,depth=7,permissions='real',rmax=160):
    ts=np.asarray(ts);z=min(b,p)*ts
    raw=np.stack((1-b-p+z,b-z,z,p-z),-1)
    bases=('Z','X') if permissions=='real' else ('Z','X','Y')
    frontier=[('',raw,np.ones(len(ts)))];records=[];out=[];ys=[];mem=[]
    for d in range(depth+1):
        nxt=[]
        for path,state,y in frontier:
            for axis in bases:
                sched=terminal_schedule(path,axis)
                records.append(dict(family='homogeneous_recurrence',path=path,final_axis=axis,depth=d,**sched))
                out.append(state[...,list(PRE[axis])]);ys.append(y);mem.append(sched['memory_upper_bound'])
            if d<depth:
                for axis in bases:
                    pp,s=b_step(state[...,list(PRE[axis])]);nxt.append((path+axis,pp,y*s/2))
        frontier=nxt
    for axis in bases:
        pp=raw[...,list(PRE[axis])];a,b0,c0,c1=bell_parameters(pp)
        A=B=v=w=np.ones(len(ts));prefix=np.ones(len(ts))
        for r in range(1,rmax+1):
            A=A*a;B=B*b0;v=v*a*c0;w=w*b0*c1;mass=A+B
            state=np.stack((A+v,B+w,B-w,A-v),-1)/(2*mass[:,None]);y=mass/prefix
            prefix=prefix+mass
            for final in bases:
                M=1 if r==1 or final=='Z' else 2
                out.append(state[...,list(PRE[final])]);ys.append(y);mem.append(M)
                records.append(dict(family='raw_restart_pumping',r=r,preparation_axis=axis,final_axis=final,memory_upper_bound=M))
    return np.array(out),np.array(ys),np.array(mem),records


def prepare(b,p,ts,record,rounds=0):
    ts=np.asarray(ts);z=min(b,p)*ts
    initial=np.stack((1-b-p+z,b-z,z,p-z),-1);state=initial.copy();y=np.ones(len(ts))
    if record['family'].startswith('real_decoder_'):
        from recurrence_v2 import prepare as v2prepare
        state,y=v2prepare(initial,record)
        for _ in range(rounds):state,s=b_step(state);y=y*s/2
        return state,y
    if record['family']=='homogeneous_recurrence':
        for axis in record['path']:
            state,s=b_step(state[...,list(PRE[axis])]);y=y*s/2
    else:
        state=initial[...,list(PRE[record['preparation_axis']])]
        a,b0,c0,c1=bell_parameters(state);r=record['r']
        A=a**r;B=b0**r;v=(a*c0)**r;w=(b0*c1)**r;mass=A+B
        prefix=1+sum(a**j+b0**j for j in range(1,r))
        state=np.stack((A+v,B+w,B-w,A-v),-1)/(2*mass[:,None]);y=mass/prefix
    state=state[...,list(PRE[record['final_axis']])]
    for _ in range(rounds):state,s=b_step(state);y=y*s/2
    return state,y


def probe(b,p,permissions,v2=False):
    ts=np.linspace(0,1,65)
    if v2:
        from recurrence_v2 import generate
        z=min(b,p)*ts;initial=np.stack((1-b-p+z,b-z,z,p-z),-1)
        ps,ys,M,records=generate(initial,7,permissions)
    else:ps,ys,M,records=states(b,p,ts,permissions=permissions)
    best=np.zeros(len(ps));choices=[None]*len(ps)
    for k in range(17):
        rates=ys*watanabe(ps);worst=rates.min(-1);take=worst>best
        for i in np.flatnonzero(take):
            best[i]=worst[i];choices[i]=dict(kind='Watanabe_2007',classical_B_rounds=k,sampled_argmin=float(ts[rates[i].argmin()]))
        ps,s=b_step(ps);ys=ys*s/2
    result={}
    for budget in range(1,9):
        ids=np.flatnonzero(M<=budget);i=int(ids[best[ids].argmax()])
        result[f'M{budget}']=dict(sampled_worst_rate=float(best[i]),ED=records[i],AD=choices[i])
    return dict(b=b,p=p,permissions=permissions,results=result,
                generator='four_real_decoders_v2' if v2 else 'common_input_basis_v1',
                scope='Fixed state-independent ED and B choices, Watanabe terminal. Grid of 65 unknown correlations is discovery only; no continuous-z guarantee from this file alone.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--q',nargs='+',type=float);parser.add_argument('--p',type=float)
    parser.add_argument('--permissions',choices=['real','full'],default='real');parser.add_argument('--dense',action='store_true');parser.add_argument('--v2',action='store_true');a=parser.parse_args()
    path=DATA/f'bb84_recursive_watanabe_{a.permissions}{"_v2" if a.v2 else ""}.json'
    output=json.loads(path.read_text()) if path.exists() else dict(results=[])
    done={(r['b'],r['p']) for r in output['results']}
    for b in np.linspace(0,.25,501) if a.dense else a.q or [.1,.15,.18,.2,.22,.24]:
        p=b if a.p is None else a.p
        if (float(b),float(p)) in done:continue
        start=time.time();r=probe(float(b),float(p),a.permissions,a.v2);output['results'].append(r);atomic_json(path,output)
        print('BB recursive W',b,p,{k:v['sampled_worst_rate'] for k,v in r['results'].items()},round(time.time()-start,1),flush=True)


if __name__=='__main__':main()
