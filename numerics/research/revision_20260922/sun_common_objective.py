"""Controlled Sun-size references; never relabel these as author data.

The zero-relative-syndrome repetition reference changes only the final
entropy bound. The strengthened control gives the same zero-flag ED
weight classes our full classical menu and includes bypassing ED.
"""
from core import *
from itertools import permutations
from known_search import optimize
import argparse,time,csv

PERMS=[(0,)+p for p in permutations((1,2,3))]
SIZES=[(2,1),(2,2),(2,3),(2,4),(3,1),(3,2),(3,3),(4,1),(4,2)]


def library():
    return json.loads((PROJECT/'manuscript/data/small_clifford_library.json').read_text())['entries']


def outputs(q):
    pp=[np.array([1-1.5*q,q/2,q/2,q/2])];yy=[1.];mem=[1]
    records=[dict(family='direct_measurement',m=1,memory_upper_bound=1)]
    for i,e in enumerate(library()):
        m=e['m'];power=np.array([(1-1.5*q)**(m-w)*(q/2)**w for w in range(m+1)])
        raw=np.array(e['weights'])@power;mass=raw.sum()
        for frame in PERMS:
            pp.append(raw[list(frame)]/mass);yy.append(mass/m);mem.append(2)
            records.append(dict(family='Sun_size_zero_flag_ED',library_index=i,m=m,
                                frame=list(frame),memory_upper_bound=2,
                                memory_reason='Complete terminal stabilizer measurement m<=4 admits two live registers; this is not a retained-quantum-output claim.'))
    return np.array(pp),np.array(yy),np.array(mem),records


def zero_repetition_reference():
    f=np.linspace(.5,1,500);curves={};winners={}
    for m,n in SIZES:
        best=np.zeros(len(f));chosen=[None]*len(f)
        for i,e in enumerate(library()):
            if e['m']!=m:continue
            powers=np.stack([f**(m-w)*((1-f)/3)**w for w in range(m+1)],axis=1)
            raw=powers@np.asarray(e['weights']).T;mass=raw.sum(-1);state=raw/mass[:,None]
            for frame in PERMS:
                a,x,y,z=state[:,frame].T
                aa=(a+z)**n;bb=(x+y)**n;cc=(a-z)**n;dd=(x-y)**n
                r=np.stack((aa+cc,bb+dd,bb-dd,aa-cc),-1)/2;s=r.sum(-1)
                rate=mass*s/(m*n)*np.maximum(0.,1-entropy(r/s[:,None]))
                for j in np.flatnonzero(rate>best):chosen[int(j)]=dict(library_index=i,frame=list(frame))
                best=np.maximum(best,rate)
        curves[f'{m}_{n}']=best;winners[f'{m}_{n}']=chosen
    names=list(curves)
    with (DATA/'sun_common_zero_repetition.csv').open('w',newline='') as file:
        writer=csv.writer(file);writer.writerow(['F','Q']+names+['envelope'])
        for j in reversed(range(len(f))):
            vv=[curves[k][j] for k in names];writer.writerow([f[j],2*(1-f[j])/3]+vv+[max(vv)])
    atomic_json(DATA/'sun_common_zero_repetition_metadata.json',dict(
        scope='Same nine ED/repetition sizes and zero relative flags as reconstructed Sun convention; replace separate bit/phase entropy by joint Bell entropy. This is a recomputed control, not published data.',
        source='sun_reconstruction_audit.json',sizes=SIZES,
        memory='At most two live local qubits after complete-terminal-measurement compilation; no false assignment M=m.',
        permutations=PERMS))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--q',nargs='+',type=float)
    parser.add_argument('--dense',action='store_true');a=parser.parse_args()
    zero_repetition_reference();kernels=load_codes();path=DATA/'sun_common_full_AD.json'
    out=json.loads(path.read_text()) if path.exists() else dict(results=[])
    done={r['Q'] for r in out['results']}
    qs=np.linspace(0,1/3,500) if a.dense else a.q or [.1,.2,.25,.28,.3,.32]
    for q in qs:
        if float(q) in done:continue
        started=time.time();p,y,M,records=outputs(q)
        r=optimize(p,y,M,records,kernels,with_watanabe=True)
        r.update(Q=float(q),seconds=time.time()-started)
        out['results'].append(r)
        out['scope']='Controlled enlargement of the twenty zero-flag Sun-size Werner ED weight classes m=2,3,4, plus ED bypass. Same n<=7 all binary codes, repetition/SPC<=128, three paid Du modes, Watanabe and pooled B<=16 as the larger search. M1 is strong AD-only; M2 also includes Sun-size ED compiled as complete terminal measurements. Floating search; original author curves remain separate.'
        atomic_json(path,out);print('Sun common AD',q,{k:v['rate'] for k,v in r['results'].items()},round(time.time()-started,2),flush=True)


if __name__=='__main__':main()
