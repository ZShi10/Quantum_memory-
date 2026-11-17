"""Outward full registered recurrence/pumping/AD menu upper bounds."""
from core import *
from fractions import Fraction as F
from recurrence_v2 import PRIMITIVES,memory,reduction
from recursive_library import PRE
from certify_bb84_recursive import step
from bb84_fixed import physical_kernel
from long_code_bounds import best_bounds
import stable_intervals as iv
import argparse,time


def legacy_generate_bounds(q,permissions='real',depth=7,rmax=160):
    q=F(str(q));row=[iv.rational(v) for v in (1-F(3,2)*q,q/2,q/2,q/2)]
    initial=tuple(np.array([[v[i] for v in row]]) for i in (0,1));frontier=initial
    y=(np.ones(1),np.ones(1));paths=[''];bases=('Z','X') if permissions=='real' else ('Z','X','Y')
    ps=[];ys=[];Ms=[]
    for d in range(depth+1):
        ps.append(tuple(np.stack([v[:,list(PRE[a])] for a in bases],1).reshape(-1,4) for v in frontier))
        ys.append(tuple(np.repeat(v,len(bases)) for v in y));Ms.extend(memory(path,axis) for path in paths for axis in bases)
        if d<depth:
            nxt=tuple(np.empty((len(paths)*4,4)) for _ in (0,1));yy=tuple(np.empty(len(paths)*4) for _ in (0,1))
            for i,primitive in enumerate(PRIMITIVES):
                state,s=step(frontier,frontier,primitive);scale=iv.scale(iv.mul(y,s),F(1,2))
                for j in (0,1):nxt[j][i::4]=state[j];yy[j][i::4]=scale[j]
            frontier=nxt;y=yy;paths=[path+c for path in paths for c in PRIMITIVES]
    for primitive in PRIMITIVES:
        state=initial;y=(np.ones(1),np.ones(1))
        for r in range(1,rmax+1):
            if r>1:
                state,s=step(state,initial,primitive);y=iv.div(iv.mul(y,s),iv.add((np.ones(1),np.ones(1)),y))
            for axis in bases:
                removable=True;current=axis
                for _ in range(r-1):
                    targets=reduction(primitive,current)
                    if targets is None:removable=False;break
                    current=targets[0]
                Ms.append(1 if removable else 2);ps.append(tuple(v[:,list(PRE[axis])] for v in state));ys.append(y)
    return tuple(np.concatenate([v[i] for v in ps]) for i in (0,1)),tuple(np.concatenate([v[i] for v in ys]) for i in (0,1)),np.array(Ms)


def generate_bounds(q,permissions='real',depth=7,rmax=160):
    from recurrence_interval_library import generate_bounds as registered_bounds,records
    q=F(str(q));row=[iv.rational(v) for v in (1-F(3,2)*q,q/2,q/2,q/2)]
    initial=tuple(np.array([[v[i] for v in row]]) for i in (0,1))
    p,y=registered_bounds(initial,permissions,depth,rmax);M,_=records(permissions,depth,rmax)
    return tuple(v[:,0,:] for v in p),tuple(v[:,0] for v in y),M


def certify(q,permissions='real',tolerance=1e-10):
    start=time.time();p,y,M=generate_bounds(q,permissions);kernels=load_codes();physical={k.entry['id']:physical_kernel(k) for k in kernels}
    real_cover=None
    if permissions=='full':
        real_path=DATA/'werner_v2_real_menu_certified.json'
        if real_path.exists():real_cover=json.loads(real_path.read_text())['results'].get(f'{q:.12g}')
        if real_cover:
            from recurrence_interval_library import records
            _,rr=records('full');ids=np.array([i for i,r in enumerate(rr) if r['final_axis']=='Y'])
            p=tuple(v[ids] for v in p);y=tuple(v[ids] for v in y);M=M[ids]
    witness_path=DATA/f'werner_v2_{permissions}_certified.json';witness=json.loads(witness_path.read_text())['results']
    lower=np.zeros(9);lower_exact={}
    for m in range(1,9):
        row=witness[f'{q:.12g}_M{m}'];lower_exact[f'M{m}']=row['lower'];lower[m]=max(0.,iv.down(float(row['lower'])))
    pending=DATA/'pending_werner_menu';pending.mkdir(exist_ok=True)
    checkpoint=pending/f'{permissions}_{q:.12g}.json'
    saved=json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    upper=np.array(saved.get('upper',[0.]*9));active=np.ones(len(M),bool);history=saved.get('history',[])
    if real_cover:
        for m in range(1,9):upper[m]=max(upper[m],real_cover['results'][f'M{m}']['upper'])
    for B in range(17):
        if B<len(history):
            if B<16:p,s=step(p,p,'Z');y=iv.scale(iv.mul(y,s),F(1,2))
            continue
        c=np.maximum(0,iv.up(2*p[1].max(-1)-1));er=iv.up(y[1]*iv.up(c*c));er=np.minimum(er,y[1])
        removed=active&(er<=lower[M]+tolerance);active[removed]=False
        for budget in range(1,9):
            ids=np.flatnonzero(removed&(M<=budget))
            if len(ids):upper[budget]=max(upper[budget],float(er[ids].max()))
        ids=np.flatnonzero(active)
        if not len(ids):break
        target=np.maximum(0,(lower[M[ids]]+tolerance)/np.maximum(y[1][ids],1e-300))
        rl,rh=best_bounds((p[0][ids],p[1][ids]),kernels,physical,128,target)
        bound=np.minimum(er[ids],iv.up(y[1][ids]*rh))
        assert np.isfinite(bound).all() and not np.isnan(rl).any(),(q,B,'nonfinite rate enclosure')
        for budget in range(1,9):
            keep=M[ids]<=budget
            if keep.any():upper[budget]=max(upper[budget],float(bound[keep].max()))
        history.append(dict(B=B,AD_states=len(ids),removed_forever_by_ER=int(removed.sum()),
                            largest_excess=float(np.max(bound-lower[M[ids]]))))
        atomic_json(checkpoint,dict(Q=q,permissions=permissions,upper=upper.tolist(),history=history))
        print('Werner upper round',permissions,q,B,'states',len(ids),'seconds',round(time.time()-start,1),flush=True)
        if B<16:
            pp,s=step((p[0][ids],p[1][ids]),(p[0][ids],p[1][ids]),'Z');scale=iv.scale(iv.mul((y[0][ids],y[1][ids]),s),F(1,2))
            for j in (0,1):p[j][ids]=pp[j];y[j][ids]=scale[j]
    results={}
    for m in range(1,9):
        assert upper[m]+1e-15>=lower[m],(q,m,upper[m],lower[m])
        results[f'M{m}']=dict(lower=lower_exact[f'M{m}'],upper=float(upper[m]),gap_upper=float(iv.up(upper[m]-lower[m])))
    return dict(Q=q,permissions=permissions,results=results,history=history,seconds=time.time()-start,
                real_class_covered_by='werner_v2_real_menu_certified.json' if real_cover else None,
                scope='All four-real-decoder words depth<=7, constant-decoder raw pumping<=160, declared final axes and explicit memory schedules; B<=16; all binary codes n<=7 via direct sums, repetition/SPC n<=128, three Du net modes and Watanabe. Outward complete-menu upper and independent 60-digit witness lower; no arbitrary fixed-memory global-optimum claim.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--q',nargs='+',type=float,default=[.2,.3,.32]);parser.add_argument('--permissions',choices=['real','full'],default='real')
    parser.add_argument('--tolerance',type=float,default=1e-10);a=parser.parse_args();path=DATA/f'werner_v2_{a.permissions}_menu_certified.json'
    out=json.loads(path.read_text()) if path.exists() else dict(results={})
    for q in a.q:
        key=f'{q:.12g}'
        if key in out['results']:continue
        out['results'][key]=certify(q,a.permissions,a.tolerance);atomic_json(path,out)
        print('Werner menu certificate',a.permissions,q,out['results'][key]['results'],flush=True)


if __name__=='__main__':main()
