"""Fixed recurrence and B prefix, fixed mixtures of the full AD menu.

Complete registered ED/B candidates are screened, but the screening is
floating; only each selected policy's continuous-z lower bound is outward.
No hidden-z code, B-prefix or ED choice is permitted.
"""
from core import *
from recurrence_v2 import generate
from bb84_recursive import prepare
from compound_terminals import actions,values,terminal_jet
from ad_modes import best_short,repetition_rates,spc_otp_rates
from known_search import canonical_classical_states,spc_upper
from spc_dynamic_fast import all_rates as spc_rates
from literature_rates import watanabe
from fractions import Fraction as F
from recurrence_equivalence import representatives
import interval_jets as jets
import argparse,subprocess,heapq,time,hashlib


def known_max(p,nmax=128,target=None):
    """Every terminal at one known input, used ONLY as an upper relaxation."""
    unique,inverse=np.unique(canonical_classical_states(p),axis=0,return_inverse=True)
    best=np.maximum(best_short(unique,load_codes())['rate'],watanabe(unique));ns=np.arange(8,nmax+1)
    for n in ns:
        best=np.maximum(best,repetition_rates(unique,int(n)).max(-1))
        best=np.maximum(best,spc_otp_rates(unique,int(n)).max(-1))
    if len(ns):
        threshold=best.copy()
        if target is not None:
            required=np.full(len(unique),np.inf);np.minimum.at(required,inverse,target)
            threshold=np.maximum(threshold,required)
        upper=spc_upper(unique,ns);needed=upper>threshold[:,None]
        exact=spc_rates(unique,ns,needed)
        # Omitted SPC candidates keep their analytic upper relaxation, so
        # the result remains an upper bound for EVERY terminal at this input.
        best=np.maximum(best,np.where(needed,exact,upper).max(-1))
    return best[inverse]+3e-13


def curve(b,p,intervals,ed,B,strategy,coarse=False):
    state,y=jets.prepare(b,p,intervals,ed,B);value=jets.Jet((np.zeros(len(intervals)),np.zeros(len(intervals))))
    for item in strategy:
        ad=item['AD']
        if coarse and ad['kind']=='SPC' and ad['mode']=='no_otp':
            from coarse_spc import bound
            rate=bound(state,ad['n'])
        else:rate=terminal_jet(state,ad)
        value=value+F(item['allocation_exact'])*rate
    return y*value


def certify_policy(b,p,ed,B,strategy,atol=2e-9,max_cells=50000):
    from terminal_equivalence import reuse
    reused=reuse(b,p,ed,B,strategy,atol)
    if reused is not None:return reused
    from z_pump_rectangle import supported,certify as rectangle_certificate
    if supported(ed,B,strategy):return rectangle_certificate(b,p,ed,B,strategy,atol,max_cells)
    heap=[];counter=0;incumbent=np.inf;argmin=None
    progress_path=ROOT/'tmp'/f'joint_lp_{os.getpid()}'/'certificate_status.json'
    progress_path.parent.mkdir(parents=True,exist_ok=True)
    identity=dict(b=b,p=p,ED=ed,B=B,strategy=strategy,engine='unnormalized_jet_closed_walk_derivative_v1')
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    work=DATA/'bb84_interval_work'/f'{digest}.json';work.parent.mkdir(parents=True,exist_ok=True)
    def evaluate(intervals):return curve(b,p,intervals,ed,B,strategy)
    def add(intervals):
        nonlocal counter,incumbent,argmin
        if any(v['AD']['kind']=='SPC' and v['AD']['mode']=='no_otp' for v in strategy):
            cheap=curve(b,p,intervals,ed,B,strategy,coarse=True).v[0]
            keep=[]
            for interval,lo in zip(intervals,cheap):
                if lo>incumbent-atol:
                    heapq.heappush(heap,(float(lo),counter,*interval));counter+=1
                else:keep.append(interval)
            intervals=keep
            if not intervals:return
        box=evaluate(intervals);mids=[((a+c)/2,(a+c)/2) for a,c in intervals];mid=evaluate(mids)
        lower=jets.mean_value_lower(box,mid,intervals)
        for (a,c),lo,hi in zip(intervals,lower,mid.v[1]):
            assert not np.isnan(lo+hi),(ed,B,strategy,a,c)
            heapq.heappush(heap,(float(lo),counter,a,c));counter+=1
            if hi<incumbent:incumbent=float(hi);argmin=float((a+c)/2)
    if work.exists():
        saved=json.loads(work.read_text());counter=saved['counter'];incumbent=saved['incumbent'];argmin=saved['argmin']
        heap=[(r[0],r[1],F(r[2]),F(r[3])) for r in saved['heap']];heapq.heapify(heap)
    else:
        endpoint=evaluate([(F(0),F(0)),(F(1),F(1))]).v[1];incumbent=float(endpoint.min());argmin=float(endpoint.argmin())
        add([(F(0),F(1))])
    last_saved=counter
    while counter<max_cells and incumbent-heap[0][0]>atol:
        intervals=[]
        for _ in range(min(24,len(heap))):
            # Do not refine cells already proved above the incumbent
            # tolerance. They remain in the heap and keep covering the
            # domain; only unresolved cells need another entropy walk.
            if not heap or incumbent-heap[0][0]<=atol:break
            lo,_,a,c=heapq.heappop(heap);mid=(a+c)/2;intervals.extend(((a,mid),(mid,c)))
        add(intervals)
        if counter-last_saved>=1000:
            atomic_json(progress_path,dict(b=b,p=p,ED=ed,B=B,strategy=strategy,cells=counter,
                                          lower=float(heap[0][0]),upper=incumbent,argmin=argmin))
            atomic_json(work,dict(**identity,counter=counter,incumbent=incumbent,argmin=argmin,
                                 heap=[[lo,i,str(a),str(c)] for lo,i,a,c in heap]))
            last_saved=counter
    return dict(lower=float(heap[0][0]),upper=incumbent,gap=float(incumbent-heap[0][0]),argmin=argmin,
                cells=counter,converged=incumbent-heap[0][0]<=atol,atol=atol,
                arithmetic='Outward values and derivative intervals; fixed rational allocations; all public syndrome streams; signed OTP costs; whole [0,1].')


def numerical_worst(b,p,ed,B,strategy,extra=()):
    """Fast cut discovery only; a final whole-interval proof is still required."""
    menu=[item['AD'] for item in strategy];weights=np.array([float(F(item['allocation_exact'])) for item in strategy])
    def evaluate(ts):
        state,y=prepare(b,p,ts,ed,B);return (values(state,menu)@weights)*y
    edge=np.logspace(-10,-1,19);ts=np.unique(np.r_[np.linspace(0,1,65),edge,1-edge,list(extra)])
    vv=evaluate(ts);idx=int(vv.argmin());best=float(vv[idx]);arg=float(ts[idx])
    local=np.flatnonzero((vv[1:-1]<=vv[:-2])&(vv[1:-1]<=vv[2:])&((vv[1:-1]<vv[:-2])|(vv[1:-1]<vv[2:])))+1
    if len(local)>10:local=local[np.argsort(vv[local])[:10]]
    if len(local):
        left=ts[local-1];right=ts[local+1];ratio=(np.sqrt(5)-1)/2
        for _ in range(42):
            c=right-ratio*(right-left);d=left+ratio*(right-left)
            cc=evaluate(c);dd=evaluate(d);take=cc<=dd
            left=np.where(take,left,c);right=np.where(take,d,right)
        mid=(left+right)/2;mv=evaluate(mid);i=int(mv.argmin())
        if mv[i]<best:best=float(mv[i]);arg=float(mid[i])
    return best,arg


def solve(b,p,ed,B,nmax=128,tolerance=1e-8,max_iterations=18,incumbent=0.):
    # Pointwise [n-H(X,Z|s)]_+-(n-k) <= [k-H(X,GZ|s)]_+.
    # Removing plain OTP changes no optimum, including fixed mixtures.
    # Independent upper certificates still cover the original full menu.
    menu=[a for a in actions(nmax) if a['mode']!='otp_plain'];ts=[0.,.00390625,.015625,.0625,.25,.5,.75,1.];tmp=ROOT/'tmp'/f'joint_lp_{os.getpid()}';tmp.mkdir(parents=True,exist_ok=True)
    identity=dict(b=b,p=p,ED=ed,B=B,nmax=nmax,objective='fixed_post_IR_code_mixture_v1')
    tag=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    checkpoint=DATA/'bb84_pending_policies'/f'{tag}.json';checkpoint.parent.mkdir(parents=True,exist_ok=True)
    if checkpoint.exists():
        saved=json.loads(checkpoint.read_text());ts=saved['samples']
        if saved.get('status')=='certified_policy' and saved['result'].get('converged'):
            result=saved['result']
            if result['upper_numerical']-result['lower']<=tolerance:return result
    interpreter=os.environ.get('QKD_SCIPY_PYTHON',str(Path.home()/'AppData/Local/Programs/Python/Python38/python.exe'))
    env=os.environ.copy();env.pop('PYTHONHOME',None);env.pop('PYTHONPATH',None);best=None;history=[]
    for iteration in range(max_iterations):
        pp,y=prepare(b,p,ts,ed,B);matrix=values(pp,menu)*y[:,None]
        np.savez(tmp/'input.npz',values=matrix,alphas=np.arange(len(menu)),groups=np.zeros(len(menu),int),betas=np.array([],int),parents=np.array([],int))
        proc=subprocess.run([interpreter,str(ROOT/'fixed_policy_lp.py'),str(tmp/'input.npz'),str(tmp/'output.npz')],capture_output=True,env=env)
        assert proc.returncode==0,proc.stderr.decode(errors='replace')
        data=np.load(tmp/'output.npz');weights=[max(F(0),F(str(x))) for x in data['weights']];total=sum(weights)
        if total>1:weights=[w/total for w in weights]
        strategy=[dict(AD=a,allocation=float(w),allocation_exact=str(w)) for a,w in zip(menu,weights) if w>F(1,10**12)]
        dual=np.maximum(0,data['dual']);dual/=dual.sum();upper=max(0.,float((dual@matrix).max()))+3e-12
        if not strategy or upper<=incumbent+tolerance:
            return dict(lower=0.,upper_numerical=upper,strategy=strategy,ED=ed,classical_B_rounds=B,
                        samples=ts[:],sampled_dual=dual.tolist(),nmax=nmax,converged=True,
                        status='dominated_by_certified_incumbent_in_numerical_screen',
                        note='No new lower witness claimed; the feasible sampled dual is saved for independent outward certification.')
        numerical,t=numerical_worst(b,p,ed,B,strategy,ts)
        atomic_json(checkpoint,dict(**identity,samples=ts,strategy=strategy,numerical_minimum=numerical,numerical_argmin=t,upper_numerical=upper,status='discovery_only'))
        if iteration+1<max_iterations and upper-numerical>tolerance/2 and min(abs(v-t) for v in ts)>1e-11:
            history.append(dict(numerical_minimum=numerical,upper=upper,worst_t=t,stage='floating_cut_discovery'))
            ts.append(t);atomic_json(checkpoint,dict(**identity,samples=ts,status='discovery_only'))
            continue
        print('BB policy certify',b,p,ed,'B',B,'iteration',iteration+1,'actions',strategy,flush=True)
        certificate=certify_policy(b,p,ed,B,strategy,atol=tolerance/4);lower=max(0.,certificate['lower'])
        record=dict(lower=lower,upper_numerical=upper,ED=ed,classical_B_rounds=B,strategy=strategy,
                    certificate=certificate,samples=ts[:],sampled_dual=dual.tolist(),nmax=nmax,
                    converged=upper-lower<=tolerance,iterations=iteration+1)
        if best is None or lower>best['lower']:best=record
        history.append(dict(lower=lower,upper=upper,worst_t=certificate['argmin'],cells=certificate['cells']))
        if record['converged']:
            record['history']=history;atomic_json(checkpoint,dict(**identity,samples=ts,status='certified_policy',result=record));return record
        t=certificate['argmin']
        if min(abs(t-v) for v in ts)<1e-12:break
        ts.append(t)
    best['history']=history;return best


def seeds(b,p,permissions):
    out={f'M{m}':dict(lower=0.,origin='discard') for m in range(1,9)}
    path=DATA/f'bb84_recursive_watanabe_{permissions}_v2_jet_certified.json'
    if path.exists():
        for row in json.loads(path.read_text())['results'].values():
            if row['b']!=b or row['p']!=p:continue
            lower=max(0.,row['certificate']['lower'])
            for m in range(row['budget'],9):
                if lower>out[f'M{m}']['lower']:
                    out[f'M{m}']=dict(lower=lower,origin=path.name,ED=row['ED'],AD=row['AD'],certificate=row['certificate'])
    return out


def search(b,p,permissions='real',nmax=128,K=16,limit=None,tolerance=1e-8):
    ts=np.array([0.,1/64,1/8,1/2,1.]);z=min(b,p)*ts
    initial=np.stack((1-b-p+z,b-z,z,p-z),-1);ps,ys,M,records=generate(initial,7,permissions)
    equivalent=representatives(records,b==p);unique=equivalent==np.arange(len(records))
    path=DATA/f'bb84_joint_extension_{permissions}_{b:.5f}_{p:.5f}_n{nmax}.json'
    if path.exists():
        out=json.loads(path.read_text());assert out['K']==K
        fresh=seeds(b,p,permissions)
        for key,row in fresh.items():
            if row['lower']>out['results'][key]['lower']:out['results'][key]=row
        out['screening']=[]
    else:out=dict(b=b,p=p,permissions=permissions,nmax=nmax,K=K,results=seeds(b,p,permissions),evaluated={},screening=[])
    start=time.time();class_upper=np.zeros(9);screened_complete=True
    for B in range(K+1):
        incumbents=np.array([0.]+[out['results'][f'M{m}']['lower'] for m in range(1,9)])
        er=ys*np.maximum(0,2*ps.max(-1)-1)**2;upper=er.min(-1)+3e-13;argmin=er.argmin(-1)
        ids=np.flatnonzero(unique&(upper>incumbents[M]+tolerance))
        if len(ids):
            local_y=ys[ids,argmin[ids]]
            target=(incumbents[M[ids]]+tolerance)/np.maximum(local_y,1e-300)
            upper[ids]=np.minimum(upper[ids],local_y*known_max(ps[ids,argmin[ids]],nmax,target)+3e-13)
        ids=ids[np.argsort(-upper[ids])];tested=0
        for i in ids:
            key=f'{i}_B{B}'
            if key in out['evaluated']:upper[i]=min(upper[i],out['evaluated'][key]['upper_numerical'])
            if upper[i]<=out['results'][f'M{M[i]}']['lower']+tolerance:continue
            if limit is not None and len(out['evaluated'])>=limit:screened_complete=False;break
            from terminal_equivalence import solved_equivalent
            result=solved_equivalent(out['evaluated'],records[i],B)
            if result is None:result=solve(b,p,records[i],B,nmax,tolerance,incumbent=out['results'][f'M{M[i]}']['lower'])
            out['evaluated'][key]=result;tested+=1
            upper[i]=min(upper[i],result['upper_numerical'])
            for budget in range(int(M[i]),9):
                if result['lower']>out['results'][f'M{budget}']['lower']:out['results'][f'M{budget}']=result.copy()
            atomic_json(path,out);print('BB joint',b,p,'B',B,'node',i,'M',M[i],result['lower'],result['upper_numerical'],'tested',len(out['evaluated']),'seconds',round(time.time()-start,1),flush=True)
        upper=upper[equivalent]
        for budget in range(1,9):class_upper[budget]=max(class_upper[budget],float(upper[M<=budget].max()))
        out['screening'].append(dict(B=B,ER_survivors=len(ids),solved=tested))
        if not screened_complete:break
        ps,s=b_step(ps);ys=ys*s/2
    for budget in range(1,9):
        out['results'][f'M{budget}']['finite_class_upper_numerical']=float(class_upper[budget]) if screened_complete else None
    out['complete_candidate_screen']=screened_complete
    out['scope']='Every depth<=7 word in the four registered real decoders and constant-decoder raw pumps<=160; final axes per permissions; one fixed B<=K and fixed code/mode mixture; all n<=7 codes via fixed direct-sum domination, repetition/SPC<=nmax, Watanabe. Continuous-z policy lower bounds outward; candidate-screen/global upper is floating, not yet an outward optimum certificate. No mixtures of ED circuits or B prefixes.'
    out['seconds']=time.time()-start;atomic_json(path,out);print('BB joint complete',b,p,screened_complete,{k:v['lower'] for k,v in out['results'].items()},flush=True)
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--q',nargs='+',type=float,default=[.2]);parser.add_argument('--p',type=float)
    parser.add_argument('--permissions',choices=['real','full'],default='real');parser.add_argument('--n',type=int,default=128)
    parser.add_argument('--K',type=int,default=16);parser.add_argument('--limit',type=int);parser.add_argument('--tolerance',type=float,default=1e-8)
    a=parser.parse_args()
    for b in a.q:
        p=b if a.p is None else a.p
        path=DATA/f'bb84_joint_extension_{a.permissions}_{b:.5f}_{p:.5f}_n{a.n}.json'
        if path.exists():
            old=json.loads(path.read_text())
            if old.get('complete_candidate_screen') and old.get('K')==a.K:continue
        search(b,p,a.permissions,a.n,a.K,a.limit,a.tolerance)


if __name__=='__main__':main()
