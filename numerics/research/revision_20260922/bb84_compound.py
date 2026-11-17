"""Fixed ED/code allocations with a streamwise post-IR entropy bound.

Every public syndrome is reconciled. Positive parts bound the entropy of
the same classical register after IR; they are NOT hidden-z decisions to
retain data. OTP net coefficients remain signed. See COMPOUND_POST_IR_BOUND.
This is a separate objective from bb84_fixed.py; preserve its certificates.
"""
from bb84_fixed import (physical_kernel, FixedCurve, GroupCurve, exact_raw,
                       raw_polynomials, complete_short_entries, evaluate,
                       IV, LN2, midpoint, certify, F, Decimal)
from core import *
from ad_modes import MODES,best_short
import argparse,subprocess,time


class CompoundMenu:
    def __init__(self,raw,m,kernels):
        self.raw,self.m,self.kernels=raw,m,kernels
        self.physical={k.entry['id']:physical_kernel(k) for k in kernels}
        self.options=[]
        for branch in range(len(raw)):
            for k in kernels:
                for mode in MODES:
                    kk=k if mode=='no_otp' else self.physical[k.entry['id']]
                    self.options.append(dict(branch=branch,kernel=k,entropy_kernel=kk,
                                             mode=mode,alpha=len(self.options)))
        self.count=len(self.options)
        self.alphas=np.arange(self.count);self.betas=np.array([],int);self.parents=self.betas
        self.groups=np.array([o['branch'] for o in self.options],int)

    def values(self,ts):
        matrix=np.zeros((len(ts),self.count))
        for branch,raw in enumerate(self.raw):
            v=np.stack([evaluate(raw,t) for t in ts]);ps=v.sum(-1)
            p=np.divide(v,ps[:,None],out=np.full_like(v,.25),where=ps[:,None]>0)
            powers={n:monomials(p,compositions(n)) for n in {k.n for k in self.kernels}}
            for o in self.options:
                if o['branch']!=branch:continue
                k,kk,mode=o['kernel'],o['entropy_kernel'],o['mode']
                prob,hmass=kk.components_from_monomials(powers[k.n])
                gross=(np.maximum(kk.k*prob-hmass,0)*kk.multiplicity).sum(-1)/k.n
                if mode=='otp_plain':cost=(k.n-k.k)/k.n
                elif mode=='otp_compressed':cost=-(xlog2x(prob)@kk.multiplicity)/k.n
                else:cost=0.
                matrix[:,o['alpha']]=(gross-cost)*ps/self.m
        return matrix

    def support(self,values,dual):
        avg=dual@values
        return sum(max(0.,float(avg[self.groups==j].max())) for j in range(len(self.raw)))


class PostIRCurve(FixedCurve):
    """Convex homogeneous per-stream entropy bound, then signed OTP cost.

    Select a supporting gradient at a cell midpoint. The selection is part
    of a mathematical lower certificate, never a physical retention flag.
    The returned midpoint interval evaluates ALL positive parts exactly.
    """
    def __init__(self,raw,option,alpha,m,lift_n=None):
        super().__init__(raw,option,alpha,[],m,lift_n)

    def tangent(self,joint,den):
        mid=[midpoint(v) for v in joint]
        logs=[v.ln()/LN2 if v.lo>0 else None for v in mid]
        probs=[sum((c*mid[j] for j,c in enumerate(row) if c),IV(0)) for row in self.hist]
        total=sum((mult*v for mult,v in zip(self.mult,probs)),IV(0))
        total_log=total.ln()/LN2 if total.lo>0 else IV(0)
        grad=[IV(0) for _ in mid];true_value=IV(0)
        for row,mult,prob in zip(self.hist,self.mult,probs):
            if prob.hi==0:continue
            lp=prob.ln()/LN2
            signed=prob*self.dimension+sum((c*mid[j]*logs[j] for j,c in enumerate(row)
                                           if c and logs[j] is not None),IV(0))-prob*lp
            active=signed.lo>0
            gain=IV(max(Decimal(0),signed.lo),max(Decimal(0),signed.hi))
            if self.mode=='otp_plain':gain=gain-prob*self.cost_dimension
            elif self.mode=='otp_compressed':gain=gain+prob*(lp-total_log)
            true_value=true_value+self.alpha*mult*gain
            for j,c in enumerate(row):
                if not c or logs[j] is None:continue
                value=IV(self.dimension)+logs[j]-lp if active else IV(0)
                if self.mode=='otp_plain':value=value-self.cost_dimension
                elif self.mode=='otp_compressed':value=value+lp-total_log
                grad[j]=grad[j]+self.alpha*mult*c*value
        coeff=[sum((g*row[i] for g,row in zip(grad,joint)),IV(0))*self.scale
               for i in range(len(den))]
        return coeff,true_value*self.scale


def exact_strategy(menu,weights,raw,m):
    weights=[max(F(0),F(str(v))) for v in weights]
    for branch in range(len(raw)):
        ids=[o['alpha'] for o in menu.options if o['branch']==branch]
        total=sum(weights[i] for i in ids)
        if total>1:
            for i in ids:weights[i]/=total
    active=[o for o in menu.options if weights[o['alpha']]>F(1,10**12)]
    curves=[];strategy=[]
    for branch in sorted({o['branch'] for o in active}):
        opts=[o for o in active if o['branch']==branch];largest=max(o['kernel'].n for o in opts)
        members=[]
        for o in opts:
            alpha=weights[o['alpha']]
            members.append(PostIRCurve(raw[branch],o,alpha,m,largest))
            strategy.append(dict(branch=branch,code=o['kernel'].entry['id'],mode=o['mode'],
                                 allocation=float(alpha),allocation_exact=str(alpha),
                                 syndrome_rule='Reconcile every public syndrome; one joint privacy-amplification hash.'))
        curves.append(GroupCurve(members))
    return curves,strategy


def solve(entry,b,p,kernels,tolerance=2e-8,max_iterations=20):
    raw=raw_polynomials(entry,b,p);menu=CompoundMenu(raw,entry['m'],kernels)
    exact=exact_raw(entry,b,p);tmp=ROOT/'tmp'/f'compound_lp_{os.getpid()}';tmp.mkdir(parents=True,exist_ok=True)
    ts=[0.,.125,.25,.5,.75,.875,1.];history=[]
    env=os.environ.copy();env.pop('PYTHONHOME',None);env.pop('PYTHONPATH',None)
    interpreter=os.environ.get('QKD_SCIPY_PYTHON',str(Path.home()/'AppData/Local/Programs/Python/Python38/python.exe'))
    best=None
    for iteration in range(max_iterations):
        values=menu.values(ts)
        np.savez(tmp/'input.npz',values=values,alphas=menu.alphas,betas=menu.betas,parents=menu.parents,groups=menu.groups)
        proc=subprocess.run([interpreter,str(ROOT/'fixed_policy_lp.py'),str(tmp/'input.npz'),str(tmp/'output.npz')],capture_output=True,env=env)
        assert proc.returncode==0,proc.stderr.decode(errors='replace')
        data=np.load(tmp/'output.npz');weights=np.maximum(0,data['weights'])
        dual=np.maximum(0,data['dual']);dual/=dual.sum();upper=menu.support(values,dual)+3e-11
        curves,strategy=exact_strategy(menu,weights,exact,entry['m'])
        if not curves:return dict(lower='0',upper_numerical=upper,strategy=[],converged=upper<tolerance,ED=entry['id'],samples=ts,sampled_dual=dual.tolist())
        certificate=certify(curves,atol=str(tolerance/5),max_cells=18000)
        lower=max(0.,float(certificate['lower']))
        record=dict(lower=str(max(Decimal(0),Decimal(certificate['lower']))),upper_numerical=upper,
                    certificate=certificate,strategy=strategy,ED=entry['id'],m=entry['m'],M=entry['M'],
                    converged=upper-lower<=tolerance,iterations=iteration+1,samples=ts[:],variables=menu.count,
                    sampled_dual=dual.tolist(),upper_status='Floating feasible-dual evaluation; outward certification is separate.')
        if best is None or lower>float(best['lower']):best=record
        history.append(dict(lower=lower,upper=upper,argmin=certificate['argmin'],cells=certificate['cells']))
        if record['converged']:record['history']=history;return record
        t=certificate['argmin']
        if min(abs(v-t) for v in ts)<1e-11:break
        ts.append(t)
    best['history']=history;return best


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--b',type=float,default=.15);parser.add_argument('--p',type=float)
    parser.add_argument('--n',type=int,default=7);parser.add_argument('--permissions',choices=['real','full'],default='real')
    parser.add_argument('--entry');parser.add_argument('--limit',type=int);parser.add_argument('--tolerance',type=float,default=2e-8)
    a=parser.parse_args();p=a.b if a.p is None else a.p
    entries=[e for e in complete_short_entries(a.b==p) if a.permissions=='full' or e['real']]
    if a.entry:entries=[e for e in entries if e['id']==a.entry]
    kernels=load_codes(a.n);ts=[0.,.25,.5,.75,1.];states=[];meta=[]
    for i,e in enumerate(entries):
        for branch in raw_polynomials(e,a.b,p):
            v=np.stack([evaluate(branch,t) for t in ts]);mass=v.sum(-1)
            state=np.divide(v,mass[:,None],out=np.full_like(v,.25),where=mass[:,None]>0)
            states.append(state);meta.append((i,mass/e['m']))
    rates=best_short(np.array(states).reshape(-1,4),kernels)['rate'].reshape(-1,len(ts))
    upper=np.zeros((len(entries),len(ts)))
    for (i,mass),r in zip(meta,rates):upper[i]+=mass*r
    upper=upper.min(-1)+3e-11
    path=DATA/f'bb84_compound_{a.permissions}_{a.b:.5f}_{p:.5f}_n{a.n}.json'
    if a.entry:path=path.with_stem(path.stem+'_'+a.entry)
    out=dict(b=a.b,p=p,permissions=a.permissions,nmax=a.n,entry_count=len(entries),results={},evaluated={})
    start=time.time()
    for i in np.argsort(-upper):
        e=entries[i]
        incumbent=min(float(out['results'].get(f'M{M}',{}).get('lower',0)) for M in range(e['M'],3))
        if upper[i]<=incumbent+a.tolerance:continue
        if a.limit is not None and len(out['evaluated'])>=a.limit:break
        result=solve(e,a.b,p,kernels,a.tolerance);out['evaluated'][e['id']]=result
        for M in range(e['M'],3):
            if float(result['lower'])>float(out['results'].get(f'M{M}',{}).get('lower',0)):out['results'][f'M{M}']=result.copy()
        atomic_json(path,out)
        print('BB compound',a.b,p,e['id'],result['lower'],result['upper_numerical'],'tested',len(out['evaluated']),'seconds',round(time.time()-start,1),flush=True)
    for M in (1,2):
        bounds=[min(upper[i],out['evaluated'].get(e['id'],{}).get('upper_numerical',np.inf)) for i,e in enumerate(entries) if e['M']<=M]
        row=out['results'].setdefault(f'M{M}',dict(lower='0',strategy=[]))
        row['finite_menu_upper_numerical']=float(max(bounds,default=0));row['finite_menu_gap_numerical']=row['finite_menu_upper_numerical']-float(row['lower'])
    out['scope']='One fixed m<=4 terminal ED; fixed allocations among n<=N codes and three Du modes; every syndrome IR, joint PA, signed OTP net; indecomposables suffice by pointwise fixed-mixture domination. No pilot, B prefix or Watanabe action in this version. Whole-z lower; menu upper floating pending separate outward certificate.'
    out['complete_candidate_screen']=a.limit is None or len(out['evaluated'])<a.limit
    atomic_json(path,out)


if __name__=='__main__':main()
