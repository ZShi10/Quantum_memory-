"""Fixed-ED, fixed-policy BB84 minimax, including all Du OTP modes.

The quantum block is fixed before the unknown Bell correlation. Classical
allocations and retention probabilities are fixed too. Numerical sampled
duals give upper estimates; rational-Bernstein/Decimal calculations give
independent continuous-z lower certificates for every selected policy.
"""
from core import *
from ad_modes import OTPKernel,MODES,best_short
from libraries import complete_short_entries
from search_bb84 import raw_polynomials
from bernstein import evaluate
from interval_certificate import IV,LN2,Curve,midpoint,mul,powpoly,mono,matpoly,certify
from fractions import Fraction as F
from decimal import Decimal
from math import comb
import argparse,subprocess,time


def physical_kernel(kernel):
    obj=CodeKernel.__new__(CodeKernel)
    obj.n=kernel.n;obj.k=kernel.n;obj.entry=kernel.entry
    otp=OTPKernel(kernel.entry);types=compositions(kernel.n)
    obj.types=np.asarray(types)
    obj.polys=np.eye(len(types),dtype=int)
    obj.hist=np.array([[int(row[x+y])*comb(x+y,y)*comb(i+z,z)
                        for i,x,y,z in types] for row in otp.counts],dtype=int)
    obj.multiplicity=np.ones(len(obj.hist),int)
    obj.marginal=obj.hist.copy()
    obj.syndrome_classes=[[r] for r in otp.representatives]
    return obj


class Menu:
    def __init__(self,raw,m,kernels):
        self.raw=raw;self.m=m;self.kernels=kernels
        self.physical={k.entry['id']:physical_kernel(k) for k in kernels}
        self.options=[];self.alphas=[];self.betas=[];self.parents=[];self.groups=[];at=0
        for branch in range(len(raw)):
            for k in kernels:
                for mode in MODES:
                    kk=k if mode=='no_otp' else self.physical[k.entry['id']]
                    alpha=at;beta=np.arange(at+1,at+1+len(kk.hist));at+=1+len(kk.hist)
                    self.alphas.append(alpha);self.betas.extend(beta);self.parents.extend([alpha]*len(beta));self.groups.append(branch)
                    self.options.append(dict(branch=branch,kernel=k,entropy_kernel=kk,mode=mode,alpha=alpha,beta=beta))
        self.count=at
        for name in ('alphas','betas','parents','groups'):setattr(self,name,np.array(getattr(self,name),int))

    def values(self,ts):
        matrix=np.zeros((len(ts),self.count))
        for branch,raw in enumerate(self.raw):
            v=np.stack([evaluate(raw,t) for t in ts])
            ps=v.sum(-1)
            p=np.divide(v,ps[:,None],out=np.full_like(v,.25),where=ps[:,None]>0)
            powers={n:monomials(p,compositions(n)) for n in {k.n for k in self.kernels}}
            for option in self.options:
                if option['branch']!=branch:continue
                k=option['kernel'];kk=option['entropy_kernel'];mode=option['mode']
                prob,hmass=kk.components_from_monomials(powers[k.n])
                gain=(kk.k*prob-hmass)*kk.multiplicity/k.n
                matrix[:,option['beta']]=gain*ps[:,None]/self.m
                if mode=='otp_plain':cost=np.full(len(ts),(k.n-k.k)/k.n)
                elif mode=='otp_compressed':cost=-(xlog2x(prob)@kk.multiplicity)/k.n
                else:cost=np.zeros(len(ts))
                matrix[:,option['alpha']]=-cost*ps/self.m
        return matrix

    def support(self,values,dual):
        average=dual@values;best=np.zeros(len(self.raw))
        for o in self.options:
            value=average[o['alpha']]+np.maximum(average[o['beta']],0).sum()
            best[o['branch']]=max(best[o['branch']],value)
        return float(best.sum())


class FixedCurve(Curve):
    def __init__(self,raw,option,alpha,beta,m,lift_n=None):
        kernel=option['entropy_kernel'];self.mode=option['mode']
        self.alpha=IV(F(str(alpha)));self.beta=[IV(F(str(v))) for v in beta]
        self.dimension=kernel.k;self.cost_dimension=option['kernel'].n-option['kernel'].k
        self.hist=kernel.hist.astype(int).tolist();self.mult=list(map(int,kernel.multiplicity))
        joint=matpoly(kernel.polys,mono(raw,kernel.types))
        ps=[sum(v) for v in zip(*raw)]
        lift_n=kernel.n if lift_n is None else lift_n
        assert lift_n>=kernel.n
        if lift_n>kernel.n:
            factor=powpoly(ps,lift_n-kernel.n)
            joint=[mul(row,factor) for row in joint]
        self.joint=[[IV(v) for v in row] for row in joint]
        self.den=[IV(v) for v in mul(powpoly(ps,lift_n-1),[F(1)]*len(ps))]
        self.scale=IV(F(1,m*kernel.n))

    def tangent(self,joint,den):
        mid=[midpoint(v) for v in joint]
        logs=[v.ln()/LN2 if v.lo>0 else None for v in mid]
        probs=[sum((c*mid[j] for j,c in enumerate(row) if c),IV(0)) for row in self.hist]
        total=sum((mult*v for mult,v in zip(self.mult,probs)),IV(0))
        total_log=total.ln()/LN2 if total.lo>0 else IV(0)
        grad=[IV(0) for _ in mid]
        for row,mult,prob,beta in zip(self.hist,self.mult,probs,self.beta):
            if prob.hi==0:continue
            lp=prob.ln()/LN2
            for j,c in enumerate(row):
                if not c or logs[j] is None:continue
                value=beta*(IV(self.dimension)+logs[j]-lp)
                if self.mode=='otp_plain':value=value-self.alpha*self.cost_dimension
                elif self.mode=='otp_compressed':value=value+self.alpha*(lp-total_log)
                grad[j]=grad[j]+mult*c*value
        value=sum((g*v for g,v in zip(grad,mid)),IV(0))*self.scale
        coeff=[sum((g*row[i] for g,row in zip(grad,joint)),IV(0))*self.scale for i in range(len(den))]
        return coeff,value

    def bound(self,joint,den):
        coeff,value=self.tangent(joint,den)
        for a,b in zip(coeff,den):
            if b.lo<=0:assert b.lo==b.hi==a.lo==a.hi==0
        lower=min((a/b).lo for a,b in zip(coeff,den) if b.lo>0)
        return lower,value/midpoint(den)


class GroupCurve(Curve):
    """Sum policy tangents before taking the Bernstein minimum.

    All members share one ED branch and have been lifted to its largest
    AD block length. Their common positive denominator is therefore exact.
    This retains cancellations between mixed policies near a minimizer.
    """
    def __init__(self,curves):
        self.curves=curves;self.den=curves[0].den
        self.sizes=[len(c.joint) for c in curves]
        self.joint=[row for c in curves for row in c.joint]
        for c in curves:
            assert [(v.lo,v.hi) for v in c.den]==[(v.lo,v.hi) for v in self.den]

    def bound(self,joint,den):
        coeff=[IV(0) for _ in den];value=IV(0);at=0
        for c,n in zip(self.curves,self.sizes):
            term,middle=c.tangent(joint[at:at+n],den);at+=n
            coeff=[a+b for a,b in zip(coeff,term)];value=value+middle
        for a,b in zip(coeff,den):
            if b.lo<=0:assert b.lo==b.hi==a.lo==a.hi==0
        lower=min((a/b).lo for a,b in zip(coeff,den) if b.lo>0)
        return lower,value/midpoint(den)


def exact_strategy(menu,weights,raw,m):
    """Repair feasibility in rational arithmetic, then construct a certificate."""
    weights=[max(F(0),F(str(v))) for v in weights]
    for branch in range(len(raw)):
        opts=[o for o in menu.options if o['branch']==branch]
        total=sum(weights[o['alpha']] for o in opts)
        if total>1:
            for o in opts:
                for index in [o['alpha'],*o['beta']]:weights[index]/=total
        for o in opts:
            for index in o['beta']:weights[index]=min(weights[index],weights[o['alpha']])
    active=[o for o in menu.options if weights[o['alpha']]>F(1,10**11)
            and (o['mode']!='no_otp' or any(weights[i]>F(1,10**11) for i in o['beta']))]
    curves=[];strategy=[]
    for branch in sorted({o['branch'] for o in active}):
        opts=[o for o in active if o['branch']==branch]
        largest=max(o['kernel'].n for o in opts);members=[]
        for o in opts:
            alpha=weights[o['alpha']];beta=[weights[i] for i in o['beta']]
            members.append(FixedCurve(raw[branch],o,alpha,beta,m,largest))
            strategy.append(dict(branch=branch,code=o['kernel'].entry['id'],mode=o['mode'],
                                 allocation=float(alpha),allocation_exact=str(alpha),
                                 retained_allocations=list(map(float,beta)),
                                 retained_allocations_exact=list(map(str,beta)),
                                 syndrome_classes=o['entropy_kernel'].syndrome_classes))
        curves.append(GroupCurve(members))
    return curves,strategy


def exact_raw(entry,b,p):
    b,p=F(str(b)),F(str(p));z=min(b,p)
    components=[[1-b-p,1-b-p+z],[b,b-z],[F(0),z],[p,p-z]]
    physical=mono(components,compositions(entry['m']))
    return [matpoly(branch,physical) for branch in entry['tensor']]


def solve(entry,b,p,kernels,tolerance=2e-8,max_iterations=15):
    raw=raw_polynomials(entry,b,p);menu=Menu(raw,entry['m'],kernels)
    exact=exact_raw(entry,b,p)
    tmp=ROOT/'tmp'/f'lp_{os.getpid()}'
    tmp.mkdir(parents=True,exist_ok=True)
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
        dual=np.maximum(0,data['dual']);dual/=dual.sum()
        upper=menu.support(values,dual)+3e-11
        for group in range(len(raw)):
            option_ids=[i for i,o in enumerate(menu.options) if o['branch']==group]
            total=sum(weights[menu.options[i]['alpha']] for i in option_ids)
            for i in option_ids:
                o=menu.options[i]
                if total>1:weights[o['alpha']]/=total;weights[o['beta']]/=total
                weights[o['beta']]=np.minimum(weights[o['beta']],weights[o['alpha']])
        curves,strategy=exact_strategy(menu,weights,exact,entry['m'])
        if not curves:
            return dict(lower='0',upper_numerical=upper,strategy=[],converged=upper<tolerance,ED=entry['id'])
        certificate=certify(curves,atol=str(tolerance/5),max_cells=18000)
        lower=max(0.,float(certificate['lower']))
        record=dict(lower=str(max(Decimal(0),Decimal(certificate['lower']))),upper_numerical=upper,
                    certificate=certificate,strategy=strategy,ED=entry['id'],m=entry['m'],M=entry['M'],
                    converged=upper-lower<=tolerance,iterations=iteration+1,samples=ts[:],variables=menu.count,
                    sampled_dual=dual.tolist(),
                    upper_status='Sample dual evaluated in floating arithmetic; outward whole-class certificate pending.')
        if best is None or lower>float(best['lower']):best=record
        history.append(dict(lower=lower,upper=upper,argmin=certificate['argmin'],cells=certificate['cells']))
        if record['converged']:record['history']=history;return record
        t=certificate['argmin']
        if min(abs(v-t) for v in ts)<1e-10:break
        ts.append(t)
    best['history']=history;return best


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--b',type=float,default=.2)
    parser.add_argument('--p',type=float);parser.add_argument('--n',type=int,default=7)
    parser.add_argument('--permissions',choices=['real','full'],default='real')
    parser.add_argument('--entry');parser.add_argument('--limit',type=int)
    parser.add_argument('--tolerance',type=float,default=2e-8);args=parser.parse_args()
    p=args.b if args.p is None else args.p
    entries=[e for e in complete_short_entries(args.b==p) if args.permissions=='full' or e['real']]
    kernels=load_codes(args.n,all_codes=True)
    if args.entry:entries=[e for e in entries if e['id']==args.entry]
    ts=[0.,.25,.5,.75,1.];raws=[];states=[];meta=[]
    for i,e in enumerate(entries):
        raw=raw_polynomials(e,args.b,p)
        for branch in raw:
            v=np.stack([evaluate(branch,t) for t in ts]);mass=v.sum(-1)
            state=np.divide(v,mass[:,None],out=np.full_like(v,.25),where=mass[:,None]>0)
            states.append(state);meta.append((i,mass/e['m']))
    state=np.array(states).reshape(-1,4)
    rates=best_short(state,[k for k in kernels if k.entry['indecomposable']])['rate'].reshape(-1,len(ts))
    upper=np.zeros((len(entries),len(ts)))
    for (i,mass),r in zip(meta,rates):upper[i]+=mass*r
    upper=upper.min(-1)+3e-11
    output=dict(b=args.b,p=p,permissions=args.permissions,nmax=args.n,entry_count=len(entries),results={},evaluated={})
    path=DATA/f'bb84_fixed_{args.permissions}_{args.b:.5f}_{p:.5f}_n{args.n}.json'
    start=time.time()
    for i in np.argsort(-upper):
        e=entries[i]
        incumbent=min((float(output['results'].get(f'M{M}',{}).get('lower',0)) for M in range(e['M'],3)))
        if upper[i]<=incumbent+args.tolerance:continue
        if args.limit is not None and len(output['evaluated'])>=args.limit:break
        result=solve(e,args.b,p,kernels,args.tolerance)
        output['evaluated'][e['id']]=result
        for M in range(e['M'],3):
            if float(result['lower'])>float(output['results'].get(f'M{M}',{}).get('lower',0)):
                output['results'][f'M{M}']=result
        atomic_json(path,output)
        print('BB84 fixed',args.b,p,e['id'],result['lower'],result['upper_numerical'],'tested',len(output['evaluated']),'seconds',round(time.time()-start,1),flush=True)
    for M in range(1,3):
        bounds=[min(upper[i],output['evaluated'].get(e['id'],{}).get('upper_numerical',np.inf)) for i,e in enumerate(entries) if e['M']<=M]
        row=output['results'].setdefault(f'M{M}',dict(lower='0',strategy=[]))
        row['finite_menu_upper_numerical']=float(max(bounds));row['finite_menu_gap_numerical']=float(max(bounds))-float(row['lower'])
    output['scope']='Fixed m<=4 terminal ED; all n<=N code classes; no-OTP/plain/compressed OTP; fixed allocations and randomized flag retention; no pilot; no B prefix. Continuous-z lower certificates. Sample/global upper arithmetic not yet outward certified.'
    atomic_json(path,output)


if __name__=='__main__':main()
