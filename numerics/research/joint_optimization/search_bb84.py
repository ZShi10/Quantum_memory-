"""Untwirled BB84 minimax search with fixed public-branch processing.

Classical time-sharing of short codes is selected BEFORE the unknown z.
Retained AD syndrome classes are also fixed. Privacy amplification is joint
over the retained public classes; measured bit statistics set reconciliation
lengths, not the quantum circuit. See SECURITY_SCOPE.md.

Finite input samples give optimization UPPER bounds. Bernstein entropy
enclosures give continuous-input LOWER bounds for each returned strategy.
"""
from pathlib import Path
from collections import defaultdict
from copy import copy
import argparse,json,time,subprocess,sys,os
from itertools import permutations
import numpy as np
from algebra import compositions,code_components
from rates import CodeKernel,best_codes,monomials
from bernstein import monomial_coefficients,evaluate,GainCurve,gain_coefficients,split,certify_minimum

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data';TMP=ROOT/'tmp'/f'process_{os.getpid()}'


def load_libraries(symmetric=True):
    codedata=json.loads((DATA/'codes.json').read_text())['entries']
    kernels=[CodeKernel(e) for e in codedata]
    upperkernels=[k for k in kernels if len(code_components(k.n,k.entry['rows']))==1]
    entries=json.loads((DATA/'blocks.json').read_text())['entries']
    seen={}
    bit_partition=({0,3},{1,2})
    frames=[p for p in permutations(range(4)) if set(p[i] for i in bit_partition[0]) in bit_partition]
    for e in entries:
        if symmetric:
            m=e['m'];types=compositions(m)
            pairs=sorted({(v[0],v[2]) for v in types});lookup={v:i for i,v in enumerate(pairs)}
            raw=np.asarray(e['tensor']);collapsed=np.zeros(raw.shape[:2]+(len(pairs),),dtype=int)
            for j,typ in enumerate(types):collapsed[:,:,lookup[typ[0],typ[2]]]+=raw[:,:,j]
        else:collapsed=np.asarray(e['tensor'])
        # Pauli frames and z -> z+x leave the full-bit/logical-phase entropy
        # invariant, with a fixed permutation of the AD syndrome. All AD
        # retention sets are allowed, so the entire fixed-strategy menu is
        # preserved. The representative is evaluated in its ORIGINAL frame.
        branches=[min(row[list(p)].tobytes() for p in frames) for row in collapsed]
        sig=b''.join(sorted(branches))
        key=(e['m'],e['real'],sig)
        if key not in seen or (e['M'],len(e['gates']))<(seen[key]['M'],len(seen[key]['gates'])):seen[key]=e
    entries=list(seen.values())
    return entries,kernels,upperkernels


def raw_polynomials(entry,eb,ep):
    zmax=min(eb,ep);s=eb+ep
    ends=np.array([[1-s,eb,0.,ep],[1-s+zmax,eb-zmax,zmax,ep-zmax]])
    mono=monomial_coefficients(ends.T,np.array(compositions(entry['m'])))
    return np.asarray(entry['tensor'])@mono


def anchor_upper_bounds(entries,eb,ep,kernels,ts=(0.,.25,.5,.75,1.)):
    values=[];weights=[];indices=[];raws=[]
    for i,e in enumerate(entries):
        raw=raw_polynomials(e,eb,ep);raws.append(raw)
        v=np.stack([evaluate(raw,t) for t in ts],axis=1) # branch,sample,logical
        ps=v.sum(-1);p=np.divide(v,ps[...,None],out=np.full_like(v,.25),where=ps[...,None]>0)
        for s in range(len(raw)):
            values.append(p[s]);weights.append(ps[s]/e['m']);indices.append(i)
    best=best_codes(np.asarray(values).reshape(-1,4),kernels)['rate'].reshape(len(values),len(ts))
    total=np.zeros((len(entries),len(ts)))
    for i,w,v in zip(indices,weights,best):total[i]+=w*v
    # Upper arithmetic pad is tracked independently of the lower enclosure.
    return total.min(1)+2e-11,raws


def possible_positive_classes(base,subdivisions=3):
    """Safe entropy-convexity pruning; never infer negativity from a grid."""
    parts=[base.joint]
    for _ in range(subdivisions):parts=[s for part in parts for s in split(part)]
    upper=np.full(len(base.kernel.hist),-np.inf)
    for joint in parts:
        _,hi=gain_coefficients(joint,base.kernel.hist,base.kernel.k)
        ps=base.kernel.hist@joint
        relative=np.divide(hi,ps,out=np.zeros_like(hi),where=ps>0)
        upper=np.maximum(upper,relative.max(1))
    return np.flatnonzero(upper>1e-12)


class Options:
    def __init__(self,raw,m,kernels):
        self.curves=[];self.groups=[];self.metadata=[];self.by_kernel=[]
        for s,branch in enumerate(raw):
            self.curves.append(None);self.groups.append(s);self.metadata.append(dict(syndrome=s,discard=True))
            # A mixture of normalized Bernstein vectors contains the full
            # state curve. If all these Bell vectors are separable, discard.
            ps=branch.sum(0);active=ps>0
            if not active.any() or np.max(branch[:,active]/ps[active])<=.5+1e-14:continue
            for kernel in kernels:
                base=GainCurve(branch,kernel,scale=1/m)
                keep=possible_positive_classes(base)
                if not len(keep):continue
                masks=[];indices=[]
                for bits in range(1,1<<len(keep)):
                    mask=np.zeros(len(kernel.hist))
                    mask[keep]=[(bits>>j)&1 for j in range(len(keep))]
                    c=copy(base);c.mask=mask
                    indices.append(len(self.curves));masks.append(mask)
                    self.curves.append(c);self.groups.append(s)
                    self.metadata.append(dict(syndrome=s,code=kernel.entry['id'],keep_classes=np.flatnonzero(mask).tolist(),
                                              keep_syndrome_representatives=[r for j in np.flatnonzero(mask) for r in kernel.syndrome_classes[j]]))
                self.by_kernel.append((base,np.array(masks),np.array(indices)))
        self.groups=np.array(self.groups)

    def values(self,ts):
        result=np.zeros((len(ts),len(self.curves)))
        for base,masks,idx in self.by_kernel:
            for j,t in enumerate(ts):
                joint=evaluate(base.joint,t);ps=base.kernel.hist@joint
                from rates import xlog2x
                gain=base.kernel.k*ps+base.kernel.hist@xlog2x(joint)-xlog2x(ps)
                den=evaluate(base.denominator,t)
                if den>0:result[j,idx]=base.scale*(masks@(base.kernel.multiplicity*gain))/den
        return result


def solve_options(options,tolerance=2e-8,max_iterations=20,cutoff=None):
    TMP.mkdir(parents=True,exist_ok=True)
    ts=[0.,.25,.5,.75,1.];history=[];best=None
    for iteration in range(max_iterations):
        values=options.values(ts)
        np.savez(TMP/'lp_input.npz',values=values,groups=options.groups)
        interpreter=os.environ.get('QKD_SCIPY_PYTHON',str(Path.home()/'AppData/Local/Programs/Python/Python38/python.exe'))
        environment=os.environ.copy()
        environment.pop('PYTHONHOME',None);environment.pop('PYTHONPATH',None)
        completed=subprocess.run([interpreter,str(ROOT/'lp_worker.py'),str(TMP/'lp_input.npz'),str(TMP/'lp_output.npz')],capture_output=True,text=True,env=environment)
        if completed.returncode:raise RuntimeError(completed.stderr)
        solution=np.load(TMP/'lp_output.npz');weights=np.maximum(solution['weights'],0.)
        for group in np.unique(options.groups):
            idx=options.groups==group;weights[idx]/=weights[idx].sum()
        dual=np.maximum(solution['dual'],0.);dual/=dual.sum()
        averaged=dual@values
        # Reconstruct a valid finite-sample dual bound without relying on
        # solver-reported optimality or the exact normalization of its dual.
        upper=sum(float(averaged[options.groups==group].max()) for group in np.unique(options.groups))
        if cutoff is not None and max(0.,upper)+2e-11<=cutoff+tolerance:
            return dict(lower=0.,upper=max(0.,upper)+2e-11,strategy=[],converged=False,
                        pruned_by_incumbent=True,iterations=iteration+1)
        curves=[];strategy=[]
        for j,w in enumerate(weights):
            if w<=1e-10 or options.curves[j] is None:continue
            curve=copy(options.curves[j]);curve.scale*=w;curves.append(curve)
            strategy.append({**options.metadata[j],'input_fraction':float(w)})
        if not curves:
            return dict(lower=0.,upper=max(0.,upper)+2e-11,strategy=[],converged=upper<tolerance,iterations=iteration+1)
        certificate=certify_minimum(curves,atol=tolerance/8,rtol=1e-8,max_cells=12000)
        lower=max(0.,certificate['lower'])
        record=dict(lower=lower,upper=max(0.,upper)+2e-11,strategy=strategy,certificate=certificate,
                    converged=max(0.,upper)-lower<=tolerance,iterations=iteration+1,options=len(weights))
        if best is None or record['lower']>best['lower']:best=record
        history.append(dict(upper=upper,lower=lower,argmin=certificate['argmin'],cells=certificate['cells']))
        if record['converged']:record['history']=history;return record
        next_t=certificate['argmin']
        if min(abs(t-next_t) for t in ts)<1e-11:
            record['history']=history;return record
        ts.append(next_t)
    best['history']=history;return best


def search_point(entries,kernels,upperkernels,eb,ep,max_entries=None,tolerance=2e-8):
    start=time.time();upper,raws=anchor_upper_bounds(entries,eb,ep,upperkernels)
    best={(mode,M):dict(lower=0.,ED=None) for mode in ('real','full') for M in range(1,5)}
    evaluated={};order=np.argsort(-upper)
    for bi in order:
        e=entries[bi]
        targets=[key for key in best if e['M']<=key[1] and (key[0]=='full' or e['real'])]
        if all(upper[bi]<=best[key]['lower']+tolerance for key in targets):continue
        if max_entries is not None and len(evaluated)>=max_entries:break
        result=solve_options(Options(raws[bi],e['m'],kernels),tolerance=tolerance,
                             cutoff=min(best[key]['lower'] for key in targets))
        evaluated[int(bi)]=result
        for key in targets:
            if result['lower']>best[key]['lower']:
                best[key]={**result,'ED':e['id'],'m':e['m'],'M':e['M']}
        print('BB84',eb,ep,'ED',e['id'],'M',e['M'],'bounds',result['lower'],result['upper'],'options',result.get('options',0),'tested',len(evaluated),'seconds',round(time.time()-start,1),flush=True)
    for key,record in best.items():
        eligible=[i for i,e in enumerate(entries) if e['M']<=key[1] and (key[0]=='full' or e['real'])]
        globalupper=max((min(upper[i],evaluated[i]['upper']) if i in evaluated else upper[i]) for i in eligible)
        record['finite_class_upper']=float(globalupper)
        record['gap']=float(globalupper-record['lower'])
        record['certified_to_tolerance']=record['gap']<=tolerance
    return dict(eb=eb,ep=ep,tolerance=tolerance,entries=len(entries),evaluated=len(evaluated),
                seconds=time.time()-start,results={f'{k[0]}_M{k[1]}':v for k,v in best.items()})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--eb',type=float,default=.20)
    parser.add_argument('--ep',type=float);parser.add_argument('--limit',type=int)
    parser.add_argument('--tolerance',type=float,default=2e-8)
    args=parser.parse_args();ep=args.eb if args.ep is None else args.ep
    entries,kernels,upperkernels=load_libraries(args.eb==ep)
    print('BB84 library',len(entries),'instruments;',len(kernels),'code orbits.',flush=True)
    result=search_point(entries,kernels,upperkernels,args.eb,ep,args.limit,args.tolerance)
    result['resource_compiler']='terminal-graph-v1'
    result['scope']='m<=4 compiled single Clifford block; n<=6 single-round linear codes; fixed syndrome retention; fixed classical input fractions. Continuous z minimization with numerical Bernstein enclosures.'
    path=DATA/f'bb84_{args.eb:.5f}_{ep:.5f}.json'
    path.write_text(json.dumps(result,indent=2,default=lambda v:v.item()))
    print('Saved',path.name,'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
