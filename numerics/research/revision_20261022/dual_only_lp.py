"""Find a finite legal dual measure without computing a new lower certificate.

The floating LP is only a proposal. The caller MUST evaluate its measure
against every classical coefficient with compound_menu_bounds.dual_upper.
No feasible key rate is returned by this module.
"""
from pathlib import Path
import json,os,subprocess,sys
import numpy as np

def solve(engine,b,p,ed,B,nmax=128,tolerance=2e-9,max_iterations=32,incumbent=0.):
    from fractions import Fraction as F
    from compound_terminals import actions,values
    from bb84_recursive import prepare
    import core
    menu=[a for a in actions(nmax) if a['mode']!='otp_plain']
    ts=[0.,.00390625,.015625,.0625,.25,.5,.75,1.]
    tmp=core.ROOT/'tmp'/f'dual_lp_{os.getpid()}';tmp.mkdir(parents=True,exist_ok=True)
    interpreter=os.environ.get('QKD_SCIPY_PYTHON',str(Path.home()/'AppData/Local/Programs/Python/Python38/python.exe'))
    env=os.environ.copy();env.pop('PYTHONHOME',None);env.pop('PYTHONPATH',None)
    for iteration in range(max_iterations):
        pp,y=prepare(b,p,ts,ed,B);matrix=values(pp,menu)*y[:,None]
        np.savez(tmp/'input.npz',values=matrix,alphas=np.arange(len(menu)),groups=np.zeros(len(menu),int),betas=np.array([],int),parents=np.array([],int))
        process=subprocess.run([interpreter,str(core.ROOT/'fixed_policy_lp.py'),str(tmp/'input.npz'),str(tmp/'output.npz')],capture_output=True,env=env)
        assert process.returncode==0,process.stderr.decode(errors='replace')
        data=np.load(tmp/'output.npz');dual=np.maximum(0,data['dual']);dual/=dual.sum()
        weights=[max(F(0),F(str(v))) for v in data['weights']];total=sum(weights)
        if total>1:weights=[v/total for v in weights]
        strategy=[dict(AD=ad,allocation=float(w),allocation_exact=str(w)) for ad,w in zip(menu,weights) if w>F(1,10**12)]
        upper=max(0.,float((dual@matrix).max()))+3e-12
        candidate=dict(ED=ed,B=B,classical_B_rounds=B,samples=ts[:],sampled_dual=dual.tolist(),
            upper_numerical=upper,strategy=strategy,nmax=nmax,iterations=iteration+1,
            status='dual_measure_proposal_only',lower=0.,converged=False,
            note='No fixed-policy lower is inferred. Requires independent whole-menu outward dual evaluation.')
        if upper<=incumbent+tolerance or not strategy:return candidate
        numerical,t=engine.numerical_worst(b,p,ed,B,strategy,ts)
        candidate.update(numerical_minimum=numerical,numerical_argmin=t)
        if upper-numerical<=tolerance/2 or min(abs(v-t) for v in ts)<1e-11:return candidate
        ts.append(t)
    return candidate
