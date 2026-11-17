"""Diagnostic only: syndrome gain signs across the hidden BB84 interval."""
from pathlib import Path
import sys,json,math
import numpy as np
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
sys.path[:0]=[str(PROJECT/'research/revision_20260922'),str(HERE.parent)]
import core
from resource_model import model
from bb84_recursive import prepare
from literature_rates import watanabe_components


def spc_gains(p,n):
    typ=np.asarray(core.compositions(n),int)
    typ=typ[np.array([tuple(v)<=tuple(v[::-1]) for v in typ])]
    coeff=np.array([math.factorial(n)/math.prod(math.factorial(int(x)) for x in v)/(2 if tuple(v)==tuple(v[::-1]) else 1) for v in typ])
    a=np.prod(p[:,None,:]**typ[None,:,:],axis=-1)
    b=np.prod(p[:,None,:]**typ[None,:,::-1],axis=-1)
    ent=-core.xlog2x(a+b)*coeff[None,:]
    parity=(typ[:,1]+typ[:,2])%2
    masses=(1+(1-2*(p[:,1]+p[:,2]))[:,None]**n*np.array([1,-1])[None,:])/2
    return ((n-1)*masses-np.stack([ent[:,parity==s].sum(1) for s in (0,1)],axis=-1)-core.xlog2x(masses))/n


points=json.loads((HERE/'expanded_points.json').read_text())['points']
out=[]
for point in points:
    q=float(point['Q']); t=np.linspace(0,1,2001); z=q*t
    initial=np.stack((1-2*q+z,q-z,z,q-z),axis=-1)
    for M,policy in point['witnesses'].items():
        if M not in ('M2','M3','M5','M6') or policy['lower']<=0:continue
        ed=policy['ED']; B=policy['classical_B_rounds']
        if ed['family']=='explicit_binary_tree':
            val=model.evaluate(model.tree_from_dict(ed['tree']),initial,ed['final_axis'])
            p,y=val['state'],val['yield_per_raw_pair']
            for _ in range(B):p,s=core.b_step(p);y=y*s/2
        else:p,y=prepare(q,q,t,ed,B)
        for choice in policy['strategy']:
            action=choice['AD']
            if action['kind']=='Watanabe_2007':
                gains=watanabe_components(p)
                out.append(dict(q=q,M=M,action=action,branch_min=gains.min(0).tolist(),branch_max=gains.max(0).tolist(),
                    argmax_counts=np.bincount(np.argmax(gains,axis=1),minlength=2).tolist()))
            elif action['kind']=='SPC' or (action['kind']=='linear' and action.get('code')=='XC0055'):
                gains=spc_gains(p,action['n'])
                out.append(dict(q=q,M=M,action=action,branch_min=gains.min(0).tolist(),branch_max=gains.max(0).tolist(),
                    sign_changes=[bool(gains[:,i].min() < -1e-12 and gains[:,i].max() > 1e-12) for i in (0,1)],
                    first_negative_t=[float(t[np.flatnonzero(gains[:,i]<-1e-12)[0]]) if (gains[:,i]<-1e-12).any() else None for i in (0,1)],
                    first_positive_t=[float(t[np.flatnonzero(gains[:,i]>1e-12)[0]]) if (gains[:,i]>1e-12).any() else None for i in (0,1)]))
(HERE/'retention_sign_diagnostic.json').write_text(json.dumps(dict(diagnostic_only=True,records=out),indent=2))
print(json.dumps(out))
