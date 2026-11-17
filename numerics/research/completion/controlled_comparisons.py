"""Matched objective/cost comparisons, with interval-certified witnesses."""
from common import *
import json
import numpy as np
from complete_werner_search import load
from libraries import code_kernels
from rates import best_codes
from streaming import rate as rep
from certify_werner_witnesses import witness


def main():
    data,weights,_,_=load();ks=code_kernels();kd={k.entry['id']:k for k in ks};output=[];baseline=[]
    qs=[.10,.15,.18,.20,.21,.22,.23,.24,.25,.26,.27,.275,.28,.285,.29,.295,.30,.305,.31,.315,.32]
    def raw(e,q):
        m=e['m'];b=next(v for v in e['branches'] if v['syndrome']==0)
        v=weights[m][b['state']]@np.array([(1-1.5*q)**(m-j)*(q/2)**j for j in range(m+1)])*b['factor']
        return v,b
    for q in qs:
        best=None
        for e in data['entries']:
            if e['m']>4:continue
            v,b=raw(e,q);ps=v.sum();p=v/ps;a=p[0]+p[3];bb=p[1]+p[2]
            if a<bb:p=p[[1,0,3,2]];a,bb=bb,a
            for n in range(1,{1:1,2:4,3:3,4:2}[e['m']]+1):
                value=float(ps/e['m']*rep(a-bb,p[3]/a,p[2]/bb,0,n))
                if best is None or value>best['rate']:
                    best=dict(Q=q,rate=value,ED=e['id'],m=e['m'],M=e['M'],branches=[dict(**b,kind='repetition',n=n,binary_rounds=0,canonicalize_bit=True,contribution=value)])
        if best['rate']>0:best['certificate']=witness(best,next(e for e in data['entries'] if e['id']==best['ED']),weights,kd)
        baseline.append(best)
        if q not in (.15,.20,.25,.28,.30):continue
        for m in (3,4):
            es=[e for e in data['entries'] if e['m']==m and e['M']<=2];vv=[raw(e,q) for e in es]
            rawvals=np.array([v for v,b in vv]);ps=rawvals.sum(1);p=rawvals/ps[:,None];fidelity=p.max(1)
            state=p.copy();factor=ps/m;rates=np.zeros(len(es));choices=[None]*len(es)
            for k in range(17):
                r=best_codes(state,ks);values=r['rate']*factor
                for i in np.flatnonzero(values>rates):
                    rates[i]=values[i];choices[i]=dict(**vv[i][1],kind='linear',code=ks[r['code'][i]].entry['id'],binary_rounds=k,contribution=float(values[i]))
                a,x,y,z=state.T;v=np.stack((a*a+z*z,x*x+y*y,2*x*y,2*a*z),1);mass=v.sum(1);factor*=mass/2;state=v/mass[:,None]
            tied=np.flatnonzero(fidelity>fidelity.max()-1e-13);fi=int(tied[np.argmax(rates[tied])]);ri=int(rates.argmax())
            result=dict(Q=q,m=m,M_budget=2,scope='All m-input fixed terminal Clifford instruments, zero ED syndrome, all n<=7 codes, B prefixes<=16; best final rate among all fidelity maximizers.')
            for label,i in [('fidelity_first',fi),('rate_first',ri)]:
                e=es[i];r=dict(Q=q,rate=float(rates[i]),ED=e['id'],m=m,M=e['M'],F=float(fidelity[i]),success=float(ps[i]),branches=[choices[i]] if choices[i] else [])
                r['certificate']=witness(r,e,weights,kd);result[label]=r
            output.append(result)
            print('Matched fidelity',q,m,result['fidelity_first']['rate'],result['rate_first']['rate'],flush=True)
    (DATA/'controlled_comparisons.json').write_text(json.dumps(dict(status='passed',fidelity=output,matched_size_baseline=baseline,baseline_scope='Joint-entropy implementation of the nine (m,n,1) sizes used in Sun et al.: m=2,n<=4; m=3,n<=3; m=4,n<=2, plus immediate hashing. Full zero-syndrome Clifford ED search in each size; zero-syndrome repetition AD; all raw costs included. This reproduces a matched-size protocol family, not every choice of the published implementation.'),separators=(',',':')))


if __name__=='__main__':main()
