"""Certified achievable BB84 input grid, with optional input H exchange.

The compact displayed catalog uses Watanabe and n<=7 Du actions from
saved witnesses. It is a constructive diagnostic, not an asymmetric
all-generator optimum. Every positive grid value is uniform over hidden z.
"""
from core import *
from bb84_policy_curves import catalog,point,policy_id
import argparse,hashlib,time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--step',type=float,default=.02)
    parser.add_argument('--pair',type=float,nargs=2);a=parser.parse_args()
    policies,existing=catalog('real')
    policies={k:v for k,v in policies.items() if all(i['AD']['n']<=7 for i in v['strategy'])}
    digest=hashlib.sha256(json.dumps(policies,sort_keys=True).encode()).hexdigest()
    registry=DATA/'bb84_asymmetric_policies.json';atomic_json(registry,dict(digest=digest,policies=policies))
    path=DATA/'bb84_asymmetric_grid.json';out=json.loads(path.read_text()) if path.exists() else dict(results={})
    count=round(.5/a.step)
    points=[tuple(a.pair)] if a.pair else [(round(i*a.step,8),round(j*a.step,8))
             for i in range(count+1) for j in range(i,count+1) if (i+j)*a.step<=.5+1e-12]
    # Useful nonzero region first; endpoint rows have analytic zero guarantees.
    points.sort(key=lambda x:(sum(x),x))
    for b,p in points:
        key=f'{b:.8f}_{p:.8f}'
        if key in out['results'] and out['results'][key]['catalog_digest']==digest:continue
        start=time.time();direct=point(b,p,policies,existing,relative=1e-3,max_cells=5000,absolute=1e-7)
        exchange=point(p,b,policies,existing,relative=1e-3,max_cells=5000,absolute=1e-7) if b!=p else direct
        results={}
        for budget in direct['results']:
            use_swap=exchange['results'][budget]['lower']>direct['results'][budget]['lower']
            row=(exchange if use_swap else direct)['results'][budget]
            results[budget]=dict(**row,input_basis_exchange=use_swap)
        out['results'][key]=dict(b=b,p=p,results=results,catalog_digest=digest,seconds=time.time()-start)
        out['scope']='Achievable finite-catalog lower bounds at displayed input points, each uniform over all compatible correlations. Catalog: saved real-operation ED witnesses with Watanabe or Du n<=7 terminals. Also allow H on every raw input, exchanging b and p without extra live memory. No inter-grid guarantee or whole-generator optimum is asserted.'
        atomic_json(path,out);print('BB asymmetric',b,p,{k:v['lower'] for k,v in results.items()},round(time.time()-start,1),flush=True)


if __name__=='__main__':main()
