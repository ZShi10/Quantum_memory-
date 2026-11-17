"""Independent finite-block controls, without reusing all-flag rate quotients."""
from __future__ import annotations
import argparse,csv,json,time,sys
from fractions import Fraction as F
from pathlib import Path
import networkx as nx
from stopping import *
from bell_core import watanabe,entropy,axis
ROOT=Path(__file__).resolve().parents[1]

def real_graphs(nmax=5):
    for index,g in enumerate(nx.graph_atlas_g()):
        n=len(g)
        if not 1<=n<=nmax:continue
        adj=[sum(1<<j for j in g.neighbors(i)) for i in range(n)]
        yield 'atlas_'+str(index),adj

def run_bounds(Q,price,z=F(0),nmax=5):
    p=[1-2*Q+z,Q-z,z,Q-z];rows=[];start=time.monotonic();fail=[];count=0
    for name,adj in real_graphs(nmax):
        for dual in range(1,1<<len(adj)):
            weights=branch_weights(adj,dual,p);mass=[sum(w) for w in weights]
            reward=source_reward(weights,exact=True)
            root=stopping_test(adj,dual,mass,reward,price)
            count+=1
            if root>0:fail.append((name,dual,float(root)))
        if len(adj)==nmax:print('bound',name,'tested',count,'not excluded',len(fail),'sec',round(time.monotonic()-start,1),flush=True)
    rec={'Q':str(Q),'z':str(z),'price':str(price),'tested':count,'excluded':count-len(fail),'not_excluded':fail,'seconds':time.monotonic()-start,
    'scope':'All graph/H real complete-readout circuits through Nmax, all one-key public-check hyperplanes; any syndrome-adaptive loading and stopping in complete-readout compiler class; square E_R upper for any downstream classical processing; symmetric source.'}
    (ROOT/'data'/f'block_source_upper_Q{float(Q):.3f}.json').write_text(json.dumps(rec,indent=2))
    print('DONE', {k:v for k,v in rec.items() if k!='not_excluded'})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--Q',default='23/100');parser.add_argument('--price',default='1/2000');parser.add_argument('--nmax',type=int,default=5)
    a=parser.parse_args();run_bounds(F(a.Q),F(a.price),nmax=a.nmax)
