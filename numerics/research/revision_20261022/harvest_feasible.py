"""Assemble rigorous feasible incumbents from completed interval work.

This is only a lower-witness assembly, not a claim of exhaustive discovery.
Independent whole-family upper calculations cover every omitted proposal.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse,hashlib,json
from binary_family_search import setup
from full_clifford_model import HERE
from full_binary_search import read,save

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--q',type=float,required=True)
    parser.add_argument('--label',choices=['full','real_xyz','real_xz'],required=True);a=parser.parse_args()
    catalog='full_binary_catalog' if a.label=='full' else 'real_binary_catalog'
    job,core,engine,model,ops,nodes,events,record=setup(a.q,catalog,a.label)
    results={f'M{m}':dict(lower=0.) for m in range(1,9)};records=[]
    def offer(row,origin):
        lower=float(row['lower']);m=row['ED']['memory_upper_bound']
        assert lower<=max(0.,row['certificate']['lower'])+1e-18
        if lower>0:
            row=dict(row,origin=origin)
            for budget in range(m,9):
                key=f'M{budget}'
                if lower>results[key]['lower']:results[key]=row
            records.append(dict(origin=origin,memory=m,lower=lower))
    # Each saved heap is a complete dyadic cover. Check this exactly rather
    # than taking a stored scalar minimum without its coverage information.
    for path in sorted((job/'data/bb84_interval_work').glob('*.json')):
        d=read(path)
        if d['b']!=a.q or d['p']!=a.q:continue
        cells=sorted((F(r[2]),F(r[3]),float(r[0])) for r in d['heap'])
        at=F(0)
        for left,right,lower in cells:
            assert left==at and right>left,(path,left,at)
            at=right
        assert at==1
        lower=min(r[2] for r in cells);upper=d['incumbent']
        cert=dict(lower=lower,upper=upper,gap=upper-lower,argmin=d['argmin'],cells=d['counter'],
            converged=upper-lower<=2e-9,atol=2e-9,
            arithmetic='Saved outward interval leaves; exact rational full-domain cover checked during lower assembly.',
            work_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        offer(dict(lower=max(0.,lower),ED=d['ED'],B=d['B'],classical_B_rounds=d['B'],strategy=d['strategy'],certificate=cert),str(path))
    pending=[]
    for path in sorted((job/'data/bb84_pending_policies').glob('*.json')):
        d=read(path)
        if d['b']!=a.q or d['p']!=a.q:continue
        if d.get('status')=='certified_policy':offer(d['result'],str(path))
        elif d.get('strategy'):pending.append((d,path))
    # Re-evaluate the best sampled Watanabe proposals. These need no LP.
    rows=read(job/'discovery.json')['candidates']
    selected={tuple(max((r for r in rows if r[4]<=m),key=lambda r:r[0])) for m in range(1,9)}
    strategy=[dict(AD=dict(kind='Watanabe_2007',n=2,mode='no_otp_hashed'),allocation=1.,allocation_exact='1')]
    for _,idx,axis,B,m in sorted(selected,reverse=True):
        ed=record(idx,axis)
        cert=engine.certify_policy(a.q,a.q,ed,B,strategy,atol=2e-9,max_cells=16000)
        offer(dict(lower=max(0.,cert['lower']),ED=ed,B=B,classical_B_rounds=B,strategy=strategy,certificate=cert),'fresh full-interval Watanabe replay')
    for d,path in pending:
        m=d['ED']['memory_upper_bound']
        if d.get('upper_numerical',0)<=results[f'M{m}']['lower']:continue
        # The numerical comparison only chooses whether to seek a new lower.
        # It is never used to upper-bound this candidate or the whole family.
        cert=engine.certify_policy(a.q,a.q,d['ED'],d['B'],d['strategy'],atol=2e-9,max_cells=16000)
        offer(dict(lower=max(0.,cert['lower']),ED=d['ED'],B=d['B'],classical_B_rounds=d['B'],strategy=d['strategy'],certificate=cert),str(path)+'; fresh fixed-policy replay')
    assert all('ED'in r for r in results.values())
    output=dict(complete=True,Q=a.q,label=a.label,catalog=catalog,axes='XZ' if a.label=='real_xz' else 'XYZ',results=results,
        execution=dict(method='Certified feasible lower assembly, with exact checking of saved interval covers.',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        witness_sources=records,
        scope='Every stored lower is achieved by its fixed policy over all hidden t. No exhaustive search or optimum is inferred from this lower assembly; independent upper drivers cover the whole finite family and classical menu.')
    save(job/'feasible.json',output)
    print('FEASIBLE LOWER ASSEMBLY',a.q,a.label,{k:r['lower'] for k,r in results.items()},flush=True)
if __name__=='__main__':main()
