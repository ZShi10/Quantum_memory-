"""Dense achievable BB84 curves for three explicitly fixed policies.

Each point has a full hidden-parameter interval lower bound. These curves
are neither sampled minima nor claimed optimal envelopes. Complete-family
upper exclusions are supplied by the separate full_bb84_control.py jobs.
"""
from pathlib import Path
from fractions import Fraction as F
import csv, hashlib, json, os, shutil, sys, time
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
HERE=Path(__file__).resolve().parent
REV=HERE.parent
PROJECT=REV.parents[1]
TASK=HERE/'full_bb84/fixed_curves'


def read(p):return json.loads(p.read_bytes())
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.writing.json')
    tmp.write_text(json.dumps(x,indent=2)+'\n')
    tmp.replace(p)


def main():
    TASK.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(REV/'bb84_search'))
    import run_point
    args=run_point.parser().parse_args(['--q','.18','--permissions','full','--short-max','8','--ed-extension'])
    save(TASK/'manifest.json',dict(specification=run_point.exact_spec(args)))
    shutil.copyfile(PROJECT/'research/revision_20260922/fixed_policy_lp.py',TASK/'fixed_policy_lp.py')
    core,engine,_,_=run_point.bootstrap(TASK)
    base=read(HERE/'full_bb84/baseline/0.17/result.json')['best']
    source=read(HERE/'full_bb84/registered_upper/0.18/data/full_terminal_axes_0.18.json')['results']
    policies={}
    for m,row in [(1,base),(2,source['M2']),(3,source['M3'])]:
        policies[str(m)]=dict(ED=row['ED'],B=row.get('classical_B_rounds',row.get('B',0)),strategy=row['strategy'])
    save(TASK/'policies.json',policies)
    digest=hashlib.sha256((TASK/'policies.json').read_bytes()).hexdigest()
    rows=[];started=time.monotonic()
    for index in range(91):
        q=F(16,100)+F(index,2000)
        file=TASK/'points'/(str(q).replace('/','-')+'.json')
        if file.exists():
            row=read(file)
            assert row['policy_sha256']==digest
        else:
            rates={}
            for m,p in policies.items():
                cert=engine.certify_policy(float(q),float(q),p['ED'],p['B'],p['strategy'],
                                           atol=2e-8,max_cells=6000)
                rates[m]=dict(lower=max(0.,cert['lower']),certificate=cert)
            row=dict(Q=str(q),policy_sha256=digest,results=rates)
            save(file,row)
        rows.append(row)
        print('FIXED BB84 CURVES',index+1,91,str(q),{m:r['lower'] for m,r in row['results'].items()},flush=True)
    with (TASK/'curves.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['Q','R1','R2','R3'])
        writer.writeheader()
        for r in rows:writer.writerow(dict(Q=str(float(F(r['Q']))),**{f'R{m}':v['lower'] for m,v in r['results'].items()}))
    save(TASK/'receipt.json',dict(complete=True,points=len(rows),range=['4/25','41/200'],
        policy_sha256=digest,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        all_continuous_hidden_z=True,seconds=time.monotonic()-started,
        interpretation='Fixed achievable policies, one continuous hidden-z interval lower per Q. Not family optima and no source-Q continuum certificate.'))


if __name__=='__main__':main()
