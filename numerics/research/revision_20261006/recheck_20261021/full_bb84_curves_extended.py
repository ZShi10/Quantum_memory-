"""Extend the fixed-policy BB84 plot to Q=.24 and a four-slot Y witness.

Reuse only unchanged lower-policy certificates, never numerical uppers.
Each new point is independently bounded over the full hidden interval.
"""
from pathlib import Path
from fractions import Fraction as F
import csv, hashlib, json, os, shutil, sys, time
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
HERE=Path(__file__).resolve().parent
REV=HERE.parent
PROJECT=REV.parents[1]
TASK=HERE/'full_bb84/fixed_curves_extended'


def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.writing.json');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)


def main():
    TASK.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(REV/'bb84_search'))
    import run_point
    args=run_point.parser().parse_args(['--q','.18','--permissions','full','--short-max','8','--ed-extension'])
    save(TASK/'manifest.json',dict(specification=run_point.exact_spec(args)))
    shutil.copyfile(PROJECT/'research/revision_20260922/fixed_policy_lp.py',TASK/'fixed_policy_lp.py')
    core,engine,_,_=run_point.bootstrap(TASK)
    old=HERE/'full_bb84/fixed_curves'
    policies=read(old/'policies.json')
    original_digest=sha(old/'policies.json')
    assert set(policies)=={'1','2','3'}
    high=read(HERE/'full_bb84/high_noise_witnesses/0.23/receipt.json')
    assert high['complete']
    p=high['best'];assert p['ED']['memory_upper_bound']==4
    policies['4']=dict(ED=p['ED'],B=p['B'],strategy=p['strategy'])
    save(TASK/'policies.json',policies);digest=sha(TASK/'policies.json')
    rows=[];started=time.monotonic()
    for index in range(161):
        q=F(16,100)+F(index,2000)
        name=str(q).replace('/','-')+'.json';file=TASK/'points'/name
        if file.exists():
            row=read(file);assert row['policy_sha256']==digest
        else:
            rates={};provenance=None
            previous=old/'points'/name
            if previous.exists():
                prior=read(previous);assert prior['policy_sha256']==original_digest
                rates=prior['results'].copy()
                provenance=dict(path=str(previous),sha256=sha(previous),unchanged_policies=['1','2','3'])
            for m,p in policies.items():
                if m in rates:continue
                cert=engine.certify_policy(float(q),float(q),p['ED'],p['B'],p['strategy'],
                                           atol=2e-8,max_cells=6000)
                rates[m]=dict(lower=max(0.,cert['lower']),certificate=cert)
            row=dict(Q=str(q),policy_sha256=digest,results=rates,unchanged_lower_source=provenance)
            save(file,row)
        rows.append(row)
        print('EXTENDED BB84 CURVES',index+1,161,str(q),{m:r['lower'] for m,r in row['results'].items()},flush=True)
    with (TASK/'curves.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['Q','R1','R2','R3','R4']);w.writeheader()
        for r in rows:w.writerow(dict(Q=str(float(F(r['Q']))),**{f'R{m}':v['lower'] for m,v in r['results'].items()}))
    save(TASK/'receipt.json',dict(complete=True,points=len(rows),range=['4/25','6/25'],
        policy_sha256=digest,source_sha256=sha(Path(__file__)),all_continuous_hidden_z=True,
        seconds=time.monotonic()-started,
        interpretation='Four fixed achievable policies. Each plotted point has a full hidden-z lower; no family optimum or continuous source-Q separation is inferred.'))


if __name__=='__main__':main()
