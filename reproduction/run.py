"""Portable launcher with per-command provenance and captured logs.

Examples: python run.py controls-cert; python run.py lower-certificates
Install requirements.txt in your environment; .deps is an optional local target.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
JOBS = {
    'controls-cert': ['reproduction/controls/code/certify_fixed_blocks.py'],
    'controls-circuits': ['reproduction/controls/code/compile_winner.py'],
    'controls-check': ['reproduction/controls/code/verify_outputs.py'],
    'controls-scan': ['reproduction/controls/code/fixed_block_scan.py'],
    'lower-certificates': ['reproduction/extensions/code/interval_rates.py', '--all'],
    'theory-checks': ['reproduction/extensions/code/verify.py', '--intervals'],
    'pumping-checks': ['reproduction/extensions/code/pumping_checks.py'],
    'bb84-block-control': ['reproduction/extensions/code/hashed_controls.py', '--certify'],
    'extended-endpoints': ['reproduction/extensions/code/analyze_evidence.py'],
    'enumeration': ['numerics/enumerate_independently.py'],
    'graph8': ['numerics/enumerate_independently.py', '--graphs-through', '8'],
    'prepare-runtime': ['numerics/prepare_runtime.py'],
    'validation-batch': ['numerics/validation_batch.py'],
    'legacy-audit': ['numerics/audit_legacy.py'],
    'matched-interval': ['numerics/matched_interval.py'],
    'results': ['numerics/assemble_results.py'],
    'figures': ['numerics/make_figures.py'],
    'freeze-inputs': ['numerics/freeze_release_inputs.py'],
    'literature': ['numerics/reconstruct_literature.py'],
}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', choices=JOBS)
    args=parser.parse_args()
    job=JOBS[args.job]
    env=os.environ.copy()
    dependency_path=ROOT/('.runtime' if (ROOT/'.runtime/READY.json').exists() else '.deps')
    if dependency_path.exists():
        env['PYTHONPATH']=str(dependency_path) + (os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env['OPENBLAS_NUM_THREADS']='1'
    env['OMP_NUM_THREADS']='1'
    env['MPLCONFIGDIR']=str(ROOT/'tmp/matplotlib')
    env['PYTHONIOENCODING']='utf-8'
    logdir=ROOT/'validation/runs'
    logdir.mkdir(parents=True,exist_ok=True)
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    script=ROOT/job[0]
    logfile=logdir/(args.job+'.log')
    start=time.monotonic()
    with logfile.open('w',encoding='utf-8') as stream:
        process=subprocess.Popen([sys.executable,'-X','utf8',*job],cwd=ROOT,env=env,
                                 stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        for line in process.stdout:
            stream.write(line);stream.flush();print(line,end='',flush=True)
        result=process.wait()
    record=dict(job=args.job,status='passed' if result==0 else 'failed',exit_code=result,
                started_utc=started,elapsed_seconds=time.monotonic()-start,command=[sys.executable,*job],
                script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                log_sha256=hashlib.sha256(logfile.read_bytes()).hexdigest())
    (logdir/(args.job+'.json')).write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2))
    raise SystemExit(result)

if __name__=='__main__':
    main()
