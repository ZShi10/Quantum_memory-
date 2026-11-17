"""Relocatable numerical entry points. Run from the staged bundle root."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parent

def launch(driver,*args):
    env=os.environ.copy();env.setdefault('QKD_SCIPY_PYTHON',sys.executable)
    env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
    command=[sys.executable,'-X','utf8',str(ROOT/'research/revision_20261022/run_numerical.py'),driver,*args]
    process=subprocess.run(command,cwd=ROOT,env=env)
    if process.returncode:raise SystemExit(process.returncode)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['verify','arithmetic','werner','bb84-discover','bb84-upper','bb84-curve','portfolio','costs','profiles'])
    ap.add_argument('--q');a=ap.parse_args()
    if a.command=='verify':
        manifest=json.loads((ROOT/'MANIFEST.json').read_bytes());count=0
        for row in manifest['source_files']+manifest['evidence_files']:
            p=ROOT/row['path'];assert p.is_file(),p
            assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],p;count+=1
        print(json.dumps(dict(status='passed',hashed_files=count,scope='Archive integrity only; this does not rerun the optimizer.')));return
    if a.command=='arithmetic':
        launch('fast_entropy.py');launch('fast_short_modes.py');launch('symmetric_spc_certificate.py','--validate')
        launch('fast_long_menu.py');launch('check_fast_long_integration.py');launch('fast_short_screen.py');return
    if a.command=='portfolio':launch('audit_portfolio_execution.py');return
    if a.command=='costs':launch('analyze_full_witnesses.py');return
    if a.command=='profiles':launch('full_witness_profiles.py');return
    assert a.q,'Supply --q for a numerical source point.'
    if a.command=='werner':launch('werner_point.py','--q',a.q)
    elif a.command=='bb84-curve':launch('reproduce_dense_policy.py','--q',a.q)
    elif a.command=='bb84-discover':launch('binary_family_search_efficient.py','--q',a.q,'--label','full')
    else:
        q=str(float(a.q));folder=ROOT/'research/revision_20261022/bb84'/q
        launch('full_binary_upper_grouped.py','--q',q,'--catalog','full_binary_catalog','--axes','XYZ','--label','full','--lower',str(folder/'full/feasible.json'))
        launch('refine_binary_upper_v2.py','--base',str(folder/'upper_grouped_full_latest.json'),'--solve','--dual-only')

if __name__=='__main__':main()
