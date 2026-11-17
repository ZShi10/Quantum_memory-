"""Recompute numerical uppers without importing any historical upper cache.

Historical discovery records provide candidate policies and proposed dual
measures only. Every candidate receives a newly evaluated outward bound.
The wrapper changes writable locations and accepts rational-string proposed
test correlations. It does not change protocol menus or the bound arithmetic.
"""
from pathlib import Path
import argparse, hashlib, json, os, shutil, sys, time

HERE = Path(__file__).resolve().parent
REV = HERE.parent
PROJECT = REV.parents[1]
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
sys.dont_write_bytecode = True

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    t=p.with_suffix('.writing.json')
    t.write_text(json.dumps(x, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    t.replace(p)

def calculate():
    ap=argparse.ArgumentParser()
    ap.add_argument('--component', choices=['bb84-registered','bb84-block','werner'], required=True)
    ap.add_argument('--q', required=True)
    args=ap.parse_args()
    args.q=str(float(args.q))
    points=json.loads((PROJECT/'revised/reproduction/data/inherited/frozen_curve_delivery.json').read_bytes())['points']
    model='Werner' if args.component=='werner' else 'BB84_real'
    row=next(p for p in points if p['model']==model and str(p['Q'])==str(float(args.q)))
    task=HERE/'runs'/args.component/args.q
    task.mkdir(parents=True, exist_ok=True)
    receipt=task/'recompute_receipt.json'
    if receipt.exists():
        old=json.loads(receipt.read_bytes())
        if old.get('status')=='complete':
            print('Already recomputed in this isolated campaign', receipt, flush=True)
            return
    record=dict(component=args.component, Q=args.q, status='running',
        historical_upper_values_used=False, wrapper_sha256=sha(__file__),
        interpretation='Fresh outward whole-class bound; discovery candidates and dual proposals may be reused. No historical upper endpoint is imported.')
    save(receipt, record)
    start=time.monotonic()
    try:
        if args.component=='bb84-registered':
            sys.path.insert(0,str(REV/'bb84_search'))
            import run_point
            source_job=REV/'bb84_search/runs'/row['registered_point_fingerprint']
            summary=json.loads((source_job/'summary.json').read_bytes())
            source=source_job/summary['source']
            old_manifest=json.loads((source_job/'manifest.json').read_bytes())
            dest=task/'data'/source.name
            save(task/'manifest.json',dict(specification=old_manifest['specification']))
            dest.parent.mkdir(exist_ok=True)
            if not dest.exists(): shutil.copyfile(source,dest)
            assert sha(source)==sha(dest)
            core, engine, upper, active=run_point.bootstrap(task)
            proposed=json.loads(source.read_bytes())
            tol=min(max(1e-11,1e-4*float(v['lower'])) for v in proposed['results'].values())/2
            # This fresh directory contains no imported_interval_dual_evidence.
            # Its checkpoint, if any, belongs only to this new recomputation.
            assert not (core.DATA/'imported_interval_dual_evidence.json').exists()
            upper.certify(dest,tolerance=tol)
            output=dest.with_name(dest.stem+'_menu_certified.json')
            result=json.loads(output.read_bytes())
            assert result['complete'] and len(result['rounds'])==proposed['K']+1
            record.update(discovery_source=str(source),discovery_sha256=sha(source),
                output=str(output),output_sha256=sha(output),results=result['results'],
                physical_candidates=result['rounds'][0]['physical_candidates'],
                objective_classes=result['rounds'][0]['exact_objective_classes'],
                freshly_replayed_duals=sum(r['new_duals_replayed'] for r in result['rounds']),
                requested_tolerance=tol)
        elif args.component=='bb84-block':
            source=REV/'code_extension/bb84_m5/audit_upper.py'
            code=source.read_text(encoding='utf-8')
            old="job=HERE/'upper_runs'/tag;job.mkdir(parents=True,exist_ok=True)"
            new="job=RECHECK_OUTPUT/tag;job.mkdir(parents=True,exist_ok=True)"
            assert code.count(old)==1
            # Some exact certificates store a dyadic minimizer as "a/b".
            # Normalize that proposal to a legal binary64 test state; the
            # selected test state is then evaluated by outward arithmetic.
            # It is a proposal only, never a supplied numerical upper.
            code=code.replace("float(cert.get('argmin',-1))",
                              "float(__import__('fractions').Fraction(str(cert.get('argmin',-1))))")
            code=code.replace("ts.add(str(cert['argmin']))",
                              "ts.add(str(float(__import__('fractions').Fraction(str(cert['argmin'])))))")
            ns=dict(__name__='recheck_block',__file__=str(source),RECHECK_OUTPUT=task)
            exec(compile(code.replace(old,new),str(source),'exec'),ns)
            settings=argparse.Namespace(p=None,permissions='real',nmax=128,K=16,
                tolerance=1e-11,quantum=1e-12,lower_file=[],test_t=[])
            result=ns['run'](settings,float(args.q))
            assert result['complete']
            output=task/result['job_fingerprint']/'result.json'
            record.update(source_sha256=sha(source), output=str(output),output_sha256=sha(output),
                results=result['results'],instruments=len(result['entries']),
                output_path_patch=dict(before=old,after=new),
                proposed_test_state_parser='Accept numeric and rational-string correlations; every legal test state is evaluated outward.')
        else:
            source=REV/'werner_search/run_v2.py'
            seed=(REV/row['evidence']['original']).parent
            code=source.read_text(encoding='utf-8')
            old="run = HERE/'runs'/job_id"
            new="run = RECHECK_OUTPUT/job_id"
            assert code.count(old)==1
            sys.path[:0]=[str(REV/'werner_search'),str(REV)]
            ns=dict(__name__='recheck_werner',__file__=str(source),RECHECK_OUTPUT=task)
            exec(compile(code.replace(old,new),str(source),'exec'),ns)
            sys.argv=[str(source),'--q',args.q,'--reuse-discovery',str(seed)]
            ns['main']()
            outputs=list(task.glob('*/result.json'));assert len(outputs)==1
            output=outputs[0];result=json.loads(output.read_bytes())
            record.update(source_sha256=sha(source),output=str(output),output_sha256=sha(output),
                results=result['results'],optimizer_status=result['status'],
                output_path_patch_only=dict(before=old,after=new))
        loaded={}
        for mod in list(sys.modules.values()):
            p=getattr(mod,'__file__',None)
            if p and Path(p).suffix=='.py' and Path(p).is_relative_to(PROJECT):
                loaded[str(Path(p).relative_to(PROJECT))]=sha(p)
        record.update(status='complete',seconds=time.monotonic()-start,loaded_source_sha256=loaded)
        save(receipt,record)
        print(json.dumps({k:record[k] for k in ('status','component','Q','seconds','results')},indent=2),flush=True)
    except BaseException as exc:
        record.update(status='failed',seconds=time.monotonic()-start,error=repr(exc))
        save(receipt,record)
        raise

def main():
    # A priority replay and the campaign may request the same point. Serialize
    # that exact task, then let calculate() recheck its completion receipt.
    # The OS releases this lock even if the process is interrupted.
    ap=argparse.ArgumentParser()
    ap.add_argument('--component',choices=['bb84-registered','bb84-block','werner'],required=True)
    ap.add_argument('--q',required=True)
    args=ap.parse_args()
    args.q=str(float(args.q))
    task=HERE/'runs'/args.component/args.q;task.mkdir(parents=True,exist_ok=True)
    with (task/'.recompute.lock').open('a+b') as lock:
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0)
        if os.name=='nt':
            import msvcrt
            while True:
                try:
                    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);break
                except OSError as error:
                    if error.errno not in (13,36):raise
                    time.sleep(2)
        else:
            import fcntl
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        calculate()


if __name__=='__main__': main()
