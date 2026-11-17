"""Certify an existing plotted Werner point in an isolated fresh directory."""
from pathlib import Path
import argparse, hashlib, json, os, sys, time
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
REV=PROJECT/'research/revision_20261006'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
sys.dont_write_bytecode=True

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.writing.json');tmp.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
    for attempt in range(100):
        try:tmp.replace(p);return
        except PermissionError:
            if attempt==99:raise
            time.sleep(.05)

def calculate(q,task):
    output_root=REV/'recheck_20261022/werner'/q
    receipt=task/'receipt.json'
    if receipt.exists() and read(receipt).get('status')=='certified':return
    points=read(PROJECT/'revised/reproduction/data/inherited/frozen_curve_delivery.json')['points']
    row=next(p for p in points if p['model']=='Werner' and p['Q']==float(q))
    evidence=row['evidence']
    seed=REV/evidence['discovery_job'] if 'discovery_job' in evidence else (REV/evidence['original']).parent
    source=REV/'werner_search/run_v2.py';code=source.read_text(encoding='utf-8')
    old="run = HERE/'runs'/job_id";assert code.count(old)==1
    record=dict(Q=q,status='running',source_sha256=sha(source),discovery=str(seed),
      discovery_hashes={n:sha(seed/'data'/n) for n in ['registered_discovery.json','werner_complete_joint_n128.json']},
      historical_upper_values_used=False,
      scope='The unchanged declared 21 October Werner finite family; not an upper for an enlarged intermediate-Clifford family.')
    # Compile the same outward atanh-series arithmetic. No statistical
    # approximation or older upper value is introduced by this backend.
    from fast_entropy import install as entropy_install
    from fast_short_modes import install as short_modes_install
    from fast_long_menu import install as long_menu_install
    validation=HERE/'diagnostics/fast_entropy/result.json'
    backend=HERE/'fast_entropy.py'
    assert read(validation)['status']=='passed' and read(validation)['source_sha256']==sha(backend)
    record['entropy_backend']=dict(source_sha256=sha(backend),validation_sha256=sha(validation),
        arithmetic='Same atanh-series/gamma/remainder bounds; fastmath=False compiled scalar loops')
    short_validation=HERE/'diagnostics/fast_short_modes/result.json'
    assert read(short_validation)['status']=='passed' and read(short_validation)['source_sha256']==sha(HERE/'fast_short_modes.py')
    record['paid_OTP_backend']=dict(source_sha256=sha(HERE/'fast_short_modes.py'),validation_sha256=sha(short_validation),
        identity='Exact syndrome bit-weight enumerators and conditional phase-entropy additivity; all 132 codes and both signed paid modes checked')
    long_backend=HERE/'fast_long_menu.py'
    validations=[HERE/'diagnostics'/name/'result.json' for name in ('fast_long_menu','fast_long_integration')]
    for validation_path in validations:
        checked=read(validation_path)
        assert checked['status']=='passed' and checked['source_sha256']==sha(long_backend)
    record['long_menu_backend']=dict(source_sha256=sha(long_backend),
        validation_sha256=[sha(p) for p in validations],
        arithmetic='Same directed repetition and Hellinger ambiguity formulas; complete length menu, gamma bounds, fastmath=False')
    def numerical_install():entropy_install();short_modes_install();long_menu_install()
    save(receipt,record);start=time.monotonic()
    try:
        sys.path[:0]=[str(REV/'werner_search'),str(REV)]
        ns=dict(__name__='fresh_werner_point',__file__=str(source),FRESH_OUTPUT=output_root,ENTROPY_INSTALL=numerical_install)
        hook='provider = install(core)';assert code.count(hook)==1
        compiled=code.replace(old,'run = FRESH_OUTPUT/job_id').replace(hook,hook+'\n    ENTROPY_INSTALL()')
        exec(compile(compiled,str(source),'exec'),ns)
        sys.argv=[str(source),'--q',q,'--reuse-discovery',str(seed)]
        ns['main']()
        results=list(output_root.glob('*/result.json'));assert len(results)==1
        out=results[0];doc=read(out)
        record.update(status=doc['status'],results=doc['results'],output=str(out),
            output_sha256=sha(out),seconds=time.monotonic()-start)
        save(receipt,record)
        print(json.dumps(dict(Q=q,status=record['status'],seconds=record['seconds'])),flush=True)
    except BaseException as exc:
        record.update(status='failed',error=repr(exc),seconds=time.monotonic()-start);save(receipt,record);raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--q',required=True);args=ap.parse_args()
    q=str(float(args.q));task=HERE/'werner'/q;task.mkdir(parents=True,exist_ok=True)
    # Forward and reverse workers may meet. Only one process can mutate a
    # point's engine directory; a second worker waits for its receipt.
    lock=task/'compute.lock'
    while True:
        if (task/'receipt.json').exists() and read(task/'receipt.json').get('status')=='certified':return
        try:
            fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
            with os.fdopen(fd,'w') as stream:stream.write(str(os.getpid()))
            break
        except FileExistsError:time.sleep(2)
    try:calculate(q,task)
    finally:lock.unlink()


if __name__=='__main__':main()
