"""Resumable completion of every plotted Werner finite-family enclosure."""
from pathlib import Path
import argparse, concurrent.futures, json, subprocess, sys, time
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1]
def read(p):return json.loads(Path(p).read_bytes())
def save(p,x):
    t=p.with_suffix('.writing.json');t.write_text(json.dumps(x,indent=2)+'\n')
    for attempt in range(100):
        try:t.replace(p);return
        except PermissionError:
            if attempt==99:raise
            time.sleep(.05)
def done(q):
    p=HERE/'werner'/q/'receipt.json'
    return p.exists() and read(p).get('status')=='certified'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--limit',type=int);ap.add_argument('--reverse',action='store_true')
    ap.add_argument('--exclude',nargs='*',default=[])
    ap.add_argument('--label',default='');ap.add_argument('--min-q',type=float,default=0.)
    ap.add_argument('--max-q',type=float,default=1.);args=ap.parse_args();assert 1<=args.workers<=6
    status_file=HERE/('werner_status'+('_'+args.label if args.label else '')+'.json')
    points=read(PROJECT/'revised/reproduction/data/inherited/frozen_curve_delivery.json')['points']
    todo=[str(p['Q']) for p in points if p['model']=='Werner' and p['status']!='certified_optimum']
    assert len(todo)==114
    # Short low-noise jobs establish throughput; expensive tail jobs follow.
    excluded={str(float(v)) for v in args.exclude}
    todo.sort(key=float,reverse=args.reverse);pending=[q for q in todo if not done(q) and q not in excluded and args.min_q<=float(q)<=args.max_q]
    if args.limit is not None:pending=pending[:args.limit]
    save(HERE/'werner_plan.json',dict(points=todo,total=114,previous_anchors=6,
       previous_anchor_root='research/revision_20261006/recheck_20261021/runs/werner',
       memory_budgets=list(range(1,9)),scope='21 October finite family; component of any subsequent union'))
    def launch(q):
        job=HERE/'werner'/q;job.mkdir(parents=True,exist_ok=True)
        with (job/'calculation.log').open('ab') as log:
            result=subprocess.run([sys.executable,'-X','utf8',str(HERE/'werner_point.py'),'--q',q],stdout=log,stderr=subprocess.STDOUT)
        return q,result.returncode
    failures=[];start=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        iterator=iter(pending);active={}
        for _ in range(args.workers):
            q=next(iterator,None)
            if q is not None:active[pool.submit(launch,q)]=q
        while active:
            finished,_=concurrent.futures.wait(active,timeout=5,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in finished:
                q,rc=future.result();active.pop(future)
                if rc or not done(q):failures.append(dict(Q=q,returncode=rc))
                if rc:iterator=iter([])  # Stop dispatch on an engine/runtime failure.
                print('WERNER',q,'certified' if done(q) else 'needs_review',flush=True)
                nxt=next(iterator,None)
                if nxt is not None:active[pool.submit(launch,nxt)]=nxt
            save(status_file,dict(status='running',complete=sum(map(done,todo)),total=114,
                 active=list(active.values()),failures=failures,seconds=time.monotonic()-start))
    complete=sum(map(done,todo))
    save(status_file,dict(status='complete' if complete==114 else 'incomplete',complete=complete,total=114,
        active=[],failures=failures,seconds=time.monotonic()-start))
if __name__=='__main__':main()
