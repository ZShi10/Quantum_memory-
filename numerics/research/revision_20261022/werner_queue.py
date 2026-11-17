"""Fill idle computation slots without dispatching an already locked point."""
from pathlib import Path
import argparse,concurrent.futures,json,subprocess,sys,time
from werner_campaign import HERE,PROJECT,done,read,save

points=read(PROJECT/'revised/reproduction/data/inherited/frozen_curve_delivery.json')['points']
parser=argparse.ArgumentParser()
parser.add_argument('--label',default='queue')
parser.add_argument('--workers',type=int,default=2)
parser.add_argument('--reverse',action='store_true')
args=parser.parse_args()
assert args.label.replace('_','').isalnum() and 1<=args.workers<=7
status_path=HERE/f'werner_status_{args.label}.json'
todo=sorted([str(p['Q']) for p in points if p['model']=='Werner' and p['status']!='certified_optimum'],key=float,reverse=args.reverse)
assert len(todo)==114

def launch(q):
    job=HERE/'werner'/q;job.mkdir(parents=True,exist_ok=True)
    with (job/'calculation.log').open('ab') as stream:
        r=subprocess.run([sys.executable,'-X','utf8',str(HERE/'werner_point.py'),'--q',q],stdout=stream,stderr=subprocess.STDOUT)
    return q,r.returncode

start=time.monotonic();failures=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
    active={}
    while True:
        remaining=[q for q in todo if not done(q)]
        for future in list(active):
            if not future.done():continue
            q,rc=future.result();del active[future]
            print('QUEUE COMPLETE',q,rc,flush=True)
            if rc or not done(q):failures.append(dict(Q=q,returncode=rc))
        if failures:break
        if not remaining and not active:break
        occupied=sum((HERE/'werner'/q/'compute.lock').exists() for q in todo)
        room=min(args.workers-len(active),max(0,7-occupied))
        available=[q for q in remaining if q not in active.values() and not (HERE/'werner'/q/'compute.lock').exists()]
        for q in available[:room]:active[pool.submit(launch,q)]=q
        save(status_path,dict(status='running',complete=len(todo)-len(remaining),total=114,
            active=list(active.values()),global_locked=occupied,failures=failures,seconds=time.monotonic()-start))
        time.sleep(5)
save(status_path,dict(status='complete' if all(map(done,todo)) else 'failed',
    complete=sum(map(done,todo)),total=114,active=[],failures=failures,seconds=time.monotonic()-start))
