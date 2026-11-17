"""Resumable fresh upper-bound campaign; no archived upper is an input proof.

Only points already carrying upper claims are scheduled. Additional Werner
lower-only drawing points are not relabelled as certified optima. A completed
receipt is subsequently checked by compare_recompute.py, not accepted as a
claim that an arbitrary-memory optimization has been solved.
"""
from pathlib import Path
import argparse, concurrent.futures, json, subprocess, sys, time
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]

def read(p):return json.loads(p.read_bytes())
def save(p,obj):
    t=p.with_suffix('.writing.json');t.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8');t.replace(p)
def taskdir(task):return HERE/'runs'/task['component']/task['Q']
def done(task):
    p=taskdir(task)/'recompute_receipt.json'
    return p.exists() and read(p).get('status')=='complete'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--wait-active',action='store_true',help='Wait for manually launched priority jobs before starting more workers.')
    parser.add_argument('--limit',type=int,help='Maximum number of pending components in this invocation.')
    args=parser.parse_args();assert 1<=args.workers<=4
    points=read(PROJECT/'revised/reproduction/data/inherited/frozen_curve_delivery.json')['points']
    tasks=[]
    for p in points:
        if p['model']=='BB84_real':
            tasks.extend(dict(component=c,Q=str(p['Q'])) for c in ('bb84-registered','bb84-block'))
        elif p.get('status')=='certified_optimum':tasks.append(dict(component='werner',Q=str(p['Q'])))
    priority={.23:0,.24:1,.28:2,.22:3,.3:4,.32:5,.2:6,.25:7,.1:8}
    tasks.sort(key=lambda t:(priority.get(float(t['Q']),20),t['Q'],t['component']))
    assert len(tasks)==148
    plan=dict(scope='71 BB84 points, each registered + fixed-block; 6 Werner whole-union anchors.',
              total_components=len(tasks),tasks=tasks,
              excluded='114 Werner achieved/floating-only points have no historical global upper to replay.',
              upper_cache_reuse='Only newly computed receipts in this campaign; no historical upper endpoints imported.')
    save(HERE/'campaign_plan.json',plan)
    if args.wait_active:
        watched=[]
        for t in tasks:
            p=taskdir(t)/'recompute_receipt.json'
            if p.exists() and read(p).get('status')=='running':watched.append(t)
        while watched:
            watched=[t for t in watched if read(taskdir(t)/'recompute_receipt.json').get('status')=='running']
            save(HERE/'campaign_status.json',dict(status='waiting_for_active_priority_jobs',
                waiting=watched,completed=sum(done(t) for t in tasks),total=len(tasks)))
            if watched:time.sleep(5)
    pending=[t for t in tasks if not done(t)]
    if args.limit is not None:pending=pending[:args.limit]
    failures=[];start=time.monotonic()
    def launch(t):
        folder=taskdir(t);folder.mkdir(parents=True,exist_ok=True)
        with (folder/'recompute.log').open('ab') as log:
            proc=subprocess.run([sys.executable,'-X','utf8',str(HERE/'recompute_upper.py'),
                '--component',t['component'],'--q',t['Q']],stdout=log,stderr=subprocess.STDOUT)
        return t,proc.returncode
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        active={};iterator=iter(pending)
        for _ in range(args.workers):
            t=next(iterator,None)
            if t is not None:active[pool.submit(launch,t)]=t
        while active:
            finished,_=concurrent.futures.wait(active,timeout=5,return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in finished:
                t,returncode=fut.result();active.pop(fut)
                if returncode or not done(t):failures.append(dict(**t,returncode=returncode))
                print('FRESH UPPER COMPONENT',t,returncode,flush=True)
                nxt=next(iterator,None)
                if nxt is not None:active[pool.submit(launch,nxt)]=nxt
            save(HERE/'campaign_status.json',dict(status='running',active=list(active.values()),
                completed=sum(done(t) for t in tasks),total=len(tasks),failures=failures,seconds=time.monotonic()-start))
    complete=sum(done(t) for t in tasks)
    save(HERE/'campaign_status.json',dict(status='all_components_computed' if complete==len(tasks) else 'incomplete',
        completed=complete,total=len(tasks),failures=failures,seconds=time.monotonic()-start))
    # A numerical comparison is required in addition to successful process exits.
    subprocess.run([sys.executable,'-X','utf8',str(HERE/'compare_recompute.py')],check=True)

if __name__=='__main__':main()
