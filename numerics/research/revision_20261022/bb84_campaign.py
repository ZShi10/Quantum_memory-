"""Sequential resumable orchestration of matched finite-family BB84 points."""
from pathlib import Path
import argparse,json,subprocess,sys,time
HERE=Path(__file__).resolve().parent

def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.writing.json')
    tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--q',nargs='+',required=True)
    ap.add_argument('--label',default='full',choices=['full','real_xyz','real_xz']);args=ap.parse_args()
    start=time.monotonic();status=HERE/f'bb84_campaign_{args.label}.json'
    for q in args.q:
        root=HERE/'bb84'/str(float(q));job=root/args.label;job.mkdir(parents=True,exist_ok=True)
        search=['binary_family_search.py','--q',q,'--label',args.label]
        if args.label.startswith('real'):search+=['--real','--axes','XZ' if args.label=='real_xz' else 'XYZ']
        upper=['full_binary_upper_grouped.py','--q',q,'--label',args.label,'--lower',str(job/'feasible.json')]
        if args.label.startswith('real'):upper+=['--catalog','real_binary_catalog','--axes','XZ' if args.label=='real_xz' else 'XYZ']
        refine=['refine_binary_upper.py','--base',str(root/f'upper_grouped_{args.label}_latest.json'),'--solve']
        for stage,command in [('search',search),('upper',upper),('refine',refine)]:
            save(status,dict(status='running',Q=q,label=args.label,stage=stage,seconds=time.monotonic()-start))
            with (job/f'campaign_{stage}.log').open('ab') as stream:
                r=subprocess.run([sys.executable,'-X','utf8',str(HERE/'run_numerical.py'),*command],stdout=stream,stderr=subprocess.STDOUT)
            if r.returncode:
                save(status,dict(status='failed',Q=q,label=args.label,stage=stage,returncode=r.returncode));return
            print('BB84 CAMPAIGN',q,args.label,stage,'complete',flush=True)
    save(status,dict(status='calculations_complete',Q=args.q,label=args.label,
        seconds=time.monotonic()-start,interpretation='Inspect every resulting gap; completed execution does not assert convergence.'))
if __name__=='__main__':main()
