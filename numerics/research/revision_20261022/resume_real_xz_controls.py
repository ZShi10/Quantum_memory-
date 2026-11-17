"""Resume incomplete real/XZ controls without rerunning completed controls."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_bytes())
def call(stage,args):
    status=HERE/'bb84_controls_resume_status.json'
    status.write_text(json.dumps(dict(stage=stage,args=args),indent=2)+'\n')
    with (HERE/'bb84_controls_resume.log').open('ab') as log:
        r=subprocess.run([sys.executable,'-X','utf8',str(HERE/'run_numerical.py'),*args],stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError((stage,r.returncode))
    print('REAL XZ COMPLETE',stage,flush=True)
def main():
    for q in ('.2','.23'):
        root=HERE/'bb84'/str(float(q));pointer=root/'refined_real_xz_latest.json'
        if pointer.exists() and read(Path(read(pointer)['path'])).get('complete'):continue
        feasible=root/'real_xz/feasible.json'
        if not(feasible.exists() and read(feasible).get('complete')):
            call(q+'_search',['binary_family_search_efficient.py','--q',q,'--label','real_xz'])
        base=root/'upper_grouped_real_xz_latest.json'
        if not(base.exists() and read(Path(read(base)['path'])).get('complete')):
            call(q+'_upper',['full_binary_upper_grouped.py','--q',q,'--catalog','real_binary_catalog','--axes','XZ','--label','real_xz','--lower',str(feasible)])
        args=['refine_binary_upper_v2.py','--base',str(base),'--solve','--dual-only']
        if q=='.2':args+=['--cache-source',str(root/'refined_real_xz/b82cb2aa99e505377aa0')]
        call(q+'_refine',args)
    (HERE/'bb84_controls_resume_status.json').write_text(json.dumps(dict(stage='complete'))+'\n')
if __name__=='__main__':main()
