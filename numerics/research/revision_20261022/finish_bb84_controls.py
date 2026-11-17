"""Sequential matched operation controls after the current real/XYZ search."""
from pathlib import Path
import json, subprocess, sys, time
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_bytes())
def save(d):
    p=HERE/'bb84_controls_status.json';tmp=p.with_suffix('.writing.json')
    tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def call(stage,args):
    save(dict(status='running',stage=stage,args=args))
    with (HERE/'bb84_controls.log').open('ab') as stream:
        r=subprocess.run([sys.executable,'-X','utf8',str(HERE/'run_numerical.py'),*args],stdout=stream,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError((stage,r.returncode))
    print('CONTROL COMPLETE',stage,flush=True)
def bounds(q,label):
    root=HERE/'bb84'/str(float(q))
    call(f'{q}_{label}_upper',['full_binary_upper_grouped.py','--q',q,'--catalog','real_binary_catalog',
         '--axes','XZ' if label=='real_xz' else 'XYZ','--label',label,'--lower',str(root/label/'feasible.json')])
    args=['refine_binary_upper_v2.py','--base',str(root/f'upper_grouped_{label}_latest.json'),'--solve','--dual-only']
    if q=='.23' and label=='real_xyz':
        args+=['--cache-source',str(root/'refined_real_xyz/37766aeccf66691f5435')]
    call(f'{q}_{label}_refine',args)
def main():
    waiting=HERE/'bb84/0.23/real_xyz/feasible.json'
    save(dict(status='waiting',dependency=str(waiting)))
    while not(waiting.exists() and read(waiting).get('complete') and read(waiting).get('execution')):time.sleep(10)
    bounds('.23','real_xyz')
    for q in ('.17','.2','.23'):
        call(f'{q}_real_xz_search',['binary_family_search_efficient.py','--q',q,'--label','real_xz'])
        bounds(q,'real_xz')
    save(dict(status='calculations_complete',note='Read each enclosure before claiming its gap is closed.'))
if __name__=='__main__':main()
