"""Replay published dense-curve witnesses over the entire hidden interval."""
from fractions import Fraction as F
from pathlib import Path
import argparse,hashlib,json,time
from full_clifford_model import HERE,PROJECT,bootstrap
def read(p):return json.loads(p.read_bytes())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--q',required=True);ap.add_argument('--all-policies',action='store_true');a=ap.parse_args()
    q=F(a.q)
    sources=[parent/folder for folder in ('curves_final','curves_extended')
             for parent in (HERE,PROJECT/'evidence/revision_20261022')]
    source=next(p for p in sources if (p/'status.json').exists() and read(p/'status.json')['stage']=='complete')
    policies=read(source/'policies.json');old=read(source/'points'/(str(q).replace('/','-')+'.json'))
    assert old['policy_sha256']==hashlib.sha256((source/'policies.json').read_bytes()).hexdigest()
    selected=sorted(policies if a.all_policies else {k for values in old['winners'].values() for k in values if k is not None})
    job=HERE/'dense_replays'/str(q).replace('/','-');core,engine,model,ops=bootstrap(job)
    from fast_entropy import install
    install();out={};start=time.monotonic()
    for key in selected:
        p=policies[key];previous=old['results'].get(key,{})
        tolerance=max(2e-10,max(0.,previous.get('lower',0.))*1e-5)
        c=engine.certify_policy(float(q),float(q),p['ED'],p['B'],p['strategy'],atol=tolerance,max_cells=64000)
        out[key]=dict(ED=p['ED'],B=p['B'],strategy=p['strategy'],certificate=c,
            previous_lower=previous.get('lower'),recomputed_lower=max(0.,c['lower']))
        print('DENSE WITNESS',str(q),key,c['lower'],c['upper'],c['converged'],flush=True)
    result=dict(Q=str(q),complete=True,policy_sha256=old['policy_sha256'],results=out,seconds=time.monotonic()-start,
        scope='Fresh whole-hidden-interval evaluation of selected explicit curve policies, or all frozen policies when requested. No supplied lower or upper is used as a proof; previous values only select the requested witness and numerical tolerance.')
    (job/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('DENSE REPLAY COMPLETE',str(job/'result.json'),flush=True)
if __name__=='__main__':main()
