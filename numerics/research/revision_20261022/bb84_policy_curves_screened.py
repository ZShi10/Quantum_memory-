"""Dense fixed-portfolio envelopes, with rigorous known-source screening."""
from fractions import Fraction as F
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from full_clifford_model import HERE,bootstrap
from full_binary_search import read,save

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--partition',type=int,default=0);ap.add_argument('--parts',type=int,default=1);a=ap.parse_args()
    source=HERE/'curves/policies.json';digest=hashlib.sha256(source.read_bytes()).hexdigest();policies=read(source)
    job=HERE/'curves_screened';job.mkdir(parents=True,exist_ok=True);(job/'policies.json').write_bytes(source.read_bytes())
    core,engine,_,_=bootstrap(job);start=time.monotonic();count=0
    ordered=sorted(policies,key=lambda k:(policies[k]['memory'],-len(policies[k]['strategy']),k))
    for index in range(161):
        if index%a.parts!=a.partition:continue
        q=F(16,100)+F(index,2000);name=str(q).replace('/','-')+'.json';file=job/'points'/name
        if file.exists():assert read(file)['policy_sha256']==digest
        else:
            prior=HERE/'curves/points'/name;cached={}
            if prior.exists():
                old=read(prior);assert old['policy_sha256']==digest;cached=old['results']
            incumbents={family:[0.]*9 for family in ('real_xyz','full')};winners={family:[None]*9 for family in incumbents};rates={}
            for key in ordered:
                p=policies[key];m=p['memory'];scope=p['permissions']
                if key in cached:
                    result=cached[key];cert=result['certificate'];result=dict(result,status='full_interval',reused_from=str(prior))
                else:
                    tests=[F(0),F(1,8),F(1,4),F(1,2),F(1)]
                    value=engine.curve(float(q),float(q),[(t,t) for t in tests],p['ED'],p['B'],p['strategy']).v[1]
                    upper=max(0.,float(np.min(value)));threshold=min(incumbents[f][m] for f in scope)
                    if upper<=threshold:
                        rates[key]=dict(status='screened',known_source_upper=upper,test_t=list(map(str,tests)),
                            screening_target=threshold,note='Fixed-policy upper only; no omitted lower is plotted as zero.')
                        continue
                    cert=engine.certify_policy(float(q),float(q),p['ED'],p['B'],p['strategy'],atol=2e-8,max_cells=6000)
                    result=dict(status='full_interval',lower=max(0.,cert['lower']),certificate=cert,known_source_upper=upper)
                rates[key]=result;lo=result['lower']
                for family in scope:
                    for budget in range(m,9):
                        if lo>incumbents[family][budget]:incumbents[family][budget]=lo;winners[family][budget]=key
            row=dict(Q=str(q),policy_sha256=digest,results=rates,lower_envelopes=incumbents,winners=winners,
                scope='Maximum of an explicit feasible portfolio. Every reported lower covers all hidden t; screened policies have valid fixed-policy known-source uppers below an already achieved lower. Not a whole-family optimum.')
            save(file,row)
        count+=1;save(job/f'status_{a.partition}.json',dict(status='running',completed=count,partition=a.partition,parts=a.parts,Q=str(q),seconds=time.monotonic()-start))
        print('SCREENED CURVE',index+1,161,str(q),flush=True)
    save(job/f'status_{a.partition}.json',dict(status='complete',completed=count,partition=a.partition,parts=a.parts,seconds=time.monotonic()-start))
if __name__=='__main__':main()
