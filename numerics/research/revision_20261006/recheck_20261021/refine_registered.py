"""Fresh legal-source refinements of residual registered-family uppers.

Only the new campaign's completed partition is consumed. Floating values
select test parameters; every accepted bound is freshly evaluated outward
over the full signed classical menu. No historical upper is imported.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse,hashlib,json,math,os,shutil,sys,time
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;REV=HERE.parent;PROJECT=REV.parents[1]


def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.writing.json');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)


def run(q):
    original=HERE/'runs/bb84-registered'/q/'recompute_receipt.json'
    base=read(original);assert base['status']=='complete' and not base['historical_upper_values_used']
    upperfile=Path(base['output']);up=read(upperfile);assert up['complete']
    src=read(base['discovery_source'])
    task=HERE/'runs/bb84-registered-refined'/q;task.mkdir(parents=True,exist_ok=True)
    receipt=task/'recompute_receipt.json'
    if receipt.exists() and read(receipt).get('status')=='complete':return read(receipt)
    shutil.copyfile(original.parent/'manifest.json',task/'manifest.json')
    sys.path.insert(0,str(REV/'bb84_search'))
    import run_point
    core,engine,_,_=run_point.bootstrap(task)
    from compound_menu_bounds import dual_upper
    from bb84_recursive import prepare
    import numpy as np
    lower={m:float(v['lower']) for m,v in src['results'].items()}
    ts=[F(0)]+[F(1,2**k) for k in [24,20,16,12,10,8,6,4]]+[F(x,8) for x in range(1,9)]
    candidates=[]
    for r in up['rounds']:
        for v in r['unresolved']:
            m=v['ED']['memory_upper_bound'];goal=lower[f'M{m}']+max(1e-11,1e-4*lower[f'M{m}'])
            if v['upper']>goal:candidates.append(dict(key=f"{v['original_index']}_B{r['B']}",B=r['B'],ED=v['ED'],base_upper=v['upper'],goal=goal))
    identity=dict(Q=q,fresh_base_sha256=sha(upperfile),script_sha256=sha(__file__),
                  candidates=candidates,test_parameters=list(map(str,ts)))
    save(task/'refinement_spec.json',identity)
    rows={};start=time.monotonic()
    for v in candidates:
        path=task/(v['key']+'.json')
        if path.exists():
            row=read(path);assert row['fresh_base_sha256']==identity['fresh_base_sha256']
        else:
            state,y=prepare(float(q),float(q),np.array(list(map(float,ts))),v['ED'],v['B'])
            floated=engine.known_max(state,128)*y
            row=dict(**v,fresh_base_sha256=identity['fresh_base_sha256'],upper=v['base_upper'],bounds=[])
            for i in np.argsort(floated):
                proposal=dict(ED=v['ED'],classical_B_rounds=v['B'],samples=[str(ts[i])],sampled_dual=['1'])
                cert=dual_upper(float(q),float(q),proposal,128)
                row['bounds'].append(cert);row['upper']=min(row['upper'],cert['upper'])
                if row['upper']<=v['goal']:break
            save(path,row)
        rows[v['key']]=row
        print('FRESH REGISTERED REFINEMENT',q,v['key'],row['upper'],v['goal'],flush=True)
    rounds=[]
    for r in up['rounds']:
        residual=[]
        for v in r['unresolved']:
            key=f"{v['original_index']}_B{r['B']}"
            residual.append(dict(memory=v['ED']['memory_upper_bound'],upper=rows[key]['upper'] if key in rows else v['upper']))
        bounds={}
        for m in range(1,9):
            key=f'M{m}'
            cap=math.nextafter(lower[key]+up['tolerance'],math.inf)
            bounds[key]=min(r['upper'][key],max([cap]+[v['upper'] for v in residual if v['memory']<=m]))
        rounds.append(dict(B=r['B'],upper=bounds))
    results={m:dict(lower=lo,upper=max(r['upper'][m] for r in rounds)) for m,lo in lower.items()}
    result=dict(**identity,complete=True,rounds=rounds,results=results,refinements=rows,
                historical_upper_values_used=False,seconds=time.monotonic()-start,
                scope='All fresh-base candidates retained; minimum with newly computed legal-source all-action bounds on selected residuals. No floating value is used as a bound.')
    output=task/'result.json';save(output,result)
    record=dict(status='complete',component='bb84-registered-refined',Q=q,results=results,
                historical_upper_values_used=False,fresh_base_sha256=sha(upperfile),
                output=str(output),output_sha256=sha(output),seconds=result['seconds'])
    save(receipt,record);print(json.dumps(record,indent=2),flush=True)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--q',required=True)
    run(str(float(parser.parse_args().q)))
