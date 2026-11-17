"""Independent input points in worker processes; only parent writes curves."""
from core import *
from bb84_policy_curves import catalog,point
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,hashlib,time

POLICIES=None
EXISTING=None


def initialize(policies,existing):
    global POLICIES,EXISTING
    POLICIES,EXISTING=policies,existing


def work(q):
    start=time.time()
    result=point(q,q,POLICIES,EXISTING,relative=1e-3,absolute=1e-8,max_cells=12000)
    result['seconds']=time.time()-start
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--permissions',choices=['real','full'],default='real');a=parser.parse_args()
    policies,existing=catalog(a.permissions)
    digest=hashlib.sha256(json.dumps(policies,sort_keys=True).encode()).hexdigest()
    atomic_json(DATA/f'bb84_curve_policies_{a.permissions}.json',dict(policies=policies,digest=digest))
    path=DATA/f'bb84_policy_curves_{a.permissions}.json';out=json.loads(path.read_text()) if path.exists() else dict(results={})
    qs=[]
    for i in range(501):
        q=i/2000;key=f'{q:.8f}_{q:.8f}';old=out['results'].get(key)
        if old and old.get('catalog_digest')==digest:continue
        qs.append(q)
    # First populate the nonzero high-noise window; cached earlier points
    # are then replayed with the finalized real catalog.
    qs.sort(key=lambda q:(q<.17,q))
    with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(policies,existing)) as pool:
        pending={pool.submit(work,q):q for q in qs}
        for future in as_completed(pending):
            r=future.result();q=r['b'];r['catalog_digest']=digest
            out['results'][f'{q:.8f}_{q:.8f}']=r
            out['scope']='Achievable saved-catalog envelope. Every displayed positive point has a whole-correlation lower certificate; point selection is numerical. Plotting certificates use a tolerance max(1e-8,1e-3 times a sampled upper estimate); individual lower/upper gaps are saved, and lower values remain valid at the cell limit. Separate anchor files certify the complete registered class.'
            atomic_json(path,out)
            print('BB dense',a.permissions,q,{k:v['lower'] for k,v in r['results'].items()},round(r['seconds'],1),flush=True)


if __name__=='__main__':main()
