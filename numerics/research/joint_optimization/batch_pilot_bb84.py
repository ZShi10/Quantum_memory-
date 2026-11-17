from pathlib import Path
import json,time
from search_pilot_bb84 import Oracle
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data'
for path in sorted(DATA.glob('bb84_[0-9]*.json')):
    old=json.loads(path.read_text());eb,ep=old['eb'],old['ep'];target=DATA/f'bb84_pilot_{eb:.5f}_{ep:.5f}.json'
    if target.exists():continue
    start=time.time();oracle=Oracle(eb,ep);results={}
    for mode,M in [('real',1),('real',2),('real',3),('real',4),('full',1),('full',2)]:
        r=oracle.certify(mode,M);results[f'{mode}_M{M}']=r
        print('pilot',eb,ep,mode,M,r['lower'],r['upper'],'seconds',round(time.time()-start,1),flush=True)
    result=dict(eb=eb,ep=ep,results=results,seconds=time.time()-start,
                scope='Finite m<=4,n<=6 library; negligible pilot stage identifies z for full M>=1 or real M>=2. Real M=1 is an optimistic state-informed oracle only. Continuous numerical enclosures.')
    target.write_text(json.dumps(result,indent=2))
