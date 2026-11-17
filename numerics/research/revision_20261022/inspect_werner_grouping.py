"""Timing diagnostic for containing-hull grouping; produces no rate claim."""
from pathlib import Path
import argparse,inspect,json,sys,time
from full_clifford_model import HERE,PROJECT
REV=PROJECT/'research/revision_20261006'
sys.path[:0]=[str(REV/'werner_search'),str(REV)]
from runtime_context import configure
core,_=configure(REV/'recheck_20261022/diagnostics/grouping')
from bb84_search.n8_adapter import install
install(core)
import upper_v2
from long_code_bounds import classical_frame_boxes

ap=argparse.ArgumentParser();ap.add_argument('--q',default='.285');a=ap.parse_args()
code=inspect.getsource(upper_v2.certify).split('    upper=np.zeros(9)')[0]
code+='    return p,y,M\n'
ns=dict(vars(upper_v2));exec(code,ns)
start=time.monotonic();p,y,M=ns['certify'](a.q,None,None,None,None,None)
print('PREPARE',len(M),time.monotonic()-start,flush=True)
out=[]
for B in range(4):
    row=dict(B=B,states=len(M),groups={})
    for quantum in (1e-14,1e-12,1e-10,1e-8,1e-6):
        boxes,index=classical_frame_boxes(p,quantum)
        row['groups'][str(quantum)]=len(boxes[0])
    print(json.dumps(row),flush=True);out.append(row)
    state,success=upper_v2.step(p,p,'Z');p=state
path=HERE/'diagnostics/grouping';path.mkdir(parents=True,exist_ok=True)
(path/'result.json').write_text(json.dumps(dict(Q=a.q,statistics=out,scope='Grouping counts only; no optimizer result or production change.'),indent=2)+'\n')
