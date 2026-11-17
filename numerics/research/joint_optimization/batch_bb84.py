"""Serial, restartable BB84 scan; each point includes all finite candidates."""
from pathlib import Path
import json,time
from search_bb84 import load_libraries,search_point

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data'

def main():
    started=time.time()
    for symmetric,points in [
        (True,[(q,q) for q in (.10,.15,.18,.21,.22,.23,.24,.245)]),
        (False,[(.05,.20),(.10,.30),(.18,.25),(.12,.35),(.20,.28)])]:
        entries,kernels,upperkernels=load_libraries(symmetric)
        for eb,ep in points:
            path=DATA/f'bb84_{eb:.5f}_{ep:.5f}.json'
            if path.exists():
                old=json.loads(path.read_text())
                if old.get('resource_compiler')=='terminal-graph-v1':continue
            result=search_point(entries,kernels,upperkernels,eb,ep,tolerance=2e-8)
            result['resource_compiler']='terminal-graph-v1'
            result['scope']='m<=4 compiled single Clifford block; n<=6 single-round linear codes; fixed syndrome retention; fixed classical input fractions. Continuous z minimization with numerical Bernstein enclosures.'
            path.write_text(json.dumps(result,indent=2,default=lambda v:v.item()))
            print('Saved',path.name,'total seconds',round(time.time()-started,1),flush=True)

if __name__=='__main__':main()
