"""Tighter certificate at the small nonzero real-class BB84 rate."""
from common import *
from fractions import Fraction as F
import json,time
from libraries import complete_short_entries,code_kernels
from complete_bb84_search import make_oracle
from certify_complete_bb84 import certify
from certify_finite_upper import bb_upper


def main():
    q=.23;path=DATA/'bb84_complete_0.23000_0.23000.json';src=json.loads(path.read_text())
    if src.get('real_refinement_complete'):return
    es=complete_short_entries(True);ks=code_kernels();o=make_oracle(q,q,es,ks);started=time.time()
    result=o.certify('real',2,atol=2e-11,max_cells=3000)
    assert result['converged'];print('Refined real BB84',result['lower'],result['upper'],'cells',result['cells'],'seconds',time.time()-started,flush=True)
    history=DATA/'history';history.mkdir(exist_ok=True)
    paths=[path,path.with_name(path.stem+'_certified.json'),DATA/'bb84_upper_0.23000_0.23000.json']
    for p in paths:
        dst=history/p.name
        if not dst.exists():dst.write_bytes(p.read_bytes())
    src['results']['real_M2']=result;src['refined_real_atol']=2e-11;path.write_text(json.dumps(src,separators=(',',':')))
    for p in paths[1:]:
        d=json.loads(p.read_text());d['results'].pop('real_M2',None);p.write_text(json.dumps(d,separators=(',',':')))
    blocks={e['id']:e for e in json.loads((OLD/'data/blocks.json').read_text())['entries']}
    certify(path,blocks,{k.entry['id']:k for k in ks})
    bb_upper(F(23,100),F(23,100),es,ks,src)
    src['real_refinement_complete']=True;path.write_text(json.dumps(src,separators=(',',':')))


if __name__=='__main__':main()
