"""Fixed real M=1 repetition baselines; no unobserved-parameter adaptation."""
from common import *
from fractions import Fraction as F
from decimal import Decimal as D
import importlib.util,json


def main():
    spec=importlib.util.spec_from_file_location('baseline_repetition_interval',EXT/'certify_long_repetition.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    results=[]
    for q in [.10,.15,.18,.20,.21,.22,.23,.24,.245]:
        path=OLD/'data'/f'streaming_{q:.5f}_{q:.5f}.json';source=json.loads(path.read_text())['results']['real_M1']
        if source['lower']<=0:results.append(dict(Q=q,lower='0',interpretation='No positive result in this registered fixed-repetition family; not a general real-operation converse.'));continue
        assert source['parameters']['family']=='direct' and source['parameters']['axis'] in (0,1)
        ratio=mod.exact_ratios(q,source);target=D(str(source['lower']))*D('.9999');stack=[(F(0),F(1),0)];low=D('Infinity');cells=[]
        while stack:
            a,b,depth=stack.pop();v=mod.bound(ratio,a,b,source['binary_rounds'],source['final_n'])
            if v<target:
                assert depth<26;mid=(a+b)/2;stack.extend([(mid,b,depth+1),(a,mid,depth+1)]);continue
            low=min(low,v);cells.append(dict(interval=[str(a),str(b)],lower=str(v)))
        results.append(dict(Q=q,lower=str(low),fixed_protocol=source,cells=cells));print('Fixed real M1',q,str(low),'cells',len(cells),flush=True)
    (DATA/'real_M1_baseline_certified.json').write_text(json.dumps(dict(status='passed',scope='Immediate X or Z measurement, one fixed choice of pooled B prefix and final zero-syndrome repetition, followed by one-way extraction. Every lower bound covers all compatible z; no pilot identifies z at M=1.',results=results),separators=(',',':')))


if __name__=='__main__':main()
