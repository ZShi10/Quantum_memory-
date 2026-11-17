from pathlib import Path
from fractions import Fraction
from decimal import localcontext
import json,sys,time
from algebra import compositions
from spc_long import spc_rate_decimal
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data'
sys.path.insert(0,str(ROOT.parents[1]/'manuscript'/'code'))
from protocols import D,dbstep
blocks={e['id']:e for e in json.loads((DATA/'blocks.json').read_text())['entries']}
witnesses=json.loads((DATA/'code_witnesses_130digit.json').read_text())
longs=json.loads((DATA/'long_endpoint_check.json').read_text())['results'];output=[];start=time.time()
for w,long in zip(witnesses,longs):
    q=Fraction(w['Q']);p=(1-Fraction(3,2)*q,q/2,q/2,q/2);ed=blocks[w['ED']];m=ed['m']
    powers=[]
    for typ in compositions(m):
        value=Fraction(1)
        for x,r in zip(p,typ):value*=x**r
        powers.append(value)
    raw=[sum(int(c)*v for c,v in zip(row,powers)) for row in ed['tensor'][0]];ps=sum(raw)
    k,n=long['SPC_optimized_prefix_and_n64_choice']
    with localcontext() as ctx:
        ctx.prec=130;p=tuple(D(v)/D(ps) for v in raw);factor=D(ps)/m
        for _ in range(k):p,s=dbstep(p);factor*=s/2
        value=factor*spc_rate_decimal(p,n,precision=110)
        assert value>D(w['rate_per_raw_pair'])
        r=dict(Q=w['Q'],ED=w['ED'],binary_rounds=k,SPC_length=n,raw_rate=str(value),short_code_rate=w['rate_per_raw_pair'],
               conclusion='Longer SPC with a reoptimized prefix beats this short-code witness; it is not a counterexample to unrestricted endpoint-code optimality.')
        output.append(r);print(r,'seconds',time.time()-start,flush=True)
(DATA/'long_spc_high_precision.json').write_text(json.dumps(dict(status='passed',results=output),indent=2))
