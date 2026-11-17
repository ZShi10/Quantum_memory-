"""Independent 130-digit checks of two non-endpoint code advantages."""
from pathlib import Path
from fractions import Fraction
from decimal import localcontext
import json,sys
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data'
sys.path.insert(0,str(ROOT.parents[1]/'manuscript'/'code'))
from linear_codes import exact_rate
from protocols import dbstep,D,d_ad
sys.path.insert(0,str(ROOT))
from algebra import compositions

def main():
    blocks={e['id']:e for e in json.loads((DATA/'blocks.json').read_text())['entries']}
    codes={e['id']:e for e in json.loads((DATA/'codes.json').read_text())['entries']}
    witnesses=[]
    for Q,ED,rounds,code in [('0.205','E00003',1,'C116'),('0.30','E05654',4,'C100')]:
        q=Fraction(Q);p=(1-Fraction(3,2)*q,q/2,q/2,q/2);ed=blocks[ED]
        powers=[]
        for typ in compositions(ed['m']):
            w=Fraction(1)
            for value,exponent in zip(p,typ):w*=value**exponent
            powers.append(w)
        raw=[sum(c*w for c,w in zip(row,powers)) for row in ed['tensor'][0]]
        ps=sum(raw);p=tuple(w/ps for w in raw)
        with localcontext() as ctx:
            ctx.prec=130
            initial=tuple(D(w) for w in p);state=initial;factor=D(ps)/ed['m']
            for _ in range(rounds):state,s=dbstep(state);factor*=s/2
            c=codes[code];selected=exact_rate(state,c['n'],c['rows'])
            endpoints=[]
            for e in codes.values():
                if e['endpoint']:endpoints.append((exact_rate(state,e['n'],e['rows']),e['id']))
            best,ci=max(endpoints)
            repetition,choice=d_ad(initial,rounds=24,final_n=64,axes=False)
            record=dict(Q=Q,ED=ED,syndrome=0,binary_rounds=rounds,code=c,
                        AD_input=[str(x) for x in state],chosen_conditional_rate=str(selected),
                        best_endpoint_same_input_rate=str(best),best_endpoint_code=ci,
                        rate_per_raw_pair=str(factor*selected),same_input_gain_per_raw_pair=str(factor*(selected-best)),
                        longer_repetition_raw_rate=str(D(ps)/ed['m']*repetition),longer_repetition_choice=choice)
            assert selected>best
            witnesses.append(record);print(record,flush=True)
    (DATA/'code_witnesses_130digit.json').write_text(json.dumps(witnesses,indent=2))

if __name__=='__main__':main()
