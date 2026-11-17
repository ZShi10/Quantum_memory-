"""Certify every observable/pilot policy with directed rounding.

The returned rules explicitly discard any branch whose cellwise lower
bound is nonpositive. This implements the positive-part lower bound used
by the numerical search; it is a fixed decision on each observed cell,
not selection from an unobserved source parameter.
"""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal as D
import hashlib,json,time
from interval_certificate import IV,DOWN,construct
from certify_pilot_policy import restrict

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data'


def main():
    blocks={e['id']:e for e in json.loads((DATA/'blocks.json').read_text())['entries']}
    codes={e['id']:e for e in json.loads((DATA/'codes.json').read_text())['entries']}
    result=dict(arithmetic='60-digit outward-rounded Decimal intervals; exact rational source, probability coefficients and dyadic cells.',
                scope='Achievability certificates for explicit cellwise rules, not all-protocol converse certificates.',results=[])
    count=0;started=time.time()
    paths=sorted(DATA.glob('bb84_pilot_[0-9]*.json'))+sorted(DATA.glob('bb84_observable_[0-9]*.json'))
    cache={};curve_cache={}
    for path in paths:
        source=json.loads(path.read_text());eb=F(str(source['eb']));ep=F(str(source['ep']))
        item=dict(source=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),eb=str(eb),ep=str(ep),results={})
        for mode,record in source['results'].items():
            if record.get('achievable') is False:continue
            if record['lower']==0:
                item['results'][mode]=dict(lower='0',rule='Discard all production blocks.',cells=[])
                continue
            if 'policy' not in record:
                item['results'][mode]=dict(reference='Fixed-strategy result; not a feedback rule.',lower_not_certified_here=record['lower'])
                continue
            lower=D('Infinity');end=F(0);cells=[];dropped=0
            for cell in record['policy']:
                a,b=map(F,cell['interval']);assert end==a and a<b;end=b
                ed=cell.get('ED',record.get('ED'));entry=blocks[ed]
                assert entry['M']<=int(mode[-1]) and (mode.startswith('full') or entry['real'])
                lo=D(0);processing=[]
                for p in cell['processing']:
                    key=json.dumps([str(eb),str(ep),ed,p],sort_keys=True)
                    if key not in curve_cache:
                        curve_cache[key]=construct(eb,ep,dict(ED=ed,strategy=[{**p,'input_fraction':1}]),blocks,codes)[0][0]
                    cellkey=(key,a,b)
                    if cellkey not in cache:
                        curve=curve_cache[key]
                        joint=[restrict(row,a,b) for row in curve.joint];den=restrict(curve.den,a,b)
                        cache[cellkey]=curve.bound(joint,den)[0]
                    branch_lower=cache[cellkey]
                    if branch_lower<=0:
                        dropped+=1;continue
                    lo=DOWN.add(lo,branch_lower);processing.append(p)
                lower=min(lower,lo)
                cells.append(dict(interval=[str(a),str(b)],ED=ed,M=entry['M'],processing=processing,lower=str(lo)))
            assert end==1
            assert float(lower)>=record['lower']-2e-10,(path.name,mode,str(lower),record['lower'])
            item['results'][mode]=dict(lower=str(lower),numerical_lower=record['lower'],cells=cells,
                                      explicitly_discarded_branches=dropped)
            count+=1
        result['results'].append(item)
        print(path.name,'policies',count,'seconds',round(time.time()-started,1),flush=True)
        (DATA/'all_policy_interval_certificates.json').write_text(json.dumps(result,indent=2))
    result.update(status='passed',positive_policies=count,seconds=time.time()-started,
                  corrected_policy_storage='Use the complete rules in this certificate. Raw search snapshots are preserved.')
    (DATA/'all_policy_interval_certificates.json').write_text(json.dumps(result,indent=2))
    print('PASSED',count,'positive policies',flush=True)


if __name__=='__main__':main()
