"""Outward-rounded validation of a complete saved pilot decision rule."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal as D
import json,time,argparse
from interval_certificate import IV,DOWN,UP,construct

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data'

def cut(coeff,t):
    t=IV(t);other=IV(1)-t
    work=coeff[:];left=[work[0]];right=[work[-1]]
    while len(work)>1:
        work=[other*a+t*b for a,b in zip(work,work[1:])]
        left.append(work[0]);right.append(work[-1])
    return left,list(reversed(right))

def restrict(coeff,a,b):
    if b<1:coeff=cut(coeff,b)[0]
    if a>0:coeff=cut(coeff,a/b)[1]
    return coeff

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--eb',default='0.20');parser.add_argument('--ep')
    parser.add_argument('--modes',nargs='+',default=['real_M2','full_M1','full_M2']);args=parser.parse_args()
    eb=F(args.eb);ep=F(args.ep or args.eb)
    result=json.loads((DATA/f'bb84_pilot_{float(eb):.5f}_{float(ep):.5f}.json').read_text())
    blocks={e['id']:e for e in json.loads((DATA/'blocks.json').read_text())['entries']}
    codes={e['id']:e for e in json.loads((DATA/'codes.json').read_text())['entries']}
    output=dict(eb=str(eb),ep=str(ep),arithmetic='60-digit outward-rounded Decimal intervals, exact rational probability polynomials, exact dyadic policy cells.',results={})
    cache={};started=time.time()
    for mode in args.modes:
        record=result['results'][mode];assert record['achievable']
        lower=D('Infinity');upper=D('Infinity');end=F(0);cells=[]
        for cell in record['policy']:
            a,b=map(F,cell['interval']);assert a==end;end=b
            key=json.dumps((cell['ED'],cell['processing']),sort_keys=True)
            if key not in cache:
                strategy=[{**p,'input_fraction':1} for p in cell['processing']]
                cache[key]=construct(eb,ep,dict(ED=cell['ED'],strategy=strategy),blocks,codes)[0]
            lo=D(0);mid=IV(0)
            for curve in cache[key]:
                joint=[restrict(p,a,b) for p in curve.joint];den=restrict(curve.den,a,b)
                bound,value=curve.bound(joint,den);lo=DOWN.add(lo,bound);mid=mid+value
            lower=min(lower,lo);upper=min(upper,mid.hi)
            cells.append(dict(interval=[str(a),str(b)],ED=cell['ED'],lower=str(lo)))
        assert end==1
        assert float(lower)>=record['lower']-2e-10
        certified=dict(lower=str(lower),sample_upper=str(upper),cells=cells,source_policy=f'bb84_pilot_{float(eb):.5f}_{float(ep):.5f}.json')
        output['results'][mode]=certified
        print(mode,'interval lower',str(lower),'cells',len(cells),'seconds',round(time.time()-started,1),flush=True)
        (DATA/f'bb84_pilot_interval_{float(eb):.5f}_{float(ep):.5f}.json').write_text(json.dumps(output,indent=2))

if __name__=='__main__':main()
