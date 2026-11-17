"""Independent exact-rational, outward-rounded whole-policy certificates."""
from common import *
from fractions import Fraction as F
from decimal import Decimal as D
import json,time,argparse
from libraries import code_kernels
from interval_certificate import IV,DOWN,Curve,mono,matpoly
from certify_pilot_policy import restrict
from algebra import compositions


def certify(path,blocks,kernels):
    source=json.loads(path.read_text());eb=F(str(source['eb']));ep=F(str(source['ep']))
    dest=path.with_name(path.stem+'_certified.json')
    result=json.loads(dest.read_text()) if dest.exists() else dict(eb=str(eb),ep=str(ep),results={})
    rawcache={};curvecache={};started=time.time()
    for mode,record in source['results'].items():
        if mode in result['results']:continue
        end=F(0);lower=D('Infinity');cells=[]
        for cell in record['policy']:
            a,b=map(F,cell['interval']);assert a==end and b>a;end=b
            ed=cell['ED'];e=blocks[ed]
            if ed not in rawcache:
                t=min(eb,ep);s=eb+ep
                physical=mono([[1-s,1-s+t],[eb,eb-t],[F(0),t],[ep,ep-t]],compositions(e['m']))
                rawcache[ed]=[matpoly(branch,physical) for branch in e['tensor']]
            lo=D(0)
            for item in cell['processing']:
                key=(ed,item['syndrome'],item['code'],tuple(item['keep_classes']))
                if key not in curvecache:
                    curvecache[key]=Curve(rawcache[ed][item['syndrome']],kernels[item['code']],item['keep_classes'],F(1,e['m']))
                curve=curvecache[key]
                v,_=curve.bound([restrict(p,a,b) for p in curve.joint],restrict(curve.den,a,b))
                lo=DOWN.add(lo,v)
            lower=min(lower,lo);cells.append(dict(interval=[str(a),str(b)],ED=ed,lower=str(lo)))
        assert end==1
        assert float(lower)>=record['lower']-2e-10
        result['results'][mode]=dict(lower=str(lower),cells=cells,source_policy=path.name)
        result.update(arithmetic='Exact rational polynomials; 60-digit outward-rounded Decimal; exact dyadic decision intervals.',
                      statement='Achievability for every compatible z. Does not certify the optimizer global upper.',seconds=time.time()-started)
        dest.write_text(json.dumps(result,separators=(',',':')))
        print('Certified complete BB84',source['eb'],source['ep'],mode,lower,'cells',len(cells),'seconds',round(time.time()-started,1),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--points',nargs='*');a=p.parse_args()
    blocks={e['id']:e for e in json.loads((OLD/'data/blocks.json').read_text())['entries']}
    kernels={k.entry['id']:k for k in code_kernels()}
    paths=[DATA/f'bb84_complete_{float(s.split(",")[0]):.5f}_{float(s.split(",")[1]):.5f}.json' for s in a.points] if a.points else sorted(DATA.glob('bb84_complete_*.json'))
    for path in paths:
        if not path.stem.endswith('_certified'):certify(path,blocks,kernels)


if __name__=='__main__':main()
