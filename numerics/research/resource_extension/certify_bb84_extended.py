from common import *
from fractions import Fraction as F
from decimal import Decimal as D
import json,time,hashlib
from interval_certificate import Curve,DOWN,mono,matpoly
from certify_pilot_policy import restrict
from graph_character import bb84_polynomials
from algebra import compositions
from kernels import load


def main():
    graphs={e['id']:e for e in json.loads((DATA/'graphs.json').read_text())['entries']}
    kernels={k.entry['id']:k for k in load()};started=time.time()
    for path in sorted(DATA.glob('bb84_[0-9]*.json')):
        out=DATA/path.name.replace('bb84_','bb84_certified_')
        if out.exists():continue
        src=json.loads(path.read_text());eb=F(str(src['eb']));ep=F(str(src['ep']))
        rawcache={};curves={};bounds={};result={};count=0
        for mode,record in src['results'].items():
            if record['lower']<=0:result[mode]=dict(lower='0',policy=[]);continue
            minimum=D('Infinity');end=F(0);policy=[]
            for cell in record['policy']:
                a,b=map(F,cell['interval']);assert a==end and b>a;end=b
                ed=cell['ED'];entry=src['protocol_library'][ed]
                assert entry['M']<=int(mode[-1]) and (mode.startswith('full') or entry['real'])
                if ed not in rawcache:
                    if entry['source']=='extended_graph':
                        base={**graphs[entry['graph_id']],'axis_permutation':entry['axis_permutation']}
                        rawcache[ed]=[bb84_polynomials(base,eb,ep,exact=True)]
                    else:
                        t=min(eb,ep);s=eb+ep
                        components=[[1-s,1-s+t],[eb,eb-t],[F(0),t],[ep,ep-t]]
                        physical=mono(components,compositions(entry['m']))
                        rawcache[ed]=[matpoly(branch,physical) for branch in entry['tensor']]
                total=D(0);processing=[]
                for item in cell['processing']:
                    key=(ed,json.dumps(item,sort_keys=True))
                    if key not in curves:
                        curves[key]=Curve(rawcache[ed][item['syndrome']],kernels[item['code']],item['keep_classes'],F(1,entry['m']))
                    point=(key,a,b)
                    if point not in bounds:
                        c=curves[key];bounds[point]=c.bound([restrict(v,a,b) for v in c.joint],restrict(c.den,a,b))[0]
                    value=bounds[point]
                    if value>0:total=DOWN.add(total,value);processing.append(item)
                minimum=min(minimum,total);policy.append(dict(interval=[str(a),str(b)],ED=ed,M=entry['M'],processing=processing,lower=str(total)))
            assert end==1 and float(minimum)>=record['lower']-5e-10,(path.name,mode,minimum,record['lower'])
            result[mode]=dict(lower=str(minimum),policy=policy);count+=len(policy)
        saved=dict(status='passed',source=path.name,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   arithmetic='Exact rational probability polynomials and 60-digit outward-rounded intervals. Complete explicit pilot rule over the entire compatible z interval.',
                   eb=str(eb),ep=str(ep),results=result)
        out.write_text(json.dumps(saved,separators=(',',':')))
        print('Certified',path.name,'cells',count,'seconds',round(time.time()-started,1),flush=True)


if __name__=='__main__':main()
