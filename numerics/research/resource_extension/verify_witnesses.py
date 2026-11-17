from common import *
from fractions import Fraction as F
from decimal import localcontext
import json,time
sys.path.insert(0,str(ROOT.parents[1]/'manuscript/code'))
from protocols import D,dbstep,drepeat,dhash
from linear_codes import exact_rate
from spc_long import spc_rate_decimal


def raw_exact(e,q):
    m=e['m']
    if 'werner_weights' in e:
        powers=[(1-F(3,2)*q)**(m-w)*(q/2)**w for w in range(m+1)]
        return [[sum(int(c)*v for c,v in zip(row,powers)) for row in branch] for branch in e['werner_weights']]
    powers=[(1-2*q)**w for w in range(m+1)]
    return [[sum(int(c)*v for c,v in zip(row,powers))/2**(m+1) for row in e['character_weights']]]


def main():
    src=json.loads((DATA/'six_state.json').read_text());codes={e['id']:e for e in json.loads((DATA/'codes.json').read_text())['entries']}
    targets=[];seen=set()
    for item in src['results']:
        if item['Q'] not in (.15,.23,.28,.30,.305,.31,.315,.32):continue
        for key in ('M1','M2','M4','M8'):
            r=item['results'][key];signature=(r['Q'],r['ED'])
            if r['rate']<=0 or signature in seen:continue
            seen.add(signature);targets.append(r)
    results=[];started=time.time()
    for record in targets:
        entry=src['protocol_library'][record['ED']];raw=raw_exact(entry,F(str(record['Q'])))
        with localcontext() as context:
            context.prec=150;total=D(0);branches=[]
            for b in record['branches']:
                if b['contribution']<=0:continue
                initial=raw[b['syndrome']];ps=sum(initial);state=tuple(D(v)/D(ps) for v in initial);yield_=D(ps)/entry['m']
                for _ in range(b['binary_rounds']):state,s=dbstep(state);yield_*=s/2
                if b['kind']=='SPC':r=yield_*spc_rate_decimal(state,b['n'],precision=130)
                elif b['kind']=='repetition':
                    final,s=drepeat(state,b['n']);r=yield_*s/b['n']*dhash(final)
                else:
                    c=codes[b['code']];r=yield_*exact_rate(state,c['n'],c['rows'])
                assert abs(float(r)-b['contribution'])<=max(3e-17,abs(float(r))*3e-7),(record['Q'],record['ED'],b,float(r))
                total+=r;branches.append({**b,'rate_130digit':str(r)})
            results.append(dict(Q=record['Q'],ED=record['ED'],rate_130digit=str(total),branches=branches))
        (DATA/'witnesses_130digit.json').write_text(json.dumps(dict(results=results),indent=2))
        print('Verified',record['Q'],record['ED'],str(total)[:44],'seconds',round(time.time()-started,1),flush=True)
    (DATA/'witnesses_130digit.json').write_text(json.dumps(dict(status='passed',results=results),indent=2))


if __name__=='__main__':main()
