"""Independent outward certificates for every selected Werner protocol."""
from common import *
from fractions import Fraction as F
import json,time
import numpy as np
from interval_certificate import IV
from libraries import code_kernels
from rates import CodeKernel
from search_werner import werner_blocks
from certify_werner_witnesses import witness,kernel_rate,spc_rate,repetition
from assemble_results import protocol_key


def main():
    current=json.loads((DATA/'current_results.json').read_text());selected={protocol_key(r):r for pid,r in current['protocols'].items() if pid.startswith('W_')}
    lib=json.loads((DATA/'werner_complete_library.json').read_text());lc={e['id']:e for e in lib['entries']}
    weights={m:np.load(DATA/f'werner_states_m{m}.npz')['weights'] for m in range(1,9)}
    small={e['id']:e for e in werner_blocks(json.loads((OLD/'data/blocks.json').read_text())['entries'])}
    old=json.loads((EXT/'data/six_state.json').read_text())['protocol_library'];small.update(old)
    if (DATA/'larger_werner.json').exists():small.update(json.loads((DATA/'larger_werner.json').read_text())['protocol_library'])
    kernels={k.entry['id']:k for k in code_kernels()};codes={e['id']:e for e in json.loads((OLD/'data/codes.json').read_text())['entries']}
    for r in selected.values():
        for b in r['branches']:
            cid=b.get('code')
            if b['contribution']>0 and cid and cid not in kernels:kernels[cid]=CodeKernel(codes[cid])
    dest=DATA/'envelope_werner_certificates.json';saved=json.loads(dest.read_text()) if dest.exists() else dict(results={});start=time.time()
    for key,r in selected.items():
        if key in saved['results']:continue
        if r['ED'] in lc:cert=witness(r,lc[r['ED']],weights,kernels)
        else:
            q=F(str(r['Q']));m=r['m'];powers=[(1-F(3,2)*q)**(m-j)*(q/2)**j for j in range(m+1)];total=IV(0);branches=[]
            for b in r['branches']:
                if b['contribution']<=0:continue
                if r['ED'].startswith('restart_pump'):
                    A=(1-q)**m;B=q**m;V=(1-2*q)**m;den=A+B
                    raw=[A+V,A-V,B,B];ps=2*den
                    cost=(1+sum((1-q)**j+q**j for j in range(1,m)))/den;factor=IV(1/cost)
                else:
                    e=small[r['ED']]
                    if 'positive_werner_weights' in e:counts=e['positive_werner_weights']
                    elif 'werner_weights' in e:counts=e['werner_weights'][b['syndrome']]
                    else:counts=None
                    if counts is not None:raw=[sum((int(c)*v for c,v in zip(row,powers)),F()) for row in counts]
                    else:raw=[sum((int(c)*(1-2*q)**j for j,c in enumerate(row)),F())/(2**(m+1)) for row in e['character_weights']]
                    assert min(raw)>=0;ps=sum(raw);factor=IV(ps/F(m))
                p=[IV(x/ps) for x in raw]
                if b.get('canonicalize_bit') and p[1].lo+p[2].lo>.5:p=[p[j] for j in (1,0,3,2)]
                for _ in range(b['binary_rounds']):
                    a,x,y,z=p;out=[a*a+z*z,x*x+y*y,2*x*y,2*a*z];mass=sum(out,IV(0));factor=factor*mass/IV(2);p=[v/mass for v in out]
                kind=b.get('kind','linear')
                gain=kernel_rate(p,kernels[b['code']]) if kind=='linear' else spc_rate(p,b['n']) if kind=='SPC' else repetition(p,b['n'])
                v=factor*gain;total=total+v;branches.append(dict(syndrome=b['syndrome'],lower=str(v.lo),upper=str(v.hi)))
                assert abs(float(v.lo)-b['contribution'])<max(5e-16,b['contribution']*2e-6),(r['ED'],q,v.lo,b)
            cert=dict(Q=r['Q'],ED=r['ED'],m=m,M=r['M'],lower=str(total.lo),upper=str(total.hi),branches=branches)
        saved['results'][key]=cert;saved.update(status='passed',arithmetic='Exact rational input and success cost, outward-rounded 60-digit interval ED/AD and entropy arithmetic.',seconds=time.time()-start)
        dest.write_text(json.dumps(saved,separators=(',',':')))
        print('Envelope certificate',r['Q'],r['ED'],str(cert['lower'])[:30],'count',len(saved['results']),flush=True)


if __name__=='__main__':main()
