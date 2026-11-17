from common import *
from streaming_clifford import *
from itertools import permutations
import json,time
import numpy as np


def solve_combination(rows,target):
    pivots={}
    for i,row in enumerate(rows):
        mask=1<<i
        while row:
            j=row.bit_length()-1
            if j in pivots:row^=pivots[j][0];mask^=pivots[j][1]
            else:pivots[j]=(row,mask);break
    answer=0
    while target:
        j=target.bit_length()-1;target^=pivots[j][0];answer^=pivots[j][1]
    return answer


def dense_check(compiled,m):
    size=1<<m;u=np.eye(size,dtype=complex);indices=np.arange(size)
    for op,a,*rest in compiled['gates']:
        if op=='h':
            lo=indices[(indices>>a&1)==0];hi=lo|(1<<a);v=u[lo].copy();w=u[hi].copy()
            u[lo]=(v+w)/np.sqrt(2);u[hi]=(v-w)/np.sqrt(2)
        elif op=='s':u[(indices>>a&1)!=0]*=1j
        else:
            b=rest[0];dest=indices^(((indices>>a)&1)<<b);u=u[dest]
    masks=[r['original_generator_mask'] for r in compiled['measurements']];largest=0.
    for j,p in enumerate(compiled['generators']):
        x=p&(size-1);z=p>>m;mat=np.zeros_like(u)
        mat[indices^x,indices]=[(1j)**((x&z).bit_count())*(-1)**((int(i)&z).bit_count()) for i in indices]
        coeff=solve_combination(masks,1<<j);phase=0;bits=0
        for i,row in enumerate(compiled['measurements']):
            if coeff>>i&1:phase^=row['offset'];bits^=1<<row['wire']
        expected=np.diag([(-1)**(phase+((int(i)&bits).bit_count())) for i in indices])
        error=float(np.max(np.abs(u@mat@u.conj().T-expected)));largest=max(largest,error)
        assert error<3e-12,(m,p,error)
    return largest


def main():
    source=json.loads((OLD/'data/graph_measurements.json').read_text());out={};count=dense=0;maximum=0.;started=time.time()
    for m in range(1,5):
        for key in source[f'full_{m}']:
            generators=list(map(int,key.split('_')));real=all(quadratic(p,m)==0 for p in generators)
            c=compile_measurement(generators,m,real=real);assert c['M']<=2
            maximum=max(maximum,dense_check(c,m));dense+=1
            out[f'{m}:{key}']=c;count+=1
        print('Verified complete terminal measurements',m,'total',count,'seconds',round(time.time()-started,1),flush=True)
    # Adversarial arrival orders check the per-cut construction, independent
    # of the optimizing order. Include complex local bases and real YY rows.
    for m in (2,3,4):
        keys=list(source[f'full_{m}'])[::max(1,len(source[f'full_{m}'])//10)]
        for key in keys:
            generators=list(map(int,key.split('_')));real=all(quadratic(p,m)==0 for p in generators)
            for order in permutations(range(m)):
                c=compile_measurement(generators,m,real=real,order=list(order))
                maximum=max(maximum,dense_check(c,m));dense+=1
    entries=json.loads((OLD/'data/blocks.json').read_text())['entries'];mapping={};changed=[]
    for e in entries:
        key=f"{e['m']}:"+'_'.join(map(str,rref(e['checks']+[e['logical_Z']],2*e['m'])))
        c=out[key];assert not e['real'] or c['real']
        mapping[e['id']]=dict(measurement_key=key,M=c['M'],real=c['real'])
        if c['M']<e['M']:changed.append(dict(ED=e['id'],m=e['m'],real=e['real'],old_M=e['M'],M=c['M']))
    result=dict(status='passed',measurements=out,entries=mapping,changed=changed,
                complete_measurements=count,dense_checks=dense,max_operator_error=maximum,
                scope='All complete stabilizer bases for m<=4, exact phases and independently constructed dense unitaries. The same minimal memory is achieved by H/CNOT alone whenever the basis is real.')
    (DATA/'small_memory_exact.json').write_text(json.dumps(result,separators=(',',':')))
    print(json.dumps({k:v for k,v in result.items() if k not in ('measurements','entries','changed')}));print('Changed ED entries',len(changed))


if __name__=='__main__':main()
