"""Independent 60-digit outward certificates for selected net-rate witnesses."""
from core import *
from recursive_library import PRE
from ad_modes import OTPKernel
from interval_certificate import IV,LN2
from certify_werner_witnesses import xlog,kernel_rate,spc_rate
from prove_single_round_counterexample import ivpower,positive
from fractions import Fraction as F
from math import comb
import argparse,time

LABELS=((0,0),(1,0),(1,1),(0,1))


def pair_enumeration(p,q):
    raw=[IV(0) for _ in range(4)]
    for i,(x,z) in enumerate(LABELS):
        for j,(a,b) in enumerate(LABELS):
            if x==a:
                idx=LABELS.index((x,z^b));raw[idx]=raw[idx]+p[i]*q[j]
    mass=sum(raw,IV(0))
    return [v/mass for v in raw],mass


def prepare(q,record):
    p=[IV(1-F(3,2)*q),IV(q/2),IV(q/2),IV(q/2)]
    if record['family'] in ('real_decoder_recurrence','real_decoder_pumping'):
        from recurrence_v2 import MAPS
        def step(a,b,primitive):
            raw=[IV(0) for _ in range(4)]
            for i,j,k in MAPS[primitive]:raw[k]=raw[k]+a[i]*b[j]
            s=sum(raw,IV(0));return [v/s for v in raw],s
        cost=IV(1);raw=p[:]
        if record['family']=='real_decoder_recurrence':
            for primitive in record['path']:p,s=step(p,p,primitive);cost=cost*2/s
        else:
            for _ in range(record['r']-1):p,s=step(p,raw,record['primitive']);cost=(cost+IV(1))/s
        return [p[i] for i in PRE[record['final_axis']]],IV(1)/cost,cost
    if record['family']=='homogeneous_recurrence':
        cost=IV(1)
        for basis in record['path']:
            state=[p[i] for i in PRE[basis]]
            p,s=pair_enumeration(state,state);cost=cost*2/s
    else:
        raw=[p[i] for i in PRE[record['preparation_axis']]]
        p=raw[:];prob=IV(1);attempt_cost=IV(1)
        for _ in range(1,record['r']):
            attempt_cost=attempt_cost+prob
            p,s=pair_enumeration(p,raw);prob=prob*s
        cost=attempt_cost/prob
    return [p[i] for i in PRE[record['final_axis']]],IV(1)/cost,cost


def otp_rate(p,n,k,counts,mode):
    a=p[0]+p[3];b=p[1]+p[2]
    h0=-(xlog(p[0])+xlog(p[3])-xlog(a))/a if a.lo>0 else IV(0)
    h1=-(xlog(p[1])+xlog(p[2])-xlog(b))/b if b.lo>0 else IV(0)
    words=[ivpower(a,n-w)*ivpower(b,w) for w in range(n+1)]
    entropies=[-xlog(u)+u*(h0*(n-w)+h1*w) for w,u in enumerate(words)]
    gross=IV(0);compression=IV(0);accepted=[]
    for j,row in enumerate(counts):
        ps=sum((int(c)*u for c,u in zip(row,words) if c),IV(0))
        h=sum((int(c)*v for c,v in zip(row,entropies) if c),IV(0))+xlog(ps)
        gain=n*ps-h
        gross=gross+positive(gain)
        compression=compression-xlog(ps)
        if gain.lo>0:accepted.append(j)
    cost=IV(n-k) if mode=='otp_plain' else compression
    return positive(gross-cost)/IV(n),accepted,cost/IV(n)


def otp_for_ad(p,ad,kernels):
    n=ad['n']
    if ad['kind']=='linear':
        entry=kernels[ad['code']].entry
        return otp_rate(p,n,entry['k'],OTPKernel(entry).counts,ad['mode'])
    if ad['kind']=='SPC':
        counts=[[comb(n,w) if w%2==s else 0 for w in range(n+1)] for s in (0,1)]
        return otp_rate(p,n,n-1,counts,ad['mode'])
    # Repetition coset entropy and compression need the multiplicity of each
    # weight class. Build integer aggregate terms without exponential cosets.
    a=p[0]+p[3];b=p[1]+p[2]
    h0=-(xlog(p[0])+xlog(p[3])-xlog(a))/a if a.lo>0 else IV(0)
    h1=-(xlog(p[1])+xlog(p[2])-xlog(b))/b if b.lo>0 else IV(0)
    gross=compression=IV(0);accepted=[]
    for w in range(n//2+1):
        u=ivpower(a,n-w)*ivpower(b,w);v=ivpower(a,w)*ivpower(b,n-w)
        ps=u+v
        h=-xlog(u)-xlog(v)+u*(h0*(n-w)+h1*w)+v*(h0*w+h1*(n-w))+xlog(ps)
        mult=comb(n,w)//(2 if 2*w==n else 1)
        gain=n*ps-h
        gross=gross+mult*positive(gain);compression=compression-mult*xlog(ps)
        if gain.lo>0:accepted.append(w)
    cost=IV(n-1) if ad['mode']=='otp_plain' else compression
    return positive(gross-cost)/IV(n),accepted,cost/IV(n)


def repetition_all(p,n):
    a=p[0]+p[3];b=p[1]+p[2];u=p[0]-p[3];v=p[1]-p[2]
    total=IV(0)
    for w in range(n//2+1):
        aa=ivpower(a,n-w)*ivpower(b,w);bb=ivpower(a,w)*ivpower(b,n-w)
        cc=ivpower(u,n-w)*ivpower(v,w);dd=ivpower(u,w)*ivpower(v,n-w)
        raw=[(aa+cc)/IV(2),(bb+dd)/IV(2),(bb-dd)/IV(2),(aa-cc)/IV(2)]
        raw=[IV(max(0,x.lo),max(0,x.hi)) for x in raw]
        mass=aa+bb;gain=mass+sum((xlog(x) for x in raw),IV(0))-xlog(mass)
        total=total+comb(n,w)//(2 if 2*w==n else 1)*positive(gain)
    return total/IV(n)


def witness(q,record,kernels):
    p,y,cost=prepare(F(str(q)),record['ED'])
    ad=record['AD']
    if ad is None:return dict(lower='0',upper='0',status='zero')
    for _ in range(ad['classical_B_rounds']):
        p,s=pair_enumeration(p,p);y=y*s/IV(2)
    if ad['kind']=='Watanabe_2007':
        out,s=pair_enumeration(p,p)
        a=p[0]+p[3];b=p[1]+p[2];v=a*b;u=p[0]*p[1]+p[3]*p[2]
        rest=v-u;rest=IV(max(0,rest.lo),max(0,rest.hi))
        first=IV(1)+sum((xlog(x) for x in p),IV(0))+xlog(v)-xlog(u)-xlog(rest)
        second=s/IV(2)*(IV(1)+sum((xlog(x) for x in out),IV(0)))
        value=IV(max(0,first.lo,second.lo),max(0,first.hi,second.hi));flags=otp_cost=None
    elif ad['mode']=='no_otp':
        if ad['kind']=='linear':value=kernel_rate(p,kernels[ad['code']])
        elif ad['kind']=='SPC':value=spc_rate(p,ad['n'])
        else:value=repetition_all(p,ad['n'])
        flags=otp_cost=None
    else:
        value,flags,otp_cost=otp_for_ad(p,ad,kernels)
    result=y*value
    discrepancy=abs(float(result.lo)-record['rate'])
    assert discrepancy<max(2e-14,abs(record['rate'])*1e-6),(q,record,result.lo,discrepancy)
    return dict(lower=str(result.lo),upper=str(result.hi),
                ED_cost_interval=[str(cost.lo),str(cost.hi)],
                positive_OTP_syndrome_classes=flags,
                OTP_cost_per_AD_input=None if otp_cost is None else [str(otp_cost.lo),str(otp_cost.hi)],
                status='positive' if result.lo>0 else 'zero_or_unresolved',
                arithmetic='Exact rational Q; 16-event Pauli propagation; 60-digit outward interval entropy and all yields.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',default='recursive_werner_real_d7_n128.json')
    parser.add_argument('--q',nargs='*',type=float);args=parser.parse_args()
    src=json.loads((DATA/args.source).read_text());kernels={k.entry['id']:k for k in load_codes()}
    dest=DATA/(Path(args.source).stem+'_certified.json')
    saved=json.loads(dest.read_text()) if dest.exists() else dict(results={})
    cache={}
    for existing in DATA.glob('werner_v2_*_certified.json'):
        for row in json.loads(existing.read_text()).get('results',{}).values():
            if not all(k in row for k in ('Q','ED','AD','lower','upper')):continue
            identity=(row['Q'],json.dumps(row['ED'],sort_keys=True),json.dumps(row['AD'],sort_keys=True))
            cache[identity]={k:v for k,v in row.items() if k not in ('Q','budget','ED','AD')}
    start=time.time()
    for r in src['results']:
        if args.q and r['Q'] not in args.q:continue
        for budget,rec in r['results'].items():
            key=f"{r['Q']:.12g}_{budget}"
            if key in saved['results']:continue
            identity=(r['Q'],json.dumps(rec['ED'],sort_keys=True),json.dumps(rec['AD'],sort_keys=True))
            if identity in cache:cert=cache[identity]
            else:cert=witness(r['Q'],rec,kernels);cache[identity]=cert
            saved['results'][key]=dict(Q=r['Q'],budget=budget,ED=rec['ED'],AD=rec['AD'],**cert)
            atomic_json(dest,saved)
            print(key,cert['status'],cert['lower'][:32],'seconds',round(time.time()-start,1),flush=True)


if __name__=='__main__':main()
