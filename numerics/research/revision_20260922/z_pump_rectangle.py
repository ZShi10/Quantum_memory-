"""Fast certified correlation bounds for Z-only schedules and Du codes.

The bit law and raw consumption are independent of BB84 correlation z.
Conditional phase-noise degradation gives a rigorous rectangular lower
bound without differentiating a length-128 entropy expression.
"""
from core import *
from fractions import Fraction as F
from functools import lru_cache
from compound_terminals import kernels,physical
from interval_modes import components,signed_mul
from long_code_bounds import LongBounds
from spc_interval_dynamic import all_bounds
import stable_intervals as iv
import heapq,hashlib


def supported(ed,B,strategy):
    if ed['final_axis']!='Z':return False
    if ed['family']=='real_decoder_pumping':
        if ed['primitive']!='Z':return False
        r=ed['r']
    elif ed['family']=='real_decoder_recurrence' and set(ed['path'])<=set('Z'):
        r=2**len(ed['path'])
    else:return False
    return r*2**B<=512 and all(v['AD']['kind']!='Watanabe_2007' for v in strategy)


@lru_cache(None)
def fixed_parameters(b,p,ed_json,B):
    b,p=F(str(b)),F(str(p));ed=json.loads(ed_json);a=1-b
    if ed['family']=='real_decoder_pumping':
        r=ed['r'];mass=a**r+b**r;y=mass/(1+sum(a**j+b**j for j in range(1,r)))
        u=a**r/mass;v=b**r/mass
    else:
        r=1;u=a;v=b;y=F(1)
        for _ in ed['path']:
            s=u*u+v*v;y*=s/2;u,v=u*u/s,v*v/s;r*=2
    for _ in range(B):
        s=u*u+v*v;y*=s/2;u,v=u*u/s,v*v/s;r*=2
    return b,p,u,v,r,y


def proxy(b,p,ed,B,intervals):
    b,p,u,v,r,y=fixed_parameters(b,p,json.dumps(ed,sort_keys=True),B);rows=[]
    def min_abs(left,right):
        return F(0) if left<=0<=right else min(abs(left),abs(right))
    for left,right in intervals:
        zl=min(b,p)*F(left);zh=min(b,p)*F(right)
        c0=min_abs(1-b-2*p+2*zl,1-b-2*p+2*zh)/(1-b)
        c1=min_abs(b-2*zh,b-2*zl)/b if b else F(0)
        c0,c1=c0**r,c1**r
        rows.append([u*(1+c0)/2,v*(1+c1)/2,v*(1-c1)/2,u*(1-c0)/2])
    val=[[(iv.rational(x)) for x in row] for row in rows]
    return (np.array([[x[0] for x in row] for row in val]),np.array([[x[1] for x in row] for row in val])),iv.rational(y)


def evaluate(b,p,ed,B,strategy,intervals):
    state,y=proxy(b,p,ed,B,intervals);lo=np.zeros(len(intervals));hi=lo.copy();long=None
    ns=sorted({v['AD']['n'] for v in strategy if v['AD']['kind']=='SPC' and v['AD']['mode']=='no_otp'})
    spc=all_bounds(state,np.array(ns),np.ones((len(intervals),len(ns)),bool)) if ns else None
    for item in strategy:
        a=item['AD'];n=a['n'];mode=a['mode'];kind=a['kind']
        if kind=='linear':
            k=kernels()[a['code']];kk=k if mode=='no_otp' else physical(a['code'])
            gain,cost=components(iv.monomial_bounds(state,np.asarray(compositions(n))),kk)
            gross=iv.positive_sum((np.maximum(0,gain[0]),np.maximum(0,gain[1])))
            rate=gross if mode=='no_otp' else iv.sub(gross,iv.rational(F(n-k.k,n)) if mode=='otp_plain' else cost)
        elif kind=='SPC' and mode=='no_otp':
            i=ns.index(n);rate=spc[0][:,i],spc[1][:,i]
        else:
            if long is None:long=LongBounds(state,max(v['AD']['n'] for v in strategy))
            rate=(long.spc(n,False) if kind=='SPC' else long.repetition(n,False))[('no_otp','otp_plain','otp_compressed').index(mode)]
        weighted=signed_mul(rate,iv.rational(F(item['allocation_exact'])))
        lo,hi=iv.add((lo,hi),weighted)
    return signed_mul((lo,hi),y)


def certify(b,p,ed,B,strategy,atol=2e-9,max_cells=50000):
    assert supported(ed,B,strategy)
    cache=certificate_path(b,p,ed,B,strategy);cache.parent.mkdir(exist_ok=True)
    if cache.exists():
        old=json.loads(cache.read_text())
        if old['gap']<=atol:return old
    heap=[];counter=0;incumbent=np.inf;argmin=None
    def add(intervals):
        nonlocal counter,incumbent,argmin
        low,_=evaluate(b,p,ed,B,strategy,intervals)
        mid=[((a+c)/2,(a+c)/2) for a,c in intervals]
        _,hi=evaluate(b,p,ed,B,strategy,mid)
        for (a,c),l,h in zip(intervals,low,hi):
            heapq.heappush(heap,(float(l),counter,a,c));counter+=1
            if h<incumbent:incumbent=float(h);argmin=float((a+c)/2)
    _,end=evaluate(b,p,ed,B,strategy,[(F(0),F(0)),(F(1),F(1))]);incumbent=float(end.min());argmin=float(end.argmin())
    add([(F(0),F(1))])
    while counter<max_cells and incumbent-heap[0][0]>atol:
        children=[]
        for _ in range(min(8,len(heap))):
            _,_,a,c=heapq.heappop(heap);mid=(a+c)/2;children.extend(((a,mid),(mid,c)))
        add(children)
    result=dict(lower=float(heap[0][0]),upper=incumbent,gap=float(incumbent-heap[0][0]),argmin=argmin,
                cells=counter,converged=incumbent-heap[0][0]<=atol,atol=atol,
                arithmetic='Exact rational Z-schedule bit law and yield; conditional-phase-degradation lemma; outward complete-code entropy bounds over a partition of the whole correlation interval.')
    atomic_json(cache,result);return result


def certificate_path(b,p,ed,B,strategy):
    geometry={k:ed[k] for k in ('family','primitive','r','path','final_axis') if k in ed}
    actions=[dict(AD=v['AD'],allocation_exact=v['allocation_exact']) for v in strategy]
    identity=json.dumps(dict(b=b,p=p,ED=geometry,B=B,strategy=actions),sort_keys=True)
    return DATA/'z_rectangle_certificates'/(hashlib.sha256(identity.encode()).hexdigest()+'.json')


def verify():
    from bb84_joint_extension import numerical_worst
    checks=0;rng=np.random.default_rng(9944)
    for b,p in ((.18,.18),(.12,.21)):
        ed=dict(family='real_decoder_pumping',primitive='Z',r=6,final_axis='Z',memory_upper_bound=1)
        strategy=[dict(AD=dict(kind='SPC',n=128,mode='no_otp'),allocation_exact='1')]
        intervals=[(F(0),F(1))]+[(F(i,8),F(i+1,8)) for i in range(8)]
        ll,_=evaluate(b,p,ed,0,strategy,intervals)
        from bb84_recursive import prepare
        from compound_terminals import values
        for (a,c),l in zip(intervals,ll):
            ts=np.linspace(float(a),float(c),33);state,y=prepare(b,p,ts,ed,0)
            vv=values(state,[strategy[0]['AD']])[:,0]*y
            assert np.all(vv>=l-1e-13),(b,p,a,c,l,vv.min());checks+=len(vv)
    b=p=.18;ed=dict(family='real_decoder_pumping',primitive='Z',r=6,final_axis='Z',memory_upper_bound=1)
    cert=certify(b,p,ed,0,strategy,atol=2.5e-9)
    atomic_json(DATA/'z_pump_rectangle_checks.json',dict(status='passed',comparisons=checks,certificate=cert))
    print('Z rectangle',checks,cert,flush=True)


if __name__=='__main__':verify()
