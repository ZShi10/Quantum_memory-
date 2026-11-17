"""Independent high-precision point, derivative and mean-value checks."""
from core import *
from decimal import Decimal as D,localcontext
from fractions import Fraction as F
from recurrence_v2 import MAPS
from recursive_library import PRE
import interval_jets as jets


def decimal_value(b,p,t,record):
    b,p=D(str(b)),D(str(p));z=min(b,p)*t
    initial=[1-b-p+z,b-z,z,p-z];state=initial[:];yield_=D(1)
    def step(a,b,primitive):
        raw=[D(0)]*4
        for i,j,k in MAPS[primitive]:raw[k]+=a[i]*b[j]
        mass=sum(raw);return [v/mass for v in raw],mass
    ed,ad=record['ED'],record['AD']
    if ed['family']=='real_decoder_recurrence':
        for primitive in ed['path']:state,s=step(state,state,primitive);yield_=yield_*s/2
    else:
        cost=D(1)
        for _ in range(ed['r']-1):state,s=step(state,initial,ed['primitive']);cost=(cost+1)/s
        yield_=1/cost
    state=[state[i] for i in PRE[ed['final_axis']]]
    for _ in range(ad['classical_B_rounds']):state,s=step(state,state,'Z');yield_=yield_*s/2
    def h(x):return -x*x.ln()/D(2).ln() if x>0 else D(0)
    a=state[0]+state[3];b=state[1]+state[2];v=a*b;u=state[0]*state[1]+state[3]*state[2]
    first=1-sum(h(x) for x in state)+h(u)+h(v-u)-h(v)
    out,s=step(state,state,'Z');second=s/2*(1-sum(h(x) for x in out))
    return yield_*max(D(0),first,second)


def main():
    records=[]
    for path in ('','Z','X','R','C','ZR','RX','CR','ZZR','RZR','ZXRC','ZZXRXRR'):
        for axis in ('Z','X','Y'):
            records.append(dict(ED=dict(family='real_decoder_recurrence',path=path,final_axis=axis),AD=dict(classical_B_rounds=1)))
    for primitive in MAPS:
        records.append(dict(ED=dict(family='real_decoder_pumping',r=9,primitive=primitive,final_axis='X'),AD=dict(classical_B_rounds=2)))
    point_checks=derivative_checks=box_checks=mean_checks=0;largest_value_error=0.
    with localcontext() as context:
        context.prec=100;eps=D('1e-25')
        for b,p in ((.13,.19),(.24,.24)):
            for record in records:
                ts=[F(j,64) for j in (0,1,7,19,37,63,64)]
                points=jets.policy(b,p,[(t,t) for t in ts],record)
                for i,t in enumerate(ts):
                    td=D(t.numerator)/D(t.denominator);value=decimal_value(b,p,td,record)
                    assert D.from_float(float(points.v[0][i]))<=value<=D.from_float(float(points.v[1][i])),(b,p,record,t,points.v,value)
                    largest_value_error=max(largest_value_error,abs(float(value)-(points.v[0][i]+points.v[1][i])/2));point_checks+=1
                    if 0<t<1:
                        derivative=(decimal_value(b,p,td+eps,record)-decimal_value(b,p,td-eps,record))/(2*eps)
                        # High-precision central difference is a diagnostic,
                        # not the proof of the interval derivative rules.
                        assert float(points.d[0][i])-1e-18<=float(derivative)<=float(points.d[1][i])+1e-18,(record,t,points.d,derivative)
                        derivative_checks+=1
                intervals=[(F(j,16),F(j+1,16)) for j in (0,1,5,9,14,15)]
                boxes=jets.policy(b,p,intervals,record);mids=jets.policy(b,p,[((a+c)/2,(a+c)/2) for a,c in intervals],record)
                lower=jets.mean_value_lower(boxes,mids,intervals)
                assert not np.isnan(lower).any()
                for i,(a,c) in enumerate(intervals):
                    for u in (F(0),F(1,4),F(1,2),F(3,4),F(1)):
                        t=a+(c-a)*u;td=D(t.numerator)/D(t.denominator);value=decimal_value(b,p,td,record)
                        assert D.from_float(float(boxes.v[0][i]))<=value<=D.from_float(float(boxes.v[1][i]))
                        assert D.from_float(float(lower[i]))<=value,(record,a,c,lower[i],value)
                        box_checks+=1;mean_checks+=1
    out=dict(status='passed',point_checks=point_checks,derivative_diagnostics=derivative_checks,
             box_checks=box_checks,mean_value_checks=mean_checks,largest_midpoint_value_error=largest_value_error,
             reference='Independent 100-digit Decimal scalar propagation and entropy; central derivative difference at 1e-25; all four registered decoders, pumps, full terminal axes and endpoint zeros.')
    atomic_json(DATA/'interval_jet_checks.json',out);print(json.dumps(out),flush=True)


if __name__=='__main__':main()
