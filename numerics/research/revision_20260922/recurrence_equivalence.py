"""Exact state/yield/schedule symmetries for p_X=p_Z source families."""
from core import *
from recurrence_v2 import generate


def key(record):
    axis=record['final_axis'];family=record['family']
    if family=='real_decoder_pumping':
        r=record['r'];primitive=record['primitive']
        if r<=2:
            family='real_decoder_recurrence';path='' if r==1 else primitive
        else:
            if primitive=='C':primitive='X'
            if primitive=='R' and axis=='X':axis='Z'
            return ('pumping',primitive,r,axis)
    else:path=record['path']
    letters=list(path)
    for i,c in enumerate(letters):
        if c!='R':letters[i]='Z';break
    else:
        if axis=='X':axis='Z'
    return ('recurrence',''.join(letters),axis)


def representatives(records,symmetric=True):
    if not symmetric:return np.arange(len(records))
    seen={};out=[]
    for i,record in enumerate(records):
        k=key(record);rep=seen.setdefault(k,i)
        assert record['memory_upper_bound']==records[rep]['memory_upper_bound']
        out.append(rep)
    return np.array(out)


def main():
    errors=[];counts={}
    for permission in ('real','full'):
        for b in (.1,.2,.24):
            z=b*np.array([0,.001,.02,.37,.7,1]);p=np.stack((1-2*b+z,b-z,z,b-z),-1)
            states,y,M,records=generate(p,7,permission);rep=representatives(records)
            err=max(float(np.max(np.abs(states-states[rep]))),float(np.max(np.abs(y-y[rep]))));errors.append(err)
            assert err<2e-13,(permission,b,err)
            counts[permission]=dict(original=len(rep),representatives=int(np.sum(rep==np.arange(len(rep)))))
    out=dict(status='passed',max_state_or_yield_difference=max(errors),classes=counts,
             scope='Analytic equivalences for every p_X=p_Z source; all finite registered memory schedules agree exactly. Applied only to symmetric BB84/Werner, never arbitrary b!=p.')
    atomic_json(DATA/'recurrence_equivalence_checks.json',out);print(json.dumps(out),flush=True)


if __name__=='__main__':main()
