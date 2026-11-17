"""Controlled AD comparisons on fixed ED outputs, and fidelity-first control."""
from core import *
from fractions import Fraction as F
from compound_terminals import actions,values
from recurrence_v2 import prepare
from sun_common_objective import outputs,library
from certify_werner_joint_union import terminal
from bb84_fixed import physical_kernel
import stable_intervals as iv
import time


def masks(menu):
    code={k.entry['id']:k for k in load_codes()};out={name:[] for name in
       ('hash','repetition_noOTP','endpoint_noOTP','short_noOTP','short_all','Du_all','endpoint_all_plus_W','all')}
    for i,a in enumerate(menu):
        kind,n,mode=a['kind'],a['n'],a['mode'];k=code[a['code']].k if kind=='linear' else (1 if kind=='repetition' else n-1)
        no=mode=='no_otp';rep=kind=='repetition' or (kind=='linear' and k==1)
        end=kind in ('repetition','SPC') or (kind=='linear' and k in (1,n-1))
        flags=dict(hash=kind=='linear' and n==1,repetition_noOTP=rep and no,endpoint_noOTP=end and no,
                   short_noOTP=kind=='linear' and no,short_all=kind=='linear',Du_all=kind!='Watanabe_2007',
                   endpoint_all_plus_W=end or kind=='Watanabe_2007',all=True)
        for name,take in flags.items():
            if take:out[name].append(i)
    return out


def optimize_fixed(state,y,menu,groups,K=16):
    states=[];scales=[];p=state.copy();scale=float(y)
    for B in range(K+1):
        states.append(p.copy());scales.append(scale);p,s=b_step(p);scale*=float(s)/2
    pp=np.array(states);vv=values(pp,menu)*np.array(scales)[:,None];out={}
    for name,ids in groups.items():
        local=vv[:,ids];at=np.unravel_index(local.argmax(),local.shape);index=ids[at[1]]
        out[name]=dict(rate=max(0.,float(local[at])),B=int(at[0]),AD=menu[index])
    return out


def exact_sun_output(q,record):
    e=library()[record['library_index']];m=e['m'];q=F(str(q));a=1-F(3,2)*q;c=q/2
    raw=[sum(F(int(w))*a**(m-j)*c**j for j,w in enumerate(row)) for row in e['weights']]
    mass=sum(raw);return [raw[i]/mass for i in record['frame']],mass/m


def certify_sun(q,record,choice,kernels,physical):
    from certify_bb84_recursive import step
    pp,y=exact_sun_output(q,record);vals=[iv.rational(x) for x in pp]
    state=tuple(np.array([[v[j] for v in vals]]) for j in (0,1));scale=iv.rational(y)
    for _ in range(choice['B']):state,s=step(state,state,'Z');scale=iv.scale(iv.mul(scale,(s[0][0],s[1][0])),F(1,2))
    val=terminal(state,choice['AD'],kernels,physical);result=iv.mul(scale,(val[0][0],val[1][0]))
    return dict(lower=float(result[0]),upper=float(result[1]),ED=record,AD=choice['AD'],B=choice['B'])


def main():
    menu=actions(128);groups=masks(menu);path=DATA/'werner_fixed_ED_ablations.json'
    out=json.loads(path.read_text()) if path.exists() else dict(results={});cache={}
    source=json.loads((DATA/'werner_v2_full.json').read_text())
    for row in source['results']:
        if row['Q'] not in (.1,.2,.25,.28,.3,.32):continue
        for M in (1,2,4,8):
            key=f'{row["Q"]:.12g}_M{M}'
            if key in out['results']:continue
            ed=row['results'][f'M{M}']['ED'];identity=(row['Q'],json.dumps(ed,sort_keys=True));start=time.time()
            if identity not in cache:
                q=row['Q'];p,y=prepare(np.array([1-1.5*q,q/2,q/2,q/2]),ed)
                cache[identity]=optimize_fixed(p,y,menu,groups)
            out['results'][key]=dict(Q=row['Q'],budget=M,ED=ed,terminals=cache[identity])
            out['scope']='Same selected ED and final axis; each named classical menu independently reoptimized over B<=16. Floating controlled comparison; does not reoptimize ED separately for each restricted classical menu.'
            atomic_json(path,out);print('AD ablation',key,{k:v['rate'] for k,v in cache[identity].items()},round(time.time()-start,1),flush=True)
    # The old intermediate-code example is re-evaluated under the stronger
    # current menu rather than silently promoted to an enlarged separation.
    state=np.array([39205,1681,1681,11357])/53924
    out['old_code_example']={f'Bcap{K}':optimize_fixed(state,1.,menu,groups,K) for K in (0,16)}
    atomic_json(path,out)
    # Joint ED selection versus maximum output fidelity: same m=3,
    # zero ED flag, all final frames, same full classical menu and B cap.
    q=.28;pp,yy,mm,rr=outputs(q);ids=[i for i,r in enumerate(rr) if r.get('m')==3]
    values_by_id={};fidelity={}
    for i in ids:
        values_by_id[i]=optimize_fixed(pp[i],yy[i],menu,groups)['all']
        exact,_=exact_sun_output(q,rr[i]);fidelity[i]=max(exact)
    highest=max(fidelity.values());ties=[i for i in ids if fidelity[i]==highest]
    joint=max(ids,key=lambda i:values_by_id[i]['rate']);first=max(ties,key=lambda i:values_by_id[i]['rate'])
    kk={k.entry['id']:k for k in load_codes()};physical={name:physical_kernel(k) for name,k in kk.items()}
    result=dict(Q=q,m=3,M=2,joint=certify_sun(q,rr[joint],values_by_id[joint],kk,physical),
                fidelity_first=certify_sun(q,rr[first],values_by_id[first],kk,physical),
                highest_fidelity_exact=str(highest),joint_fidelity_exact=str(fidelity[joint]),
                tie_count=len(ties),scope='All Sun-size m=3 zero-flag ED weight classes and six final Pauli frames, common full n<=128 AD menu and B<=16. Maximum-fidelity ties resolved in favor of final key rate. Exact rational fidelity comparison; outward chosen rates.')
    atomic_json(DATA/'fidelity_control_current_menu.json',result);print('Fidelity control',result,flush=True)


if __name__=='__main__':main()
