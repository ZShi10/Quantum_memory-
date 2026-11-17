"""Recompute Du v2 Fig.15's no-OTP code menu and its Watanabe control.

No author arrays were obtained. Werner values are formula evaluations;
BB84 uses independently certified worst-correlation values for each fixed
code, then maximizes over that code menu. No ED and no preliminary B.
"""
from core import *
from fractions import Fraction as F
from compound_terminals import values
from bb84_joint_extension import certify_policy
from bb84_recursive import prepare
import argparse,time,csv


def menu():
    codes=[k.entry for k in load_codes()]
    out=[]
    for n in range(1,8):
        for k in sorted({1,n-1}):
            if k<1:continue
            row=next(e for e in codes if e['n']==n and e['k']==k and e['d']==(n if k==1 else 2))
            out.append(dict(kind='linear',code=row['id'],n=n,mode='no_otp'))
    for kind in ('repetition','SPC'):out.append(dict(kind=kind,n=8,mode='no_otp'))
    out.append(dict(kind='Watanabe_2007',n=2,mode='no_otp_hashed'))
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['Werner','BB84'],required=True);a=parser.parse_args()
    actions=menu();ed=dict(family='real_decoder_recurrence',path='',final_axis='Z',memory_upper_bound=1)
    path=DATA/f'du_Fig15_{a.model.lower()}_recomputed.json'
    out=json.loads(path.read_text()) if path.exists() else dict(results={})
    out.update(menu=actions,reference='https://arxiv.org/pdf/2404.14733v2',figure=15,
        reference_scope='Repetition/SPC lengths 2..8 without OTP, Watanabe comparator, and optional direct one-way bypass. No quantum ED, no extra B prefix. Formula reconstruction; no author numerical array was obtained.',
        BB84_scope='Max over fixed codes of the whole-z lower certificate. Sampling only selects trial codes; this is an achievable reconstruction, not an upper certificate for every code at every Q.')
    qs=np.linspace(0,1/3,500) if a.model=='Werner' else [i/2000 for i in range(501)]
    start=time.time()
    for i,qq in enumerate(qs):
        q=float(qq);key=f'{q:.12g}'
        old=out['results'].get(key)
        if old and old['Q']==q and not (a.model=='BB84' and q>=.2 and 'sampled_menu_upper' not in old):continue
        if a.model=='Werner':
            state=np.array([[1-1.5*q,.5*q,.5*q,.5*q]]);vv=values(state,actions)[0];j=int(vv.argmax())
            row=dict(Q=q,rate=max(0.,float(vv[j])),AD=actions[j])
        elif q==0:row=dict(Q=q,lower=1.,upper_for_selected_policy=1.,AD=actions[0])
        elif q>=.2:
            pp,_=prepare(q,q,[0.],ed,0);test=values(pp,actions)[0]
            assert float(test.max())<1e-11
            row=dict(Q=q,lower=0.,sampled_menu_upper=float(test.max()),
                     reason='Explicit discard; all fixed actions also tested at compatible z=0. No universal impossibility claim.')
        else:
            ts=np.linspace(0,1,65);pp,_=prepare(q,q,ts,ed,0);vv=values(pp,actions).min(0)
            row=dict(Q=q,lower=0.,reason='discard');certificates={}
            for j in np.argsort(-vv):
                if vv[j]<=row['lower']+1e-9:continue
                action=actions[int(j)];strategy=[dict(AD=action,allocation_exact='1')]
                cert=certify_policy(q,q,ed,0,strategy,atol=1e-9,max_cells=8000)
                certificates[str(int(j))]=cert
                if cert['lower']>row['lower']:
                    row=dict(Q=q,lower=cert['lower'],upper_for_selected_policy=cert['upper'],AD=action,certificate=cert)
            row['evaluated_certificates']=certificates
        out['results'][key]=row;atomic_json(path,out)
        if i%25==0:print('Du formula reconstruction',a.model,i,'seconds',round(time.time()-start,1),flush=True)
    rows=[]
    for q,row in sorted(out['results'].items(),key=lambda pair:float(pair[0])):
        rows.append(dict(Q=row['Q'],rate=row.get('lower',row.get('rate',0.))))
    with (DATA/f'du_Fig15_{a.model.lower()}_recomputed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['Q','rate']);w.writeheader();w.writerows(rows)
    out['complete']=len(out['results'])==len(qs);atomic_json(path,out)
    print('Du complete',a.model,len(rows),round(time.time()-start,1),flush=True)


if __name__=='__main__':main()
