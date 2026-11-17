"""Compare fresh bounds and rebuild the five target windows from fresh uppers.

Archived upper values are read only for comparison AFTER recomputation. None is
substituted for a missing fresh component. The complete flag requires all 148
components and the stated optimization gaps, not merely matching file hashes.
"""
from pathlib import Path
from fractions import Fraction as F
import json, math
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
def read(p):return json.loads(p.read_bytes())
def get(component,q):
    p=HERE/'runs'/component/q/'recompute_receipt.json'
    if not p.exists():return None
    x=read(p)
    return x if x.get('status')=='complete' else None
def up(x):
    y=float(x);return math.nextafter(y,math.inf) if F.from_float(y)<x else y
def main():
    points=read(PROJECT/'revised/reproduction/data/inherited/frozen_curve_delivery.json')['points']
    rows=[];pending=[];fresh={}
    for p in points:
        model=p['model'];q=str(p['Q'])
        if model=='BB84_real':
            a,b=get('bb84-registered',q),get('bb84-block',q)
            if a is None or b is None:
                pending.append(dict(model=model,Q=q,registered_complete=a is not None,block_complete=b is not None));continue
            registered_refined=get('bb84-registered-refined',q)
            if registered_refined is not None:
                assert registered_refined['historical_upper_values_used'] is False
                assert registered_refined['fresh_base_sha256']==a['output_sha256']
                a=registered_refined
            refined=get('bb84-block-refined',q)
            if refined is not None:
                assert refined['historical_upper_values_used'] is False
                assert refined['fresh_base_sha256']==b['output_sha256']
                b=refined
            bounds={m:max(a['results'][m]['upper'],b['results'][m]['upper']) for m in p['results']}
            relative=1e-4
        elif p.get('status')=='certified_optimum':
            a=get('werner',q)
            if a is None:pending.append(dict(model=model,Q=q));continue
            bounds={m:r['upper'] for m,r in a['results'].items()};relative=1e-5
        else:continue
        fresh[model,q]=bounds
        for m,upper in bounds.items():
            old=p['results'][m];lower=float(old['lower']);tol=max(1e-11,relative*lower)
            assert upper+1e-14>=lower,(model,q,m,'upper below previous feasible lower')
            rows.append(dict(model=model,Q=q,memory=int(m[1:]),fresh_upper=upper,
                historical_upper_for_comparison_only=old['upper'],difference=upper-old['upper'],
                lower_reference=lower,gap=upper-lower,tolerance=tol,
                declared_gap_met=upper-lower<=tol+1e-18))
    windows=[]
    targets={'BB23_m5':'0.0005','BB23_m6':'0.0008','BB24_m6':'0.00001','BB24_m7':'0.00008','BB24_m8':'0.000095'}
    for name,t in targets.items():
        cert=read(PROJECT/'revised/reproduction/reproduction/extensions/data'/(name+'_lower_certificate.json'))
        spec=cert['specification'];Q=F(spec['Q']);q=str(float(Q));m=int(name.split('_m')[1])
        if ('BB84_real',q) not in fresh:continue
        assert cert['status']=='verified_continuous_fixed_policy_lower'
        L=F(spec['target']);assert all(F(v['lower'])>=L for v in cert['leaves'])
        base=fresh['BB84_real',q]
        terms=dict(finite=F.from_float(base[f'M{m-1}']),stopping=5*F.from_float(base['M3']),
                   B_tail=F(1,2**17),pump_tail=(1-Q)**160/2)
        U=max(terms.values());T=F(t)
        windows.append(dict(policy=name,Q=str(Q),memory=m,fresh_extended_upper=up(U),
            fresh_extended_upper_exact=str(U),lower=float(L),target=float(T),
            strict_window_verified=bool(U<T<=L),active_upper_term=next(k for k,v in terms.items() if v==U),
            source='Fresh registered and fixed-block uppers plus analytic extension; full-z lower certificate.'))
    issues=[r for r in rows if not r['declared_gap_met']]
    result=dict(status='complete' if not pending and not issues and all(w['strict_window_verified'] for w in windows) else 'incomplete',
        completed_noise_points=len(rows)//8,total_upper_bearing_noise_points=77,
        fresh_upper_rows=len(rows),total_upper_rows=616,missing=pending,gap_issues=issues,
        fresh_target_windows=windows,rows=rows,
        scope='All inherited whole-family uppers, not lower-only Werner drawing points. Missing components are never filled from historical bounds.')
    (HERE/'comparison.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','missing','gap_issues')},indent=2))
if __name__=='__main__':main()
