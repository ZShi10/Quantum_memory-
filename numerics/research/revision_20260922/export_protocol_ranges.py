"""Protocols selected on consecutive evaluated grid points, not continuum optima."""
from core import *
import csv,hashlib


def main():
    policies={};records=[]
    def add(model,q,M,rate,protocol,evidence):
        tag=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()[:20]
        policies[tag]=protocol
        records.append(dict(model=model,Q=q,M=M,rate=rate,protocol=tag,evidence=evidence))
    rec=json.loads((DATA/'werner_v2_full.json').read_text())['results']
    sun=json.loads((DATA/'sun_common_full_AD.json').read_text())['results']
    for q in np.linspace(0,1/3,500):
        if q>.3201:continue
        r=next((v for v in rec if abs(v['Q']-q)<1e-14),None)
        if r is None:continue
        s=next(v for v in sun if abs(v['Q']-q)<1e-14)
        for M in (1,2,4,8):
            options=[r['results'][f'M{M}'],s['results'][f'M{min(M,2)}']]
            row=max(options,key=lambda v:v['rate'])
            add('Werner',float(q),M,max(0.,row['rate']),dict(ED=row['ED'],AD=row['AD']),
                'Numerical formula at this known input; complete-union intervals are separate anchor data.')
    bb=json.loads((DATA/'bb84_policy_curves_real.json').read_text())['results']
    catalog=json.loads((DATA/'bb84_curve_policies_real.json').read_text())['policies']
    du=json.loads((DATA/'du_Fig15_bb84_recomputed.json').read_text())['results']
    for row in sorted(bb.values(),key=lambda v:v['b']):
        if row['b']!=row['p']:continue
        q=row['b'];control=du[f'{q:.12g}'];assert control['Q']==q
        for M in (1,2,4,8):
            selected=row['results'][f'M{M}'];lower=max(selected['lower'],control['lower'])
            if lower==0:protocol=dict(action='discard')
            elif control['lower']>selected['lower'] or 'policy' not in selected:
                protocol=dict(ED=dict(family='direct_measurement',axis='Z',M=1),B=0,AD=control['AD'])
            else:protocol=catalog[selected['policy']]
            add('BB84',q,M,lower,protocol,'Whole-compatible-correlation lower bound for the fixed displayed policy.')
    ranges=[]
    for model in ('Werner','BB84'):
        for M in (1,2,4,8):
            active=None
            for r in sorted([v for v in records if v['model']==model and v['M']==M],key=lambda v:v['Q']):
                if active and active['protocol']==r['protocol']:
                    active.update(Q_last=r['Q'],grid_points=active['grid_points']+1,
                                  min_rate=min(active['min_rate'],r['rate']),max_rate=max(active['max_rate'],r['rate']))
                else:
                    active=dict(model=model,M=M,Q_first=r['Q'],Q_last=r['Q'],grid_points=1,
                                protocol=r['protocol'],min_rate=r['rate'],max_rate=r['rate'],evidence=r['evidence'])
                    ranges.append(active)
    with (DATA/'selected_protocol_grid_ranges.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(ranges[0]));w.writeheader();w.writerows(ranges)
    atomic_json(DATA/'selected_protocol_grid_ranges.json',dict(ranges=ranges,policies=policies,
        scope='A range groups consecutive evaluated grid points selecting the same construction. It is not a proof that this protocol is optimal at unevaluated interior Q, and is not an all-generator optimum classification.'))
    print('Selected protocol grid runs:',len(ranges),'protocol definitions:',len(policies),flush=True)


if __name__=='__main__':main()
