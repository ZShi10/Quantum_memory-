"""Find plotted-point candidate gains; no continuum claim or fresh upper implied."""
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
points=json.loads((ROOT/'revised/reproduction/data/inherited/frozen_curve_delivery.json').read_bytes())['points']
report=[]
for model in ('Werner','BB84_real'):
    ps=sorted((p for p in points if p['model']==model),key=lambda p:p['Q'])
    for m in range(2,9):
        gains=[]
        for p in ps:
            a=p['results'][f'M{m-1}'];b=p['results'][f'M{m}']
            lo=b['lower'];prev=a['lower'];u=a.get('upper')
            if lo<1e-5 or lo-prev<=max(1e-8,.01*prev):continue
            gains.append(dict(Q=p['Q'],higher_lower=lo,lower_memory_achieved=prev,
                achieved_difference=lo-prev,ratio=lo/prev if prev>0 else None,
                historical_certified_gap=(lo-u if u is not None else None)))
        if gains:
            best=max(gains,key=lambda v:v['achieved_difference'])
            report.append(dict(model=model,memory_step=[m-1,m],selected_points=len(gains),
                Q_points=[v['Q'] for v in gains],largest_absolute_achieved_gain=best,
                status='Existing sampled feasible values; historical uppers when present. Not an interval certificate.'))
(HERE/'sampled_memory_gains.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
for r in report:
    print(json.dumps(r,ensure_ascii=False))
