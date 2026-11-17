from pathlib import Path
import csv,json
from collections import Counter

ROOT=Path(__file__).resolve().parent;D=ROOT/'data'
blocks={e['id']:e for e in json.loads((D/'blocks.json').read_text())['entries']}
codes={e['id']:e for e in json.loads((D/'codes.json').read_text())['entries']}
for suffix in ('k0','k12'):
    result=json.loads((D/f'werner_joint_{suffix}.json').read_text())
    print('\nWerner',suffix)
    rows=list(csv.DictReader((D/f'werner_joint_{suffix}.csv').open()))
    for key,records in result['results'].items():
        selected=[r for r in rows if key==r['permissions']+'_M'+r['memory_budget']]
        gains=[float(r['joint_all_branches'])-float(r['endpoint_codes']) for r in selected]
        i=max(range(len(gains)),key=lambda i:gains[i])
        gain=gains[i];q=float(selected[i]['Q']);w=next(v for v in records if v['Q']==q)
        print(key,'max code gain',gain,'Q',q,'ED',w['ED'],'codes',[(b['code'],b['binary_rounds']) for b in w['branches'] if b['contribution']>0])
        print(' rates',[(q,next((v['rate'] for v in records if abs(v['Q']-q)<1e-9),None)) for q in (.20,.25,.28,.30,.306,.31,.32)])
for p in sorted(D.glob('bb84_[0-9]*.json')):
    r=json.loads(p.read_text());print('\n',p.name)
    for name in ('real_M1','real_M2','real_M3','full_M1','full_M2'):
        v=r['results'][name];print(name,v['lower'],v['finite_class_upper'],'ED',v['ED'])
print('\nResources',Counter((e['real'],e['M']) for e in blocks.values()))
