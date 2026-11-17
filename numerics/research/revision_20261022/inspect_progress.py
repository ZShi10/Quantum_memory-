"""Compact read-only view of the ongoing revision calculations."""
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_bytes())
for p in [*HERE.glob('*status*.json'),*HERE.glob('bb84_campaign*.json'),
          *HERE.glob('curves_screened/status*.json'),*HERE.glob('bb84/*/*/status.json')]:
    d=read(p)
    print(p.relative_to(HERE).as_posix(),json.dumps(d,ensure_ascii=False))
for p in HERE.glob('bb84/*/*/tmp/joint_lp_*/certificate_status.json'):
    d=read(p);ed=d['ED']
    print('POLICY',str(p.relative_to(HERE)),json.dumps({k:d[k] for k in ('cells','lower','upper','argmin')},ensure_ascii=False),
          'm',ed['memory_upper_bound'],'N',ed.get('successful_ancestry_inputs'),'B',d['B'],
          'actions',len(d['strategy']))
for p in HERE.glob('bb84/*/refined_*_latest.json'):
    d=read(Path(read(p)['path']))
    print('BOUND',p.parent.name,p.stem,'closed',d['all_gaps_closed'],
          {k:[v['lower'],v['upper']] for k,v in d['results'].items() if k in ('M1','M2','M3','M4','M6','M8')})
for folder in ('bb84/0.22/full','bb84/0.23/real_xyz'):
    for p in (HERE/folder/'data/bb84_pending_policies').glob('*.json'):
        d=read(p);r=d.get('result',{});ed=d['ED']
        print('LP',folder,p.stem[:8],d['status'],'m',ed['memory_upper_bound'],
              'N',ed.get('successful_ancestry_inputs'),'B',d['B'],
              'bounds',[r.get('lower'),r.get('upper_numerical')],r.get('converged'))
