"""Final consistency audit and reproducibility manifest (no new search)."""
from pathlib import Path
from decimal import Decimal as D
from collections import Counter
import ast,csv,hashlib,json,math

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data';WORK=ROOT.parents[1]


def read(name):return json.loads((DATA/name).read_text())


def finite(obj):
    if isinstance(obj,float):assert math.isfinite(obj),obj
    elif isinstance(obj,dict):
        for value in obj.values():finite(value)
    elif isinstance(obj,list):
        for value in obj:finite(value)


def main():
    syntax=0
    for path in ROOT.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'),filename=str(path));syntax+=1
    files=list(DATA.glob('*.json'))
    for path in files:finite(json.loads(path.read_text()))
    blocks=read('blocks.json');codes=read('codes.json')
    assert len(blocks['entries'])==5659 and len(codes['entries'])==124
    assert sum(codes['labeled_counts'].values())==3283
    assert sum(e['indecomposable'] for e in codes['entries'])==26
    blockmap={e['id']:e for e in blocks['entries']}
    statuses={}
    for name in ('foundation_verification','measurement_cut_verification','bernstein_verification',
                 'frame_verification','streaming_verification','pumping_witnesses_high_precision',
                 'graph_family_witnesses_130digit','long_spc_high_precision',
                 'single_round_counterexamples_interval','all_policy_interval_certificates'):
        value=read(name+'.json');assert value['status']=='passed',name;statuses[name]='passed'
    certs=read('all_policy_interval_certificates.json');certmap={v['source']:v for v in certs['results']}
    certificate_cells=0;dropped=0
    for source,item in certmap.items():
        assert item['sha256']==hashlib.sha256((DATA/source).read_bytes()).hexdigest()
        for mode,record in item['results'].items():
            if not record.get('cells'):continue
            end='0'
            for cell in record['cells']:
                assert cell['interval'][0]==end;end=cell['interval'][1]
                entry=blockmap[cell['ED']]
                assert entry['M']<=int(mode[-1]) and (mode.startswith('full') or entry['real'])
                assert D(cell['lower'])>=D(record['lower']);certificate_cells+=1
            assert end=='1'
            dropped+=record.get('explicitly_discarded_branches',0)
    groups={kind:{} for kind in ('fixed','observable','pilot')};gaps={};rows=[]
    patterns={'fixed':'bb84_[0-9]*.json','observable':'bb84_observable_[0-9]*.json','pilot':'bb84_pilot_[0-9]*.json'}
    for kind,pattern in patterns.items():
        local=[]
        for path in sorted(DATA.glob(pattern)):
            value=json.loads(path.read_text());point=(value['eb'],value['ep']);groups[kind][point]=value
            for mode,record in value['results'].items():
                lower=record['lower'];upper=record.get('finite_class_upper',record.get('upper'))
                assert lower>=0 and lower<=upper+1e-10,(path.name,mode)
                gap=upper-lower;local.append((gap,path.name,mode))
                threshold=5.03e-9 if kind=='pilot' else 2.01e-8
                assert gap<=threshold,(path.name,mode,gap)
                certified=certmap.get(path.name,{}).get('results',{}).get(mode,{})
                rows.append(dict(kind=kind,eb=value['eb'],ep=value['ep'],permissions=mode.split('_')[0],
                                 M=int(mode[-1]),achievable=record.get('achievable',True),
                                 numerical_lower=lower,numerical_upper=upper,
                                 outward_rounded_lower=certified.get('lower','')))
            for permission in ('real','full'):
                previous=0
                for M in range(1,5):
                    record=value['results'].get(f'{permission}_M{M}')
                    if record is None:continue
                    assert record['lower']+3e-8>=previous,(kind,point,permission,M)
                    previous=record['lower']
            for M in range(1,5):
                real=value['results'].get(f'real_M{M}');full=value['results'].get(f'full_M{M}')
                if real is not None and full is not None:assert full['lower']+3e-8>=real['lower']
        gaps[kind]=max(local)
        assert len(groups[kind])==14
    assert groups['fixed'].keys()==groups['observable'].keys()==groups['pilot'].keys()
    for point in groups['fixed']:
        for smaller,larger in (('fixed','observable'),('observable','pilot')):
            a=groups[smaller][point]['results'];b=groups[larger][point]['results']
            for mode in a.keys()&b.keys():
                assert b[mode]['lower']+4e-8>=a[mode]['lower'],(point,mode,smaller,larger)
    known={key:read(f'werner_joint_{key}.json')['results'] for key in ('k0','k12','k12f')}
    comparisons=0
    for name,results in known.items():
        for mode,records in results.items():
            last=-1
            for record in records:
                assert record['Q']>last;last=record['Q']
                assert 0<=record['rate']<=1
                assert record['M']<=int(mode[-1]);comparisons+=1
        for permission in ('real','full'):
            for M in range(1,4):
                a={round(v['Q'],8):v['rate'] for v in results[f'{permission}_M{M}']}
                b={round(v['Q'],8):v['rate'] for v in results[f'{permission}_M{M+1}']}
                for q in a.keys()&b.keys():assert a[q]<=b[q]+2e-12
    for left,right in (('k0','k12'),('k12','k12f')):
        for mode in known[left]:
            a={round(v['Q'],8):v['rate'] for v in known[left][mode]}
            b={round(v['Q'],8):v['rate'] for v in known[right][mode]}
            for q in a.keys()&b.keys():assert a[q]<=b[q]+2e-12
    with (DATA/'bb84_summary.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    original={}
    for relative,size in (('manuscript/main.tex',52094),('output/pdf/main.pdf',446142)):
        path=WORK/relative;assert path.stat().st_size==size
        original[relative]=dict(bytes=size,mtime=path.stat().st_mtime,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    summary=dict(status='passed',python_files_parsed=syntax,json_files_checked=len(files),verification_statuses=statuses,
                 bb84_points_per_class=14,bb84_classes=3,bb84_maximum_gaps=gaps,
                 positive_feedback_rules=certs['positive_policies'],certified_policy_cells=certificate_cells,
                 explicitly_discarded_branches_in_certificate=dropped,werner_record_checks=comparisons,
                 original_manuscript_snapshot=original,
                 limitations='The global finite-library upper bounds are numerical, not outward-rounded converses. Positive feedback rules have separate interval achievability certificates. General CRO bounds and infinite fixed-memory optimality are not claimed.')
    (DATA/'validation_summary.json').write_text(json.dumps(summary,indent=2))
    targets=[]
    for folder in (ROOT,DATA,ROOT/'figures'):
        targets.extend(p for p in folder.iterdir() if p.is_file() and p.suffix in {'.py','.md','.json','.csv','.tex','.dat'} and p.name!='manifest.json')
    targets.append(WORK/'output/pdf/research_numerics.pdf')
    manifest={str(p.relative_to(WORK)).replace('\\','/'):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(set(targets))}
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
