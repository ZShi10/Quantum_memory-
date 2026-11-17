"""Freeze the relevant uploaded evidence; derive stopping-robust inequalities.

Checks the supplied archive, not the missing upstream optimizer dependencies.
The new inequalities are exact rational consequences of archived upper endpoints.
"""
from pathlib import Path
from fractions import Fraction as F
import json,csv,hashlib,shutil,argparse
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=Path('/mnt/data/inputs/original_project')

def run(base=DEFAULT):
    m=json.loads((base/'input_manifest.json').read_text())
    checked=0
    for name,rec in m['files'].items():
        p=base/name
        data=p.read_bytes()
        if len(data)!=rec['size'] or hashlib.sha256(data).hexdigest()!=rec['sha256']:
            raise ValueError(f'Archive integrity failure: {name}')
        checked+=1
    delivery=json.loads((base/m['roles']['delivery']).read_text())
    witnesses=json.loads((base/m['roles']['witnesses']).read_text())
    points=delivery['points']
    rows=[];stopping=[]
    for p in points:
        for mm in range(1,9):
            r=p['results'][f'M{mm}']
            row={'model':p['model'],'Q':p['Q'],'memory':mm,'lower':r['lower'],'upper':r.get('upper'),
                 'registered_upper':r.get('registered_upper'),'block_upper':r.get('complete_block_upper'),
                 'source_status':p['status']}
            rows.append(row)
        if p['model']=='BB84_real' and p['status']=='certified_optimum':
            for mm in range(2,9):
                prev=p['results'][f'M{mm-1}'];rr=p['results'][f'M{mm}']
                if prev.get('registered_upper') is None or prev.get('complete_block_upper') is None:continue
                upper=max(F.from_float(prev['registered_upper']),5*F.from_float(prev['complete_block_upper']))
                lower=F.from_float(rr['lower'])
                stopping.append({'model':p['model'],'Q':p['Q'],'smaller_memory':mm-1,'larger_memory':mm,
                    'archived_large_lower':float(lower),'early_stop_small_upper':float(upper),
                    'exact_small_upper':str(upper),'exact_large_lower':str(lower),
                    'strict_gap':float(lower-upper),'ratio_lower':float(lower/upper) if upper else None,
                    'strict':lower>upper})
    def writecsv(name,rs):
        with (ROOT/'data'/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    writecsv('archived_rate_endpoints.csv',rows);writecsv('early_stopping_separations.csv',stopping)
    slim={}; selections=[]
    for s in witnesses['selections']:
        did=s['protocol_id']
        if did is None:
            selections.append(s);continue
        d=witnesses['definitions'][did]
        ed=d['definition'].get('source_ED_record',{})
        keep={k:d[k] for k in ['model','source_parameters','AD_strategy','classical_B_rounds','saved_rate_certificate','certified_lower_from_source'] if k in d}
        keep['ED']=ed
        if 'resource_contract' in d['definition']:keep['resource_contract']=d['definition']['resource_contract']
        # Complete terminal schedules are essential, and are retained in full.
        if d['definition'].get('terminal_schedule'):keep['terminal_schedule']=d['definition']['terminal_schedule']
        if d['definition'].get('compilation'):keep['compilation']=d['definition']['compilation']
        slim[did]=keep;selections.append(s)
    (ROOT/'data/selected_protocols.json').write_text(json.dumps({'definitions':slim,'selections':selections},indent=2))
    # Full original point records at the principal comparison inputs and their bound seeds.
    seed_records={};seed_manifest={}
    for p in points:
        if (p['model']=='BB84_real' and p['Q'] in [.22,.23,.24]) or (p['model']=='Werner' and p['Q'] in [.2,.25,.28,.3]):
            seed_records[p['model']+'_'+str(p['Q'])]=p
            refs=p.get('block_evidence',[])+([p['registered_evidence']] if p.get('registered_evidence') else [])
            for rec in refs:
                loc='research/revision_20261006/reporting/curves_delivery/'+rec['snapshot'].replace('\\','/')
                src=base/loc
                if src.exists():
                    target=ROOT/'source_provenance'/src.name
                    shutil.copyfile(src,target);seed_manifest[src.name]={'original_path':loc,'sha256':hashlib.sha256(src.read_bytes()).hexdigest()}
    (ROOT/'source_provenance/selected_archived_points.json').write_text(json.dumps(seed_records,indent=2))
    # Original reference curves, with their provenance labels kept.
    refdir=ROOT/'data/literature';refdir.mkdir(exist_ok=True)
    figurebase=base/'research/revision_20261006/reporting/paper_figures'
    copied=[]
    for src in figurebase.glob('*.csv'):
        if any(x in src.name for x in ['Du','Sun','sun','control']):
            shutil.copyfile(src,refdir/src.name);copied.append(src.name)
    (ROOT/'source_provenance/input_scope.json').write_text(json.dumps(m['scope'],indent=2))
    report={'status':'all_supplied_payload_hashes_and_lengths_verified','path_records':checked,
        'unique_payloads':len({r['payload'] for r in m['files'].values()}),
        'source_status_counts':m['point_status_counts'],'reference_csvs':copied,
        'seed_manifest':seed_manifest,
        'independent_scope':'New validators do not independently replay every upstream class-wide bound. Archived class-wide certificates are input evidence. New stopping consequences use exact rational endpoint arithmetic.'}
    (ROOT/'verification/archive_audit.json').write_text(json.dumps(report,indent=2))
    print('Archive checked:',checked,'records;',len(stopping),'adjacent-budget stopping comparisons')
    for s in stopping:
        if s['Q'] in [.23,.24] and s['strict']:print(s)
    return rows,stopping

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--project',type=Path,default=DEFAULT);args=a.parse_args();run(args.project)
