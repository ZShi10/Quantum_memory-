"""Export only completed enlarged-class point certificates and witnesses."""
from pathlib import Path
import csv
import hashlib
import json
import math

HERE=Path(__file__).resolve().parent
selected={}


def add_certified_subset_witnesses(result, source, manifest):
    """Union exact feasible old-class witnesses, never an old optimum claim."""
    provenance={}
    target=manifest['specification']
    for prior in (HERE/'runs').glob('*/summary.json'):
        summary=json.loads(prior.read_text())
        old=json.loads((prior.parent/'manifest.json').read_text())
        spec=old['specification']
        if (not summary.get('complete') or spec.get('ed_extension',False)
            or spec.get('permissions')!=target['permissions']
            or float(spec['b'])!=float(target['b']) or float(spec['p'])!=float(target['p'])
            or spec.get('short_max',7)>target['short_max']
            or spec.get('nmax',128)>target['nmax']
            or spec.get('classical_B_max',16)>target['classical_B_max']
            or spec.get('quantum_class')!='four_real_decoder_words_depth_0_to_7_and_constant_raw_pumps_r_1_to_160'):
            continue
        dependencies=[key for key in old['source_files'] if not key.startswith('research/revision_20261006/')]
        if any(manifest['source_files'].get(key)!=old['source_files'][key] for key in dependencies):
            continue
        old_source=prior.parent/summary['source']
        witnesses=json.loads(old_source.read_text())['results']
        for key,old_bound in summary['results'].items():
            if old_bound['lower']<=result['results'][key]['lower']:
                continue
            witness=witnesses[key]
            assert witness['certificate']['lower']>=old_bound['lower']
            assert witness['ED']['memory_upper_bound']<=int(key[1:])
            assert old_bound['lower']<=result['results'][key]['upper']
            result['results'][key]['lower']=old_bound['lower']
            gap=result['results'][key]['upper']-old_bound['lower']
            result['results'][key]['gap_upper']=math.nextafter(gap,math.inf)
            source['results'][key]=witness
            provenance[key]=dict(kind='certified_feasible_subset_witness',
                source=str(old_source),source_sha256=hashlib.sha256(old_source.read_bytes()).hexdigest(),
                manifest_fingerprint=old['fingerprint'])
    return provenance
for summary in (HERE/'runs').glob('*/summary.json'):
    result=json.loads(summary.read_text())
    manifest=json.loads((summary.parent/'manifest.json').read_text())
    spec=manifest['specification']
    if (not result.get('complete') or spec.get('short_max')!=8
            or not spec.get('ed_extension') or spec.get('permissions')!='real'
            or spec['b']!=spec['p'] or spec.get('tree_leaves')!=5
            or spec.get('pump_checks')!=7):
        continue
    q=spec['b']
    if q not in selected or summary.stat().st_mtime>selected[q][0].stat().st_mtime:
        selected[q]=(summary,result,manifest)
rows=[]
points=[]
for q,(path,result,manifest) in sorted(selected.items(),key=lambda item:float(item[0])):
    source_path=path.parent/result['source']
    source=json.loads(source_path.read_text())
    proof_path=path.parent/result['certificate']
    proof=json.loads(proof_path.read_text())
    assert result['fingerprint']==manifest['fingerprint']
    assert proof['registered_class_manifest']==manifest['fingerprint']
    assert len(proof['rounds'])==17
    subset_provenance=add_certified_subset_witnesses(result,source,manifest)
    for key,bound in result['results'].items():
        assert bound['converged'] and bound['upper']-bound['lower']<=result['required_gaps'][key]
        policy=source['results'][key]
        rows.append(dict(Q=q,M=key[1:],lower=bound['lower'],upper=bound['upper'],
            gap=bound['upper']-bound['lower'],family=policy.get('ED',{}).get('family','discard'),
            ED=json.dumps(policy.get('ED'),sort_keys=True),B=policy.get('classical_B_rounds',''),
            strategy=json.dumps(policy.get('strategy',[]),sort_keys=True),fingerprint=result['fingerprint']))
    points.append(dict(Q=q,results=result['results'],witnesses=source['results'],
        specification=manifest['specification'],fingerprint=result['fingerprint'],job=str(path.parent),
        stronger_subset_witnesses=subset_provenance,
        source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        upper_proof_sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest(),seconds=result['seconds']))
destination=HERE/'expanded_points.json'
temporary=destination.with_suffix('.writing.json')
temporary.write_text(json.dumps(dict(points=points,scope='Pointwise registered recurrence/tree/pump/B/AD class only. Add the separate flag-resolved fixed-block class by certified union; no continuum or arbitrary-memory optimality claim.'),indent=2))
temporary.replace(destination)
with (HERE/'expanded_points.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=['Q','M','lower','upper','gap','family','ED','B','strategy','fingerprint'])
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(dict(complete_points=len(points),inputs=[point['Q'] for point in points])))
