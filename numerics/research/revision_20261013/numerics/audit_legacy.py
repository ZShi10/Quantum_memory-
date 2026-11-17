"""Verify the frozen release and reconstruct both global rate tables exactly.

Hash integrity is not an independent rerun of a historical optimization.
This distinction is recorded in every exported curve row.
"""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import csv
import hashlib
import json
import shutil
import time

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
MANIFEST_SHA='6894ed43294fc2cecb41b004bcea0fb138751929a3ee8bdeb0deaeb174f443f4'
DELIVERY_SHA='1dd58d89189dbf90ec09f22281b31fc3f8c65efc2f52348fd037592c79a5daf8'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    started=time.monotonic()
    # A packaged copy takes precedence; the development workspace is fallback.
    frozen=ROOT/'evidence/legacy_frozen'
    if not frozen.exists():
        frozen=PROJECT/'research/revision_20261006/release_tools/release_ready_bundles'/MANIFEST_SHA
    manifest_path=frozen/'manifest.json'
    assert sha(manifest_path)==MANIFEST_SHA
    manifest=json.loads(manifest_path.read_bytes())
    assert manifest['release_eligible'] and manifest['preview_consistent']
    unique=set();total_bytes=0
    for key,row in manifest['files'].items():
        assert row['payload'],('External-only entry',key)
        p=frozen/row['payload']
        assert p.stat().st_size==row['size'] and sha(p)==row['sha256'],key
        unique.add(row['payload']);total_bytes+=row['size']
    key=next(k for k in manifest['files'] if k.endswith(f'curves_delivery/delivery_{DELIVERY_SHA}.json'))
    payload=frozen/manifest['files'][key]['payload']
    assert sha(payload)==DELIVERY_SHA
    delivery=json.loads(payload.read_bytes())
    points=delivery['points']
    (ROOT/'data/inherited').mkdir(parents=True,exist_ok=True)
    shutil.copyfile(payload,ROOT/'data/inherited/frozen_curve_delivery.json')
    shutil.copyfile(manifest_path,ROOT/'data/inherited/archive_manifest.json')
    index={(p['model'],F(str(p['Q']))):p for p in points}
    assert len(index)==191
    checked={}
    for label,path,qfield,mfield in [
        ('QKD1',ROOT/'reproduction/extensions/data/legacy_rate_bounds.csv','Q','live_memory_budget'),
        ('QKD2',ROOT/'reproduction/controls/data/archived_rate_endpoints.csv','Q','memory')]:
        count=0
        for r in csv.DictReader(path.open(encoding='utf-8')):
            p=index[r['model'],F(r[qfield])];m=int(r[mfield]);s=p['results'][f'M{m}']
            assert float(r['lower'])==s['lower'],(label,r)
            upper=r['upper']
            if upper not in ('','None','nan'):assert float(upper)==s['upper'],(label,r)
            else:assert s.get('upper') is None
            count+=1
        assert count==191*8
        checked[label]=count
    rows=[]
    for p in points:
        for m in range(1,9):
            result=p['results'][f'M{m}']
            rows.append(dict(source_model=p['model'],Q=str(p['Q']),Q_exact=str(F(str(p['Q']))),
                memory_budget=m,lower=result['lower'],upper=result.get('upper'),
                source_status=p['status'],reproduction_status='inherited_bound; exact table and archive integrity rechecked',
                ED_family='finite union: complete small blocks plus registered recursive/tree/pump families',
                input_scope='complete N<=8 Werner or real N<=5 BB84; structured ancestry up to 160',
                retry_rule='full-input complete blocks; registered independent-child retry for recursive constructions',
                classical_menu='B<=16; all binary codes n<=8; repetition/SPC<=128; clear/plain/compressed modes; Watanabe',
                BB84_policy='fixed ED and classical allocations before infimum over z' if p['model']=='BB84_real' else 'known source',
                full_protocol_record=f'data/inherited/frozen_curve_delivery.json: model={p["model"]}, Q={p["Q"]}, M{m}'))
    with (ROOT/'data/global_rate_landscape.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    result=dict(status='passed',logical_files=len(manifest['files']),unique_payloads=len(unique),
        logical_bytes=total_bytes,manifest_sha256=MANIFEST_SHA,delivery_sha256=DELIVERY_SHA,
        checked_import_rows=checked,source_points=dict(Counter(p['model'] for p in points)),
        source_status_counts=dict(Counter(p['model']+':'+p['status'] for p in points)),
        elapsed_seconds=time.monotonic()-started,
        boundary='All 933 file records and both 1528-row imported rate tables rechecked; the historical whole-class optimizer is not rerun by this audit.')
    (ROOT/'validation/legacy_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    for model in ('Werner','BB84_real'):
        p=next(p for p in points if p['model']==model)
        print(model,'point fields',list(p),'witness type',type(p.get('witnesses')).__name__)

if __name__=='__main__':main()
