"""Collect immutable completed old-class regression proofs, not plot smoothing."""
from pathlib import Path
import csv
import json

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
old = {float(row['Q']): row for row in csv.DictReader((PROJECT/'research/revision_20260922/figures/bb84_comparison.csv').open())}
rows = []
for path in (HERE/'runs').glob('*/summary.json'):
    doc = json.loads(path.read_text())
    manifest = json.loads((path.parent/'manifest.json').read_text())
    spec = manifest['specification']
    if not doc.get('complete') or spec.get('short_max', 7) != 7 or spec.get('ed_extension', False):
        continue
    if doc['b'] not in (.225, .227, .23, .235) or doc['b'] != doc['p']:
        continue
    row = dict(Q=doc['b'], seconds=doc['seconds'], fingerprint=doc['fingerprint'])
    for M in (1, 2, 4, 8):
        result = doc['results'][f'M{M}']
        row[f'M{M}_lower'], row[f'M{M}_upper'] = result['lower'], result['upper']
        row[f'M{M}_old_curve'] = float(old[doc['b']][f'M{M}'])
        row[f'M{M}_improvement_factor'] = result['lower']/row[f'M{M}_old_curve'] if row[f'M{M}_old_curve'] else None
        assert result['upper']-result['lower'] <= doc['required_gaps'][f'M{M}']
    rows.append(row)
rows.sort(key=lambda row: row['Q'])
assert [row['Q'] for row in rows] == [.225, .227, .23, .235], rows
with (HERE/'baseline_regression_summary.csv').open('w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(HERE/'baseline_regression_summary.json').write_text(json.dumps(dict(complete=True, rows=rows,
    interpretation='Every registered legacy recurrence/pump/B/all-short-code/long-endpoint-code candidate is bounded at every one of these actual input values. This is a regression benchmark, separate from the new n8/tree union.'), indent=2))
print(json.dumps(rows, indent=2))
