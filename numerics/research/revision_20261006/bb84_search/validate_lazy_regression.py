"""Compare the lazy upper proof with an already completed eager proof."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
OLD=HERE/'runs/7b1a38b08662ddd4f004f7dca4c69ae96ae02c4bf2919308a106be9c1ef989d0'
NEW=HERE/'runs/1199a859e3f50021cc850ffc08d37643c323c42ab9785a808129c04eb8d90c06'
a=json.loads((OLD/'summary.json').read_text())
b=json.loads((NEW/'summary.json').read_text())
proof=json.loads((NEW/b['certificate']).read_text())
assert a['complete'] and b['complete']
assert len(proof['rounds'])==17
for key in a['results']:
    assert b['results'][key]['upper']==a['results'][key]['upper']
    assert b['results'][key]['lower']>=a['results'][key]['lower']
    assert b['results'][key]['upper']-b['results'][key]['lower']<=b['required_gaps'][key]
assert all(row['physical_candidates']==44970 for row in proof['rounds'])
assert all(row['exact_objective_classes']==15356 for row in proof['rounds'])
out=dict(status='passed',source_error_rate=.23,memory_budgets=8,classical_B_prefixes=17,
    physical_candidates_per_prefix=44970,exact_objective_classes_per_prefix=15356,
    checks='All eight lazy upper bounds equal the completed eager bounds exactly; fresh subset-witness replay only increases feasible lower bounds; every requested gap passes.',
    old_manifest_sha256=hashlib.sha256((OLD/'manifest.json').read_bytes()).hexdigest(),
    new_manifest_sha256=hashlib.sha256((NEW/'manifest.json').read_bytes()).hexdigest(),
    new_upper_proof_sha256=hashlib.sha256((NEW/b['certificate']).read_bytes()).hexdigest(),
    new_duals=sum(row['new_duals_replayed'] for row in proof['rounds']),
    upper_seconds=b['phases'][0]['upper_seconds'])
(HERE/'lazy_upper_validation.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
