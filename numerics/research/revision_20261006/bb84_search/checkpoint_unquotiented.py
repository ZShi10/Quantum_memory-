from pathlib import Path
import json
p=Path('research/revision_20261006/bb84_search/runs/705a628c40d89588c33f0d3f613d11b5b3b2a383b89c302e75c57eb958ff6371/status.json')
v=json.loads(p.read_text());v.update(stage='checkpoint_before_exact_objective_quotient',complete=False,reason='Unquotiented discovery benchmark stopped after repeated equivalent LP problems; retained feasible policy and dual records will be replayed, not promoted to a complete class certificate.')
p.write_text(json.dumps(v,indent=2))
