import json
from pathlib import Path
r=Path('research/revision_20261006/bb84_search/runs/bf5e7ecdfe9440a1577576aec43f6badc16aa1da088e9cf44294767a91c1a59c')
(r/'status.json').write_text(json.dumps(dict(stage='abandoned_cold_start_benchmark',complete=False,reason='Initial zero-incumbent cold search interrupted before any lower/upper certificate; replaced by full-generator feasible warm-start workflow.',superseded_by='ad238d55f22bb552cde69334351f0b10d786e936540ec95d872e6b685ac99dbc',snapshot_note='run_point_snapshot.py was copied only after the warm-start update and is not the original abandoned executable; the manifest hash remains the authoritative original identity.'),indent=2))
