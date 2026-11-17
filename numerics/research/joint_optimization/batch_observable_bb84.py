from pathlib import Path
import json
from search_observable_bb84 import search
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data'
for path in sorted(DATA.glob('bb84_[0-9]*.json')):
    data=json.loads(path.read_text());eb,ep=data['eb'],data['ep']
    target=DATA/f'bb84_observable_{eb:.5f}_{ep:.5f}.json'
    if target.exists():continue
    result=search(eb,ep);target.write_text(json.dumps(result,indent=2))
    print('Saved',target.name,flush=True)
