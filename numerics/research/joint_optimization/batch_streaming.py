from pathlib import Path
import json
from search_streaming import search

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data'
for eb,ep in [(q,q) for q in (.10,.15,.18,.20,.21,.22,.23,.24,.245,.248,.249)]+[(.05,.20),(.10,.30),(.18,.25),(.12,.35),(.20,.28)]:
    path=DATA/f'streaming_{eb:.5f}_{ep:.5f}.json'
    if path.exists():continue
    result=search(eb,ep)
    path.write_text(json.dumps(result,indent=2,default=lambda x:x.item()))
    print('Saved',path.name,flush=True)
