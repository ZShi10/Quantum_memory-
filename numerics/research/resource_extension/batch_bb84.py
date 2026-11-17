from common import *
import subprocess,os

points=[(.10,.10),(.15,.15),(.18,.18),(.20,.20),(.21,.21),(.22,.22),(.23,.23),(.24,.24),(.245,.245),(.10,.30),(.18,.25),(.12,.35)]
for eb,ep in points:
    target=DATA/f'bb84_{eb:.5f}_{ep:.5f}.json'
    if target.exists():continue
    command=[sys.executable,str(ROOT/'search_bb84_extended.py'),'--eb',str(eb),'--ep',str(ep)]
    result=subprocess.run(command,env=os.environ)
    if result.returncode:raise SystemExit(result.returncode)
