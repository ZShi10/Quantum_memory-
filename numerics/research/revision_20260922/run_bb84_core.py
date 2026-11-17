"""Resumable fixed-policy core sweep, without restarting the Python runtime."""
from core import *
import argparse,subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--permissions',choices=['real','full'],default='real')
    p.add_argument('--q',nargs='+',type=float);p.add_argument('--dense',action='store_true')
    p.add_argument('--n',type=int,default=7);a=p.parse_args()
    qs=np.linspace(0,.25,501) if a.dense else a.q or [.05,.1,.12,.15,.17,.18,.19,.2,.21,.22,.23,.24,.245]
    for q in qs:
        name=f'bb84_fixed_{a.permissions}_{q:.5f}_{q:.5f}_n{a.n}.json';path=DATA/name
        if path.exists() and 'scope' in json.loads(path.read_text()):continue
        command=[sys.executable,str(ROOT/'bb84_fixed.py'),'--b',str(float(q)),'--n',str(a.n),'--permissions',a.permissions]
        result=subprocess.run(command)
        if result.returncode:raise RuntimeError(f'Failed at {q}: {result.returncode}')


if __name__=='__main__':main()
