"""Recover exact-test dual measures omitted by an early empty-LP return."""
from core import *
from bb84_joint_extension import solve
import argparse


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',nargs='+',required=True);a=parser.parse_args()
    for name in a.source:
        path=DATA/name;src=json.loads(path.read_text());assert src.get('complete_candidate_screen')
        for key,row in list(src['evaluated'].items()):
            if 'sampled_dual' in row:continue
            M=row['ED']['memory_upper_bound'];inc=src['results'][f'M{M}']['lower']
            result=solve(src['b'],src['p'],row['ED'],row['classical_B_rounds'],src['nmax'],incumbent=inc)
            assert 'sampled_dual' in result
            src['evaluated'][key]=result;atomic_json(path,src)
            print('Dual repaired',name,key,result['lower'],result['upper_numerical'],flush=True)


if __name__=='__main__':main()
