"""Resume discovery while using existing certified feasible rates as screens.

Only an unnecessary lower-policy certificate is skipped when its sampled
upper cannot improve a feasible incumbent. Independent whole-menu upper
drivers still evaluate every candidate; this floating screen proves nothing.
"""
from pathlib import Path
import argparse,hashlib,json
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--q',type=float,required=True)
    p.add_argument('--label',required=True,choices=['full','real_xyz','real_xz']);a=p.parse_args()
    original=HERE/'binary_family_search.py';source=original.read_text(encoding='utf-8')
    old='nmax=128,tolerance=2e-9,max_iterations=30,incumbent=0.'
    new="nmax=128,tolerance=2e-9,max_iterations=30,incumbent=results[f'M{ed[\"memory_upper_bound\"]}']['lower']"
    assert source.count(old)==1
    executed=source.replace(old,new)
    namespace=dict(__name__='efficient_binary_discovery',__file__=str(original))
    exec(compile(executed,str(original),'exec'),namespace)
    axes='XZ' if a.label=='real_xz' else 'XYZ'
    catalog='real_binary_catalog' if a.label.startswith('real') else 'full_binary_catalog'
    namespace['run'](a.q,catalog,a.label,axes,True)
    path=HERE/'bb84'/str(a.q)/a.label/'feasible.json'
    result=json.loads(path.read_bytes())
    result['execution']=dict(original_source_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        executed_source_sha256=hashlib.sha256(executed.encode()).hexdigest(),
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        change='Pass current certified feasible incumbent to floating lower-policy discovery. No certified upper pruning is inferred.')
    namespace['save'](path,result)

if __name__=='__main__':main()
