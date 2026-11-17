"""Cross-check published coordinates against independent family bounds."""
from pathlib import Path
import argparse,csv,json,math
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_bytes())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--require-complete',action='store_true');a=ap.parse_args()
    root=HERE/'export';status=read(root/'status.json')
    if a.require_complete:assert not status['missing'],status
    bounds=read(root/'bb84_bounds.json')
    by={(r['label'],float(r['Q']),r['memory']):r for r in bounds}
    for r in bounds:
        assert 0<=r['lower']<=r['upper'],r
        if r['union_upper'] is not None:assert r['upper']<=r['union_upper'],r
    with (root/'bb84_dense_curves.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    checked=0
    for r in rows:
        q=float(r['Q'])
        for family in ('full','real_xyz','real_xz'):
            vals=[float(r[f'{family}_R{m}']) for m in range(1,9)]
            assert all(math.isfinite(v) and v>=0 for v in vals)
            assert all(b+1e-16>=a for a,b in zip(vals,vals[1:])),(family,q)
            for m,v in enumerate(vals,1):
                b=by.get((family,q,m))
                if b:
                    assert v<=b['upper']+1e-14,(family,q,m,v,b['upper'])
                    checked+=1
        for m in range(1,9):
            assert float(r[f'full_R{m}'])+1e-16>=float(r[f'real_xyz_R{m}'])
            assert float(r[f'real_xyz_R{m}'])+1e-16>=float(r[f'real_xz_R{m}'])
    claims=[]
    for q,small,big,target in [(.17,1,2,.028),(.2,2,3,.009),(.23,3,4,.0008),(.23,5,6,.0009)]:
        lo=by['full',q,big]['lower'];hi=by['full',q,small]['union_upper']
        assert hi is not None and hi<target<=lo,(q,small,big,target,hi,lo)
        claims.append(dict(Q=q,excluded_memory=small,minimum_memory=big,target=target,lower=lo,upper=hi,relative_gain_lower=lo/hi-1,absolute_gap_lower=lo-hi))
    separations=[]
    for q,small,big in [(.19,2,3),(.2,2,3),(.21,2,3),(.22,2,3),(.23,3,4),(.23,5,6)]:
        lo=by['full',q,big]['lower'];hi=by['full',q,small]['union_upper']
        assert hi is not None and 0<hi<lo,(q,small,big,hi,lo)
        separations.append(dict(Q=q,smaller_memory=small,larger_memory=big,
            lower=lo,whole_smaller_family_upper=hi,absolute_gap_lower=lo-hi,
            relative_gain_lower=lo/hi-1))
    report=dict(status='passed',complete=not status['missing'],dense_points=len(rows),
        dense_vs_independent_upper_checks=checked,certified_target_claims=claims,
        strict_memory_separations=separations,
        scope='Cross-artifact consistency, nesting, and strict lower-versus-whole-family-upper logic; this does not replace arithmetic or enumeration proofs.')
    (root/'cross_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
