"""Tighten previously certified unresolved candidates with new feasible duals.

All other candidates retain the certified incumbent-plus-tolerance bound
used by the complete earlier pass. No floating discovery is trusted here.
"""
from core import *
from compound_menu_bounds import dual_upper
from recurrence_equivalence import key as canonical_key
import argparse,hashlib
import stable_intervals as iv


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);a=parser.parse_args()
    path=DATA/a.source;src=json.loads(path.read_text());dest=path.with_name(path.stem+'_menu_certified.json')
    out=json.loads(dest.read_text());assert out['complete'] and src['complete_candidate_screen']
    old_hash=hashlib.sha256(dest.read_bytes()).hexdigest();tolerance=out['tolerance']
    old_lower={key:value['lower'] for key,value in out['results'].items()}
    def identity(ed,B):return json.dumps([canonical_key(ed) if src['b']==src['p'] else ed,B],sort_keys=True)
    evaluated={identity(r['ED'],r['classical_B_rounds']):r for r in src['evaluated'].values() if 'sampled_dual' in r}
    for rr in out['rounds']:
        unresolved=rr['unresolved']
        for row in unresolved:
            tag=identity(row['ED'],rr['B'])
            if tag not in evaluated:continue
            if tag not in out['duals']:
                out['duals'][tag]=dual_upper(src['b'],src['p'],evaluated[tag],src['nmax'])
            row['upper']=min(row['upper'],out['duals'][tag]['upper'])
        for m in range(1,9):
            key=f'M{m}'
            # The earlier exact binary64 comparison classified every
            # omitted candidate at M<=m below this representable threshold.
            resolved_cap=float(old_lower[key]+tolerance)
            remaining=max([r['upper'] for r in unresolved if r['ED']['memory_upper_bound']<=m]+[0.])
            rr['upper'][key]=min(rr['upper'][key],max(resolved_cap,remaining))
        rr['unresolved']=[r for r in unresolved if r['upper']>src['results'][f'M{r["ED"]["memory_upper_bound"]}']['lower']+tolerance]
    for m in range(1,9):
        key=f'M{m}';lo=src['results'][key]['lower'];hi=max(r['upper'][key] for r in out['rounds'])
        assert hi+1e-14>=lo
        out['results'][key]=dict(lower=lo,upper=hi,gap_upper=float(iv.up(hi-lo)),converged=hi<=lo+tolerance)
    out['source_digest']=hashlib.sha256(path.read_bytes()).hexdigest()
    out['dual_refinement']=dict(previous_certificate_sha256=old_hash,previous_incumbents=old_lower,
        explanation='Every old omitted candidate retains its proven incumbent-plus-tolerance cap; only explicitly recorded unresolved candidates are reduced by new exact feasible duals.')
    atomic_json(dest,out);print(json.dumps(out['results'],indent=2),flush=True)


if __name__=='__main__':main()
