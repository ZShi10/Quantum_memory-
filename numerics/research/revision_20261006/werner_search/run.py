"""One isolated Werner input: discovery and whole-declared-class enclosure.

This is a new run, not a relabeling of the historical dense policy envelope.
The complete m<=8 terminal instruments and every registered recursive/tree
candidate use the same n<=8 code provider and paid classical menu.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import shutil

HERE = Path(__file__).resolve().parent
REVISION = HERE.parent
sys.path.insert(0, str(REVISION))
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--q', required=True, type=str)
    parser.add_argument('--abs-tol', default=1e-11, type=float)
    parser.add_argument('--rel-tol', default=1e-5, type=float)
    parser.add_argument('--discover-only', action='store_true')
    parser.add_argument('--reuse-discovery', type=Path,
                        help='Reuse only same-input feasible candidate records; all lowers are replayed and all uppers recomputed.')
    args = parser.parse_args()
    q = float(args.q)
    assert 0 < q < 1/3
    from runtime_context import configure, fingerprint
    sources = list(HERE.glob('*.py')) + list((REVISION/'resource_model').glob('*.py'))
    sources += list((REVISION/'code_extension').glob('*.py'))
    sources += [REVISION/'code_extension/data/codes_complete_n8.json']
    provenance = fingerprint(sources)
    spec = dict(Q=args.q, permissions='enlarged_axes',
                complete_Werner_terminal_m_max=8, homogeneous_depth=7,
                constant_pump_inputs=160, ordered_tree_leaves=5,
                variable_pump_checks=7, all_codes_n_max=8, endpoint_n_max=128,
                classical_B_max=16, abs_tol=args.abs_tol, rel_tol=args.rel_tol,
                source_digest=provenance['digest'])
    job_id = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
    run = HERE/'runs'/job_id
    core, _ = configure(run)
    if args.reuse_discovery:
        seed = args.reuse_discovery.resolve()/'data'
        candidate = json.loads((seed/'registered_discovery.json').read_text())
        block_seed = json.loads((seed/'werner_complete_joint_n128.json').read_text())
        assert candidate['Q']==q and all(r['Q']==q for r in block_seed['results'])
        for name in ('registered_discovery.json','werner_complete_joint_n128.json'):
            if not (core.DATA/name).exists():shutil.copy2(seed/name,core.DATA/name)
    from bb84_search.n8_adapter import install
    provider = install(core)
    from core import np, atomic_json
    from fractions import Fraction as F
    from recurrence_v2 import generate as old_generate
    from recurrence_equivalence import representatives
    from resource_model.catalog import generate as tree_generate, coverage
    from resource_model.model import homogeneous_tree, pumping_tree, structural_record, tree_from_dict, evaluate
    from resource_model.bounds import prepare_intervals
    from known_search import optimize
    from certify_werner_joint_union import terminal, raw_bounds
    from complete_werner_search import load
    from bb84_fixed import physical_kernel
    import stable_intervals as iv
    kernels = provider.load_codes()
    physical = {k.entry['id']:physical_kernel(k) for k in kernels}
    kd = {k.entry['id']:k for k in kernels}
    start = time.time()
    atomic_json(run/'manifest.json', dict(status='running', spec=spec,
                provenance=provenance, extension_coverage=coverage(),
                feasible_discovery_import=str(args.reuse_discovery) if args.reuse_discovery else None))
    status = lambda stage: atomic_json(run/'status.json', dict(stage=stage, seconds=time.time()-start))
    print('WERNNER_JOB', job_id, spec, flush=True)

    source = np.array([1-1.5*q, q/2, q/2, q/2])
    row = [iv.rational(v) for v in (1-F(3,2)*F(args.q), *([F(args.q)/2]*3))]
    source_bounds = tuple(np.array([[v[i] for v in row]]) for i in (0,1))

    def as_tree(record):
        if record['family']=='explicit_binary_tree':return record
        tree = (homogeneous_tree(record['path']) if record['family']=='real_decoder_recurrence'
                else pumping_tree(record['primitive']*(record['r']-1)))
        result = structural_record(tree, record['final_axis'], 'enlarged_axes')
        assert result['memory_upper_bound']==record['memory_upper_bound']
        result['legacy_record']=record
        return result

    def replay_witness(result, budget):
        ad = result['AD']
        record = as_tree(result['ED'])
        accounting = evaluate(tree_from_dict(record['tree']), source)
        if ad is None:
            lo=hi=0.
        else:
            p,y = prepare_intervals(source_bounds,record,ad['classical_B_rounds'])
            value = iv.mul(y,terminal(p,ad,kd,physical))
            lo,hi = max(0.,float(value[0][0])),max(0.,float(value[1][0]))
            assert lo-2e-11 <= result['rate'] <= hi+2e-11
        return dict(Q=q,budget=budget,lower=lo,upper=hi,ED=record,AD=ad,
                    expected_raw_pairs_per_ED_output=float(accounting['expected_raw_cost']),
                    ED_yield_per_raw_pair=float(accounting['yield_per_raw_pair']),
                    scope='Independent outward replay of this executable witness only')

    discovery_path = core.DATA/'registered_discovery.json'
    if discovery_path.exists():
        registered=json.loads(discovery_path.read_text())
    else:
        status('registered_candidate_discovery')
        p,y,M,records=old_generate(source,7,'full',160)
        rep=representatives(records)
        ids=np.flatnonzero(rep==np.arange(len(rep)))
        p,y,M,records=p[ids],y[ids],M[ids],[records[i] for i in ids]
        ep,ey,em,er=tree_generate(source,'full',5,7)
        p=np.concatenate((p,ep));y=np.concatenate((y,ey));M=np.concatenate((M,em));records+=er
        print('REGISTERED_CANDIDATES',len(records),'MEMORIES',dict(zip(*np.unique(M,return_counts=True))),flush=True)
        registered=optimize(p,y,M,records,kernels,K=16,long_max=128,tolerance=args.abs_tol/32,with_watanabe=True)
        registered['Q']=q
        registered['scope']='All declared homogeneous, constant raw-pump, variable raw-pump and ordered-tree candidates; new complete n<=8 classical menu. Floating discovery only.'
        atomic_json(discovery_path,registered)
        del p,y,M,records,ep,ey,em,er
    witnesses={f'{q:.12g}_{budget}':replay_witness(result,budget) for budget,result in registered['results'].items()}
    atomic_json(core.DATA/'werner_v2_full_certified.json',dict(results=witnesses,
                scope='New registered-union witnesses, including explicit trees; legacy filename is an interface only'))
    lower=np.zeros(9)
    for row in witnesses.values():lower[int(row['budget'][1:]):]=np.maximum(lower[int(row['budget'][1:]):],row['lower'])

    data,weights,offsets,agg=load()
    discovery_path=core.DATA/'werner_complete_joint_n128.json'
    if discovery_path.exists():
        block=json.loads(discovery_path.read_text())['results'][0]
    else:
        status('complete_terminal_core_discovery')
        import werner_complete_joint as block_search
        ordinary_screen=block_search.relevant_states
        def union_screen(d,a,potential,incumbent,padding=0):
            return ordinary_screen(d,a,potential,np.maximum(incumbent,lower[:len(incumbent)]),padding)
        block_search.relevant_states=union_screen
        block=block_search.optimize(q,data,weights,offsets,agg,kernels,nmax=128,tolerance=args.abs_tol/32)
        block['scope']='Complete m<=8 terminal class, all flags; external executable recurrence/tree incumbents used only to prune the union objective, not to claim individual core optimality.'
        atomic_json(discovery_path,dict(results=[block]))
    if args.discover_only:
        status('discovery_complete_certification_pending')
        print('DISCOVERY', {k:v['rate'] for k,v in registered['results'].items()},
              {k:v['rate'] for k,v in block['results'].items()},flush=True)
        return
    status('complete_terminal_core_upper')
    from certify_werner_joint_union import certify as block_certify
    core_path=core.DATA/'complete_core_upper.json'
    if core_path.exists():block_upper=json.loads(core_path.read_text())
    else:
        block_upper=block_certify(q,data,weights,offsets,agg,kernels,tolerance=args.abs_tol/4)
        block_upper['scope']='Complete m<=8 terminal core with n<=8 general codes and endpoint codes<=128, three paid modes, Watanabe and B<=16; outward union upper.'
        atomic_json(core_path,block_upper)
    for witness in block_upper['core_witnesses']:
        lower[witness['M']:]=np.maximum(lower[witness['M']:],witness['lower'])
    status('registered_whole_class_upper')
    from upper import certify as registered_certify
    upper_path=core.DATA/'registered_upper.json'
    if upper_path.exists():registered_upper=json.loads(upper_path.read_text())
    else:
        registered_upper=registered_certify(args.q,kernels,physical,lower,args.abs_tol/4,run)
        atomic_json(upper_path,registered_upper)
    results={}
    for budget in range(1,9):
        key=f'M{budget}'
        high=max(block_upper['results'][key]['core_upper'],registered_upper['results'][key]['upper'],float(lower[budget]))
        gap=float(iv.up(high-lower[budget]))
        target=max(args.abs_tol,args.rel_tol*lower[budget])
        results[key]=dict(lower=float(lower[budget]),upper=high,gap=gap,target=target,converged=gap<=target,
                          registered_achievable=registered['results'][key]['rate'],
                          complete_block_achievable=block['results'][f'M{min(budget,4)}']['rate'])
    result=dict(status='certified' if all(v['converged'] for v in results.values()) else 'gap_not_closed',
                Q=q,spec=spec,results=results,seconds=time.time()-start,job_id=job_id,
                evidence=dict(registered_witnesses='data/werner_v2_full_certified.json',
                              core='data/complete_core_upper.json',registered_upper='data/registered_upper.json'))
    atomic_json(run/'result.json',result)
    status(result['status'])
    print('FINAL',json.dumps(result),flush=True)


if __name__=='__main__':main()
