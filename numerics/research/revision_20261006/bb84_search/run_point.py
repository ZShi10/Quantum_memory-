"""Isolated, resumable *whole registered class* BB84 optimization.

Imports the audited 2026-09-22 arithmetic, but redirects every writable
ROOT/DATA binding before dependent imports. No legacy result or cache is
overwritten. Each exact input/specification/source snapshot has its own job.
"""
from pathlib import Path
import argparse
from decimal import Decimal
import hashlib
import json
import os
import shutil
import sys
import time

sys.dont_write_bytecode = True
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
LEGACY = PROJECT / 'research/revision_20260922'


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.writing.json')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def source_snapshot(args=None):
    paths = [Path(__file__), HERE/'n8_adapter.py', HERE/'ed_adapter.py', HERE/'lazy_upper.py']
    for directory in (LEGACY, PROJECT/'research/joint_optimization', PROJECT/'research/completion'):
        paths.extend(directory.glob('*.py'))
    paths.extend((PROJECT/'research/resource_extension/data').glob('codes.json'))
    paths.extend((PROJECT/'research/resource_extension/data/kernels').glob('*.npz'))
    if args is not None and args.short_max == 8:
        directory = HERE.parent/'code_extension'
        paths.extend(directory.glob('*.py'))
        paths.extend((directory/'data').glob('codes_complete_n8.json'))
        paths.extend((directory/'data/kernels').glob('*.npz'))
    if args is not None and args.ed_extension:
        directory = HERE.parent/'resource_model'
        paths.extend(directory.glob('*.py'))
        paths.extend(directory.glob('catalog.json.gz'))
        if quotient_choice(args) != 'none':
            directory = HERE.parent/'objective_equivalence'
            paths.extend(directory.glob('*.py'))
            paths.extend(directory.glob('equal_xz_map.json'))
            paths.extend(directory.glob('equal_xz_validation.json'))
    return {str(path.relative_to(PROJECT)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths))}


def quotient_choice(args):
    if (args.ed_extension and not args.no_objective_quotient and args.tree_leaves == 5
            and args.pump_checks == 7 and Decimal(args.q) == Decimal(args.p or args.q)):
        return 'equal_xz'
    return 'none'


def exact_spec(args):
    quantum = 'four_real_decoder_words_depth_0_to_7_and_constant_raw_pumps_r_1_to_160'
    if args.ed_extension:
        quantum += f'; all_labeled_ordered_binary_trees_leaves_le_{args.tree_leaves}; variable_decoder_raw_pump_checks_le_{args.pump_checks}'
    return dict(schema='bb84_whole_class_20261006_v2',
                b=format(Decimal(args.q).normalize(), 'f'), p=format(Decimal(args.p or args.q).normalize(), 'f'),
                permissions=args.permissions, quantum_class=quantum,
                classical_class=f'all_binary_codes_n_1_to_{args.short_max}_via_direct_sum_dominance; repetition_SPC_{args.short_max+1}_to_nmax; three_Du_modes; Watanabe',
                short_max=args.short_max, ed_extension=args.ed_extension, tree_leaves=args.tree_leaves,
                pump_checks=args.pump_checks, registered_quantum_union=quantum,
                exact_objective_quotient=quotient_choice(args),
                classical_B_max=args.K, nmax=args.nmax, memory_budgets=list(range(1, 9)),
                absolute_tolerance=args.absolute, relative_tolerance=args.relative,
                discovery_initial_tolerance=args.discovery_tolerance,
                security='ideal_IID_Bell_diagonal_fixed_ED_fixed_B_fixed_AD_mixture_all_hidden_t',
                class_exclusions=['arbitrary_memory_bounded_ED', f'general_codes_n_above_{args.short_max}', 'mixing_different_ED_or_B', 'complete_fixed_block_union'])


def prepare_job(args):
    snapshot = source_snapshot(args)
    spec = exact_spec(args)
    identity = dict(specification=spec, source_digest=digest_json(snapshot))
    tag = digest_json(identity)
    job = HERE/'runs'/tag
    job.mkdir(parents=True, exist_ok=True)
    manifest = dict(**identity, fingerprint=tag, source_files=snapshot,
                    writable_root=str(job), read_only_legacy_source=str(LEGACY))
    old = job/'manifest.json'
    if old.exists():
        assert json.loads(old.read_text()) == manifest, 'Manifest mismatch: never reuse this job.'
    else:
        atomic(old, manifest)
    shutil.copyfile(LEGACY/'fixed_policy_lp.py', job/'fixed_policy_lp.py')
    snapshot_directory = job/'source_snapshot'
    snapshot_directory.mkdir(exist_ok=True)
    for path in (Path(__file__), HERE/'n8_adapter.py', HERE/'ed_adapter.py', HERE/'lazy_upper.py'):
        shutil.copyfile(path, snapshot_directory/path.name)
    return job, manifest


def bootstrap(job):
    sys.path.insert(0, str(LEGACY))
    import core
    # All subsequently imported modules use from core import *.
    core.ROOT = job
    core.DATA = job/'data'
    core.DATA.mkdir(exist_ok=True)
    specification = json.loads((job/'manifest.json').read_text())['specification']
    if specification['short_max'] == 8:
        from n8_adapter import install
        install(core)
    if specification['ed_extension']:
        from ed_adapter import install
        install(core, specification['tree_leaves'], specification['pump_checks'], specification['exact_objective_quotient'])
    import bb84_joint_extension as search_engine
    import lazy_upper as upper_engine
    import terminal_equivalence
    original_equivalent = terminal_equivalence.solved_equivalent
    active_tolerance = [0.]

    def sufficiently_precise_equivalent(evaluated, ed, B):
        value = original_equivalent(evaluated, ed, B)
        if value is not None and value['upper_numerical']-value['lower'] <= active_tolerance[0]:
            return value
        return None

    terminal_equivalence.solved_equivalent = sufficiently_precise_equivalent
    # A cheap full-generator Watanabe pass establishes feasible incumbents
    # before the costly all-code screening. These are seeds, never a
    # restriction of the subsequently searched ED/B/code class.
    search_engine.seeds = lambda b, p, permissions: warm_seeds(core, search_engine, b, p, permissions)
    # Assert redirection on all loaded old-engine modules, not just core.
    for name, module in tuple(sys.modules.items()):
        origin = getattr(module, '__file__', '') or ''
        if str(LEGACY) in origin:
            if hasattr(module, 'DATA'):
                assert module.DATA == core.DATA, (name, module.DATA)
            if hasattr(module, 'ROOT'):
                assert module.ROOT == job, (name, module.ROOT)
    return core, search_engine, upper_engine, active_tolerance


def warm_seeds(core, engine, b, p, permissions):
    import numpy as np
    from recurrence_v2 import generate
    from recurrence_equivalence import representatives
    from literature_rates import watanabe
    saved = core.DATA/f'warm_seeds_{permissions}_{b:.12g}_{p:.12g}.json'
    if saved.exists():
        return json.loads(saved.read_text())['results']
    started = time.time()
    ts = np.array([0., 1/64, 1/8, 1/2, 1.])
    z = min(b, p)*ts
    initial = np.stack((1-b-p+z, b-z, z, p-z), -1)
    states, yields, memory, records = generate(initial, 7, permissions)
    representative = representatives(records, b == p)
    ids = np.flatnonzero(representative == np.arange(len(records)))
    candidates = []
    # K is read from the immutable job specification, not hard-coded here.
    K = json.loads((core.ROOT/'manifest.json').read_text())['specification']['classical_B_max']
    for B in range(K+1):
        sampled = (watanabe(states)*yields).min(-1)
        for budget in range(1, 9):
            allowed = ids[memory[ids] <= budget]
            chosen = allowed[np.argsort(-sampled[allowed])[:8]]
            candidates.extend((float(sampled[i]), int(i), B) for i in chosen if sampled[i] > 1e-12)
        states, success = core.b_step(states)
        yields = yields*success/2
    candidates = sorted(set(candidates), reverse=True)
    best = {f'M{m}': dict(lower=0., origin='discard') for m in range(1, 9)}
    strategy = [dict(AD=dict(kind='Watanabe_2007', n=2, mode='no_otp_hashed'), allocation_exact='1')]
    tested = 0
    for sampled, index, B in candidates:
        ed = records[index]
        incumbent = best[f'M{memory[index]}']['lower']
        if sampled <= incumbent+1e-11:
            continue
        numerical, _ = engine.numerical_worst(b, p, ed, B, strategy)
        if numerical <= incumbent+1e-11:
            continue
        cert = engine.certify_policy(b, p, ed, B, strategy, atol=max(1e-11, numerical*1e-5))
        tested += 1
        lower = max(0., cert['lower'])
        for budget in range(int(memory[index]), 9):
            if lower > best[f'M{budget}']['lower']:
                best[f'M{budget}'] = dict(lower=lower, ED=ed, classical_B_rounds=B,
                                          strategy=strategy, certificate=cert,
                                          origin='full_ED_B_Watanabe_feasible_seed')
        print('WARM FEASIBLE', b, p, ed, B, lower, flush=True)
    atomic(saved, dict(results=best, seconds=time.time()-started, certified_candidates=tested,
                       meaning='Feasible incumbents only; subsequent search and upper certificate cover every registered ED/B/AD action.'))
    return best


def required_gaps(results, args):
    return {key: max(args.absolute, args.relative*max(0., row['lower'])) for key, row in results.items()}


def certify_complete(source, upper_engine, tolerance):
    target = source.with_name(source.stem+'_menu_certified.json')
    if target.exists():
        old = json.loads(target.read_text())
        current_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        # The original engine's initial complete flag precedes its digest
        # check. Enforce both checks here, including changed primal/dual data.
        old['complete'] = False
        if old.get('source_digest') != current_digest:
            old['duals'] = {}
            old['rounds'] = []
        atomic(target, old)
    upper_engine.certify(source, tolerance=tolerance)
    result = json.loads(target.read_text())
    assert result.get('complete') and len(result['rounds']) == json.loads(source.read_text())['K']+1
    return result


def validate_lower_witnesses(src):
    for budget, row in src['results'].items():
        if row['lower'] <= 0:
            continue
        assert 'certificate' in row, (budget, 'missing whole-t certificate')
        assert row['certificate']['lower'] >= row['lower']-1e-18, budget
        assert row['ED']['memory_upper_bound'] <= int(budget[1:]), budget
        assert row['certificate']['lower'] <= row['certificate']['upper'], budget


def import_compatible_evidence(job, manifest, destination):
    """Replay primal/dual evidence across a proved quotient implementation.

    This never imports a whole-class upper certificate or a completion flag.
    Underlying generators, code kernels, and rate arithmetic must have the
    same hashes. The current complete scan and outward upper run still run.
    """
    if destination.exists():
        return
    target = manifest['specification']
    required = ['permissions', 'classical_B_max', 'nmax', 'short_max', 'ed_extension']
    if target['ed_extension']:
        required += ['tree_leaves', 'pump_checks']
    old_defaults = dict(short_max=7, ed_extension=False)
    candidates = []
    for path in (HERE/'runs').glob('*/manifest.json'):
        if path.parent == job:
            continue
        old = json.loads(path.read_text())
        spec = old['specification']
        if any(spec.get(key,old_defaults.get(key)) != target.get(key) for key in required):
            continue
        if any(Decimal(spec[key]) != Decimal(target[key]) for key in ('b', 'p')):
            continue
        mathematical_paths = [key for key in old['source_files']
                              if (not key.startswith('research/revision_20261006/bb84_search/')
                                  or (target['short_max'] == 8 and key.endswith('/n8_adapter.py'))
                                  or (target['ed_extension'] and key.endswith('/ed_adapter.py')))
                              and not key.startswith('research/revision_20261006/objective_equivalence/')]
        if any(manifest['source_files'].get(key) != old['source_files'][key] for key in mathematical_paths):
            continue
        source = path.parent/'data'/destination.name
        if not source.exists():
            continue
        data = json.loads(source.read_text())
        validate_lower_witnesses(data)
        candidates.append((len(data.get('evaluated', {})), source, data, old['fingerprint']))
    if not candidates:
        return
    _, source, data, fingerprint = max(candidates, key=lambda item: item[0])
    data['complete_candidate_screen'] = False
    data['screening'] = []
    data['imported_evidence'] = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                     source_fingerprint=fingerprint,
                                     meaning='Feasible policy lower certificates and candidate dual measures only; complete current-class screening and outward upper still required.')
    atomic(destination, data)
    for seed in source.parent.glob('warm_seeds_*.json'):
        shutil.copyfile(seed, destination.parent/seed.name)
    previous_upper = source.with_name(source.stem+'_menu_certified.json')
    if previous_upper.exists():
        upper = json.loads(previous_upper.read_text())
        # Individual interval dual bounds are feasible at this exact source
        # and unchanged arithmetic. Never import rounds or class-completion.
        evidence = {key: upper[key] for key in ('b','p','nmax','duals')}
        evidence['source_file'] = str(previous_upper)
        evidence['source_sha256'] = hashlib.sha256(previous_upper.read_bytes()).hexdigest()
        atomic(destination.parent/'imported_interval_dual_evidence.json', evidence)
    print('IMPORTED FEASIBLE EVIDENCE', source, 'fixed-menu rows', len(data.get('evaluated', {})), flush=True)


def replay_subset_incumbents(job, manifest, destination, engine):
    """Re-certify better witnesses from mathematically contained old classes.

    This avoids plotting a spurious tiny decrease when the enlarged-class
    discovery tolerance is looser than a completed old-class certificate.
    No old optimality claim is transferred to the larger class.
    """
    if not destination.exists():
        return
    data = json.loads(destination.read_text())
    target = manifest['specification']
    candidates = []
    for path in (HERE/'runs').glob('*/manifest.json'):
        if path.parent == job:
            continue
        old = json.loads(path.read_text())
        spec = old['specification']
        if (spec.get('permissions') != target['permissions']
            or any(Decimal(spec[key]) != Decimal(target[key]) for key in ('b','p'))
            or spec.get('short_max',7) > target['short_max']
            or spec.get('nmax',128) > target['nmax']
            or spec.get('classical_B_max',16) > target['classical_B_max']
            or (spec.get('ed_extension') and not target['ed_extension'])):
            continue
        # Legacy arithmetic and the old code catalog must remain identical.
        dependencies = [key for key in old['source_files']
                        if not key.startswith('research/revision_20261006/')]
        if any(manifest['source_files'].get(key) != old['source_files'][key] for key in dependencies):
            continue
        previous = path.parent/'data'/destination.name
        if not previous.exists():
            continue
        source = json.loads(previous.read_text())
        for budget,row in source.get('results',{}).items():
            if row['lower'] > data['results'][budget]['lower'] and row.get('ED',{}).get('family') != 'explicit_binary_tree':
                candidates.append((row['lower'],budget,row,str(previous)))
    seen = set()
    for _,budget,row,origin in sorted(candidates,reverse=True,key=lambda v:v[0]):
        key = json.dumps([row['ED'],row['classical_B_rounds'],row['strategy']],sort_keys=True)
        if key in seen or row['lower'] <= data['results'][budget]['lower']:
            continue
        seen.add(key)
        gap = row['certificate']['upper']-row['certificate']['lower']
        cert = engine.certify_policy(float(target['b']),float(target['p']),row['ED'],
            row['classical_B_rounds'],row['strategy'],atol=min(1e-11,max(1e-13,gap/2)))
        candidate = dict(row,lower=max(0.,cert['lower']),certificate=cert,
                         origin='replayed_stronger_subset_witness',subset_witness_source=origin)
        for m in range(row['ED']['memory_upper_bound'],9):
            if candidate['lower'] > data['results'][f'M{m}']['lower']:
                data['results'][f'M{m}'] = candidate
        print('REPLAYED SUBSET WITNESS',budget,candidate['lower'],origin,flush=True)
    if seen:
        data['complete_candidate_screen'] = False
        atomic(destination,data)


def run(args):
    job, manifest = prepare_job(args)
    summary_path = job/'summary.json'
    if summary_path.exists():
        saved = json.loads(summary_path.read_text())
        if saved.get('complete') and saved.get('fingerprint') == manifest['fingerprint']:
            for key, row in saved['results'].items():
                assert row['upper']-row['lower'] <= saved['required_gaps'][key]
            print('ALREADY VERIFIED', job, flush=True)
            return saved
    core, engine, upper_engine, active_tolerance = bootstrap(job)
    b, p = float(args.q), float(args.p or args.q)
    source = core.DATA/f'bb84_joint_extension_{args.permissions}_{b:.5f}_{p:.5f}_n{args.nmax}.json'
    import_compatible_evidence(job, manifest, source)
    replay_subset_incumbents(job, manifest, source, engine)
    started = time.time()
    rounds = []
    tolerance = args.discovery_tolerance
    outcome = None
    for iteration in range(args.max_refinements):
        active_tolerance[0] = tolerance
        atomic(job/'status.json', dict(stage='complete_candidate_discovery', iteration=iteration,
                                      fingerprint=manifest['fingerprint'], tolerance=tolerance, b=b, p=p))
        phase_start = time.time()
        src = engine.search(b, p, args.permissions, args.nmax, args.K, None, tolerance)
        assert src['complete_candidate_screen']
        validate_lower_witnesses(src)
        src['registered_class_manifest'] = manifest['fingerprint']
        src['scope'] = manifest['specification']
        atomic(source, src)
        discovery_seconds = time.time()-phase_start
        atomic(job/'status.json', dict(stage='whole_class_outward_upper', iteration=iteration,
                                      fingerprint=manifest['fingerprint'], tolerance=tolerance, b=b, p=p))
        phase_start = time.time()
        # Zero-rate budgets need an absolute upper bound even when the
        # discovery tolerance suffices for every visible positive curve.
        # Tighten the inexpensive upper screening first; do not force all
        # positive-policy LP/lower certificates to the smallest zero-rate
        # tolerance unless the resulting per-budget gaps actually require it.
        upper_precision = min(tolerance, args.absolute/2)
        certificate = certify_complete(source, upper_engine, upper_precision)
        certificate['registered_class_manifest'] = manifest['fingerprint']
        certificate['scope'] = manifest['specification']
        atomic(source.with_name(source.stem+'_menu_certified.json'), certificate)
        targets = required_gaps(certificate['results'], args)
        certificate['requested_budget_gaps'] = targets
        for key, row in certificate['results'].items():
            row['uniform_engine_converged'] = row['converged']
            row['converged'] = row['upper']-row['lower'] <= targets[key]
        atomic(source.with_name(source.stem+'_menu_certified.json'), certificate)
        missing = {key: row['gap_upper'] for key, row in certificate['results'].items()
                   if row['upper']-row['lower'] > targets[key]}
        rounds.append(dict(iteration=iteration, requested_global_tolerance=tolerance,
                           discovery_seconds=discovery_seconds, upper_seconds=time.time()-phase_start,
                           evaluated_fixed_menu_problems=len(src['evaluated']),
                           required_gaps=targets, unresolved_budget_gaps=missing))
        outcome = dict(complete=not missing, fingerprint=manifest['fingerprint'],
                       b=b, p=p, results=certificate['results'], required_gaps=targets,
                       phases=rounds, seconds=time.time()-started,
                       source=str(source.relative_to(job)),
                       certificate=str(source.with_name(source.stem+'_menu_certified.json').relative_to(job)),
                       interpretation='Certified complete registered recurrence/pump/B/AD class at this exact input; not unrestricted fixed-memory optimum and not the separate complete fixed-block union.')
        atomic(summary_path, outcome)
        if not missing:
            break
        tolerance = min(tolerance/10, min(targets[key] for key in missing)/2)
    atomic(job/'status.json', dict(stage='complete' if outcome['complete'] else 'needs_refinement',
                                  fingerprint=manifest['fingerprint'], results=outcome['results'],
                                  seconds=time.time()-started))
    print('POINT SUMMARY', json.dumps(outcome), flush=True)
    return outcome


def parser():
    cli = argparse.ArgumentParser()
    cli.add_argument('--q', required=True, type=str)
    cli.add_argument('--p', type=str)
    cli.add_argument('--permissions', choices=['real', 'full'], default='real')
    cli.add_argument('--nmax', type=int, default=128)
    cli.add_argument('--K', type=int, default=16)
    cli.add_argument('--short-max', type=int, choices=[7, 8], default=7)
    cli.add_argument('--ed-extension', action='store_true')
    cli.add_argument('--tree-leaves', type=int, default=5)
    cli.add_argument('--pump-checks', type=int, default=7)
    cli.add_argument('--no-objective-quotient', action='store_true')
    cli.add_argument('--absolute', type=float, default=1e-11)
    cli.add_argument('--relative', type=float, default=1e-4)
    cli.add_argument('--discovery-tolerance', type=float, default=1e-9)
    cli.add_argument('--max-refinements', type=int, default=3)
    return cli


if __name__ == '__main__':
    answer = run(parser().parse_args())
    if not answer['complete']:
        raise SystemExit(2)
