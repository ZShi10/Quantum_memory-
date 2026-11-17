"""Sequential adaptive actual-Q evaluations; never smooth a saved catalog.

Every admitted point calls run_point.py, hence scans and bounds the complete
registered class. Plot interpolation is NOT asserted to certify unsampled Q.
The explicit job ceiling makes expensive refinement inspectable/resumable.
"""
from pathlib import Path
from decimal import Decimal
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_point


def qstr(q):
    return format(Decimal(str(q)).normalize(), 'f')


def point_args(q, settings):
    arguments = ['--q', qstr(q), '--permissions', settings.permissions,
        '--absolute', str(settings.absolute), '--relative', str(settings.relative),
        '--discovery-tolerance', str(settings.discovery_tolerance), '--short-max', str(settings.short_max),
        '--tree-leaves', str(settings.tree_leaves), '--pump-checks', str(settings.pump_checks)]
    if settings.ed_extension:
        arguments.append('--ed-extension')
    if settings.no_objective_quotient:
        arguments.append('--no-objective-quotient')
    return run_point.parser().parse_args(arguments)


def job_for(q, settings):
    job, _ = run_point.prepare_job(point_args(q, settings))
    return job


def read_point(q, settings):
    job = job_for(q, settings)
    path = job/'summary.json'
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    if not row.get('complete'):
        return None
    source = json.loads((job/row['source']).read_text())
    policies = {}
    for key, policy in source['results'].items():
        actions = [v['AD'] for v in policy.get('strategy', [])]
        policies[key] = json.dumps([policy.get('ED'), policy.get('classical_B_rounds'), actions], sort_keys=True)
    return dict(q=qstr(q), results=row['results'], policies=policies,
                fingerprint=row['fingerprint'], job=str(job), seconds=row['seconds'])


def run_one(q, settings):
    existing = read_point(q, settings)
    if existing:
        return existing, False
    job = job_for(q, settings)
    command = [sys.executable, str(HERE/'run_point.py'), '--q', qstr(q),
               '--permissions', settings.permissions, '--absolute', str(settings.absolute),
               '--relative', str(settings.relative), '--discovery-tolerance', str(settings.discovery_tolerance),
               '--short-max', str(settings.short_max), '--tree-leaves', str(settings.tree_leaves),
               '--pump-checks', str(settings.pump_checks)]
    if settings.ed_extension:
        command.append('--ed-extension')
    if settings.no_objective_quotient:
        command.append('--no-objective-quotient')
    with (job/'process.log').open('ab') as output:
        process = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
    result = read_point(q, settings)
    if process.returncode or not result:
        raise RuntimeError(f'Point {q} did not meet all requested gaps; inspect {job}/process.log')
    return result, True


def interval_error(a, m, b, rate_floor):
    worst = 0.
    changed = False
    fraction = float((Decimal(m['q'])-Decimal(a['q']))/(Decimal(b['q'])-Decimal(a['q'])))
    for key in ('M1', 'M2', 'M4', 'M8'):
        ra, rm, rb = (row['results'][key]['lower'] for row in (a, m, b))
        if max(ra, rm, rb) <= rate_floor:
            continue
        la, lm, lb = [math.log10(max(rate_floor, rate)) for rate in (ra, rm, rb)]
        worst = max(worst, abs(lm-((1-fraction)*la+fraction*lb)))
        changed |= a['policies'][key] != m['policies'][key] or m['policies'][key] != b['policies'][key]
    return worst, changed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--q', nargs='+', default=['.225', '.227', '.23', '.235'])
    p.add_argument('--adaptive', action='store_true')
    p.add_argument('--max-new-points', type=int, default=4)
    p.add_argument('--minimum-spacing', type=Decimal, default=Decimal('0.00001'))
    p.add_argument('--log-chord-tolerance', type=float, default=.002)
    p.add_argument('--rate-floor', type=float, default=1e-7)
    p.add_argument('--permissions', choices=['real', 'full'], default='real')
    p.add_argument('--absolute', type=float, default=1e-11)
    p.add_argument('--relative', type=float, default=1e-4)
    p.add_argument('--discovery-tolerance', type=float, default=1e-9)
    p.add_argument('--short-max', type=int, choices=[7,8], default=7)
    p.add_argument('--ed-extension', action='store_true')
    p.add_argument('--tree-leaves', type=int, default=5)
    p.add_argument('--pump-checks', type=int, default=7)
    p.add_argument('--no-objective-quotient', action='store_true')
    args = p.parse_args()
    spec = {key: str(value) if isinstance(value, Decimal) else value for key, value in vars(args).items()}
    # The point source manifest needs a scalar Q and its own full options;
    # the sweep namespace contains a list of Q values instead.
    spec['source_digest'] = run_point.digest_json(run_point.source_snapshot(point_args(args.q[0], args)))
    # The run ceiling is scheduling, not mathematical content.
    identity = {key: value for key, value in spec.items() if key != 'max_new_points'}
    tag = run_point.digest_json(identity)
    folder = HERE/'sweeps'/tag
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder/'sweep.json'
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else dict(specification=spec, points={}, intervals=[], complete=False)
    initial = sorted(set(Decimal(q) for q in args.q))
    pending = [(a, b) for a, b in zip(initial, initial[1:])]
    # Preserve an explicit user/research priority order for initial inputs.
    # Only the later adaptive interval queue needs numeric sorting.
    requests = list(dict.fromkeys(Decimal(q) for q in args.q))
    started = time.time()
    launched = 0

    def ensure(q):
        nonlocal launched
        key = qstr(q)
        result = read_point(q, args)
        if result is None:
            if launched >= args.max_new_points:
                return None
            print('START ACTUAL COMPLETE-CLASS INPUT', key, flush=True)
            result, new = run_one(q, args)
            launched += int(new)
        state['points'][key] = result
        state.update(new_points_this_run=launched, elapsed_seconds=time.time()-started)
        run_point.atomic(checkpoint, state)
        return result

    for q in requests:
        if ensure(q) is None:
            state['stop_reason'] = 'explicit_new_point_budget'
            run_point.atomic(checkpoint, state)
            return
    if args.adaptive:
        while pending:
            a, b = pending.pop(0)
            if b-a <= args.minimum_spacing:
                state['intervals'].append(dict(a=qstr(a), b=qstr(b), status='minimum_spacing'))
                continue
            m = (a+b)/2
            middle = ensure(m)
            if middle is None:
                state['pending_intervals'] = [(qstr(a), qstr(b))]+[(qstr(x), qstr(y)) for x, y in pending]
                state['stop_reason'] = 'explicit_new_point_budget'
                run_point.atomic(checkpoint, state)
                return
            error, switch = interval_error(state['points'][qstr(a)], middle, state['points'][qstr(b)], args.rate_floor)
            refine = error > args.log_chord_tolerance or switch
            state['intervals'].append(dict(a=qstr(a), midpoint=qstr(m), b=qstr(b), log_chord_error=error,
                                          protocol_switch=switch, status='refine' if refine else 'accepted_drawing_resolution'))
            if refine:
                pending.extend(((a, m), (m, b)))
    state.update(complete=True, stop_reason='all_requested_actual_points_certified',
                 interpretation='Complete registered-class pointwise bounds; midpoint tests assess plotting resolution only, not continuum optimality.')
    run_point.atomic(checkpoint, state)
    print('SWEEP COMPLETE', checkpoint, 'actual inputs', len(state['points']), flush=True)


if __name__ == '__main__':
    main()
