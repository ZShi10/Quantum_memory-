"""Fresh full-Clifford N<=5 flagged-block upper at a declared screening ceiling.

The ceiling is only an algorithmic pruning target, never a feasible rate.
Every entry and memory budget is retained.  The engine independently bounds
all public flags, the complete short-code menu, and the long endpoint codes.
"""
from pathlib import Path
import argparse, hashlib, json, sys, time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ENGINE = PROJECT / 'research/revision_20261006/code_extension/bb84_m5/audit_upper.py'


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.writing.json')
    temporary.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--q', type=float, required=True)
    parser.add_argument('--ceiling', type=float, required=True)
    parser.add_argument('--test-t', nargs='+', default=['0', '.125', '.1875', '.25', '1'])
    args = parser.parse_args()
    assert 0 < args.q < .25 and args.ceiling > 0
    task = HERE / 'bb84' / str(args.q) / 'block_component'
    task.mkdir(parents=True, exist_ok=True)
    source = ENGINE.read_text(encoding='utf-8')
    old = "job=HERE/'upper_runs'/tag"
    assert source.count(old) == 1
    from fast_entropy import install as install_entropy
    from fast_short_modes import install as install_paid_modes
    def accelerated_arithmetic():install_entropy();install_paid_modes()
    namespace = dict(__name__='fresh_block_component', __file__=str(ENGINE), OUTPUT=task,FAST_INSTALL=accelerated_arithmetic)
    hook='    import stable_intervals as iv\n';assert source.count(hook)==1
    modified=source.replace(old,'job=OUTPUT/tag').replace(hook,hook+'    FAST_INSTALL()\n')
    exec(compile(modified,str(ENGINE),'exec'),namespace)

    def targets(b, p, permission, explicit=()):
        assert b == p == args.q and permission == 'full' and not explicit
        return [0.] + [args.ceiling] * 8, [], {}

    namespace['lower_sources'] = targets
    options = argparse.Namespace(p=args.q, permissions='full', K=16, nmax=128,
        tolerance=1e-9, quantum=1e-12, lower_file=[], test_t=args.test_t)
    start = time.monotonic()
    result = namespace['run'](options, args.q)
    original = task / result['job_fingerprint'] / 'result.json'
    # Leave the original engine output intact. This explicit interpretation is
    # the exported result; the engine's historical lower_witness field contains
    # the numerical pruning target, NOT a block lower witness.
    exported = dict(complete=result['complete'], Q=args.q, all_memory_budgets=True,
        permissions='full', successful_input_cap=5, all_public_flags=True,
        B_max=16, general_code_max=8, long_code_max=128,
        ceiling=args.ceiling, tolerance=options.tolerance, test_t=args.test_t,
        source_sha256=hashlib.sha256(ENGINE.read_bytes()).hexdigest(),
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        arithmetic_backends={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ('fast_entropy.py','fast_short_modes.py')},
        engine_output=str(original), engine_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        results={key:dict(upper=row['upper']) for key,row in result['results'].items()},
        seconds=time.monotonic()-start,
        scope='Entire full-Clifford N<=5 complete flagged-block component, full-input attempt cost, every registered minimal-width realization and complete stated AD menu. Ceiling is only an upper-screening target and supplies no achievable rate or optimum.')
    save(task / 'result.json', exported)
    print(json.dumps(exported, indent=2), flush=True)


if __name__ == '__main__':
    main()
