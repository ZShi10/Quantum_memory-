"""Replay fixed-block refinements using only this campaign's fresh uppers.

The test correlations are an explicit rational grid, not archived numerical
upper endpoints. Each new per-instrument bound is intersected with the fresh
base bound by the existing pointwise engine. Historical result caches are not
available in the isolated output directory.
"""
from pathlib import Path
import argparse, hashlib, json, os, sys, time

HERE = Path(__file__).resolve().parent
REV = HERE.parent
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
sys.dont_write_bytecode = True
GRID = ['0', '.025', '.05', '.075', '.1', '.125', '.15', '.175',
        '.2', '.225', '.25', '.3', '.375', '.5', '1']


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_suffix('.writing.json')
    tmp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def run(q):
    base_receipt = HERE / 'runs/bb84-block' / q / 'recompute_receipt.json'
    base_record = read(base_receipt)
    assert base_record['status'] == 'complete'
    base = Path(base_record['output'])
    assert base.is_relative_to(HERE / 'runs/bb84-block')
    assert sha(base) == base_record['output_sha256']
    task = HERE / 'runs/bb84-block-refined' / q
    task.mkdir(parents=True, exist_ok=True)
    receipt = task / 'recompute_receipt.json'
    if receipt.exists() and read(receipt).get('status') == 'complete':
        assert read(receipt)['fresh_base_sha256'] == sha(base)
        print('Already freshly refined', q, flush=True)
        return
    source = REV / 'reporting/core_refinement/refine_pointwise.py'
    record = dict(status='running', component='bb84-block-refined', Q=q,
                  historical_upper_values_used=False, wrapper_sha256=sha(Path(__file__)),
                  fresh_base=str(base), fresh_base_sha256=sha(base),
                  fresh_base_receipt_sha256=sha(base_receipt), test_t=GRID)
    save(receipt, record)
    start = time.monotonic()
    try:
        code = source.read_text(encoding='utf-8')
        old = "job=HERE/'runs'/fingerprint;job.mkdir(parents=True,exist_ok=True)"
        new = "job=RECHECK_OUTPUT/fingerprint;job.mkdir(parents=True,exist_ok=True)"
        assert code.count(old) == 1
        ns = dict(__name__='fresh_refine_block', __file__=str(source),
                  RECHECK_OUTPUT=task)
        exec(compile(code.replace(old, new), str(source), 'exec'), ns)
        sys.argv = [str(source), '--base', str(base), '--test-t', *GRID]
        ns['main']()
        outputs = list(task.glob('*/result.json'))
        assert len(outputs) == 1
        output = outputs[0]
        result = read(output)
        assert result['complete']
        assert result['inputs'][str(base.relative_to(REV.parents[1]))] == sha(base)
        record.update(status='complete', seconds=time.monotonic()-start,
                      source_sha256=sha(source), output=str(output),
                      output_sha256=sha(output), results=result['results'],
                      refined_instruments=len(result['refinements']),
                      all_instruments=len(result['entries']),
                      union_gap_closed=result['union_gap_closed'],
                      output_path_patch_only=dict(before=old, after=new))
        save(receipt, record)
        print('FRESH BLOCK REFINEMENT', q, result['union_gap_closed'], flush=True)
    except BaseException as exc:
        record.update(status='failed', seconds=time.monotonic()-start, error=repr(exc))
        save(receipt, record)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--q', nargs='+', required=True)
    args = parser.parse_args()
    for q in args.q:
        run(q)
