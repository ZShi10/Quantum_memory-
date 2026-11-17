"""Attach immutable current feasible lower witnesses to saved block uppers.

A saved block upper is valid even when the external lower file used only
as a pruning target has since advanced. This records a fresh dominance
comparison, with exact source snapshots, without changing any upper run.
"""
from pathlib import Path
import json, hashlib
from fractions import Fraction

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
OUTPUT = HERE / 'upper_summary'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def snapshot_lowers(b, p, permission):
    name = f'bb84_joint_extension_{permission}_{b:.5f}_{p:.5f}_n128.json'
    files = [PROJECT / 'research/revision_20260922/data' / name]
    files += list((PROJECT / 'research/revision_20261006/bb84_search/runs')
                  .glob(f'*/data/{name}'))
    lower = [0.] * 9
    witnesses = [None] * 9
    snapshots = []
    for path in sorted(set(files)):
        if not path.exists():
            continue
        raw = path.read_bytes()
        source = json.loads(raw)
        assert source['permissions'] == permission
        assert source['b'] == b and source['p'] == p
        digest = sha(raw)
        accepted = False
        for key, row in source.get('results', {}).items():
            if not key.startswith('M') or not isinstance(row, dict):
                continue
            budget = int(key[1:])
            assert 1 <= budget <= 8
            value = float(row.get('lower', 0.))
            cert = row.get('certificate')
            if value <= 0 or not cert or float(cert.get('lower', -1)) < value:
                continue
            assert row.get('ED', {}).get('memory_upper_bound', budget) <= budget
            accepted = True
            for k in range(budget, 9):
                if value > lower[k]:
                    lower[k] = value
                    witnesses[k] = dict(source_sha256=digest,
                                        source_memory_budget=budget,
                                        row=key)
        if accepted:
            target = OUTPUT / 'lower_sources' / f'{digest}.json'
            if target.exists():
                assert sha(target.read_bytes()) == digest
            else:
                target.write_bytes(raw)
            snapshots.append(dict(source=str(path.relative_to(PROJECT)),
                                  sha256=digest,
                                  snapshot=str(target.relative_to(PROJECT))))
    return lower, witnesses, snapshots


def run():
    (OUTPUT / 'lower_sources').mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted((HERE / 'upper_runs').glob('*/result.json')):
        raw = path.read_bytes()
        result = json.loads(raw)
        if not result['complete']:
            continue
        assert result['general_code_max'] == 8
        assert result['fixed_block_max'] == 5
        assert result['B_max'] == 16 and result['long_max'] == 128
        matches = {rel: (PROJECT / rel).exists() and
                   sha((PROJECT / rel).read_bytes()) == digest
                   for rel, digest in result['inputs'].items()}
        assert all(matches.values()), matches
        old_lower_matches = {rel: (PROJECT / rel).exists() and
                             sha((PROJECT / rel).read_bytes()) == digest
                             for rel, digest in result['lower_sources'].items()}
        lower, witnesses, snapshots = snapshot_lowers(
            result['b'], result['p'], result['permissions'])
        comparisons = {}
        for k in range(1, 9):
            upper = result['results'][f'M{k}']['upper']
            gap = Fraction(upper) - Fraction(lower[k])
            comparisons[f'M{k}'] = dict(
                core_upper=upper, registered_lower=lower[k],
                lower_witness=witnesses[k],
                strictly_dominated=gap < 0,
                dominated_within_tolerance=gap <= Fraction(result['tolerance']))
        rows.append(dict(
            b=result['b'], p=result['p'], permission=result['permissions'],
            upper_result=str(path.relative_to(PROJECT)),
            upper_result_sha256=sha(raw),
            arithmetic_sources_currently_match=matches,
            original_external_lower_sources_currently_match=old_lower_matches,
            current_lower_snapshots=snapshots,
            absolute_tolerance=result['tolerance'], comparisons=comparisons,
            all_budgets_dominated_within_tolerance=all(
                v['dominated_within_tolerance'] for v in comparisons.values())))
    rows.sort(key=lambda v: (v['permission'], v['b'], v['p'], v['upper_result']))
    doc = dict(
        status='snapshot_comparison_of_completed_upper_runs', results=rows,
        note=('Each referenced result already supplies an outward block '
              'upper. Lower witnesses affect computational pruning targets '
              'but are not assumptions in the validity of those uppers: '
              'test correlations are legal states, and pruned branches '
              'retain their actual outward upper. This record attaches '
              'current certified feasible lower witnesses with immutable '
              'source snapshots. It does not certify the entire registered '
              'class upper or an unrestricted fixed-memory optimum. '
              'A one-register gap within tolerance is not zero capacity.'))
    raw = json.dumps(doc, indent=2).encode()
    digest = sha(raw)
    target = OUTPUT / f'comparison_{digest}.json'
    target.write_bytes(raw)
    (OUTPUT / 'latest.json').write_text(json.dumps(dict(
        comparison=str(target.relative_to(PROJECT)), sha256=digest), indent=2))
    for row in rows:
        print(row['permission'], row['b'],
              row['all_budgets_dominated_within_tolerance'],
              {k: (v['core_upper'], v['registered_lower'])
               for k, v in row['comparisons'].items() if k in ('M1', 'M2', 'M3')})
    print(target)


if __name__ == '__main__':
    run()
