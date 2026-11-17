"""In-memory union adapter for explicit trees and variable raw pumping.

Installation precedes the joint-search/upper imports. No legacy source is
edited. Extension records keep exact structural identities: no unproved
Clifford, fidelity, child-order, or schedule quotient is used here.
"""
from pathlib import Path
import hashlib
import json
import sys
from functools import lru_cache
import numpy as np

HERE = Path(__file__).resolve().parent
EXTENSION = HERE.parent/'resource_model'


def fingerprint():
    files = list(EXTENSION.glob('*.py'))
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)}


def install(core, max_leaves=5, max_pump_checks=7, objective_quotient='none'):
    assert 'bb84_joint_extension' not in sys.modules, 'Install before joint-search imports.'
    assert 'certify_bb84_joint_menu' not in sys.modules, 'Install before whole-menu upper imports.'
    sys.path.insert(0, str(HERE.parent))
    from resource_model import catalog, bounds, model
    import recurrence_v2 as recurrence
    import recurrence_interval_library as interval_library
    import recurrence_equivalence as equivalence
    import bb84_recursive
    import interval_jets
    import certify_bb84_recursive
    original_generate = recurrence.generate
    original_generate_bounds = interval_library.generate_bounds
    original_records = interval_library.records
    original_key = equivalence.key
    original_representatives = equivalence.representatives
    original_prepare = bb84_recursive.prepare
    original_jet_prepare = interval_jets.prepare
    original_interval_prepare = certify_bb84_recursive.prepare
    quotient_maps = None
    quotient_fingerprint = None
    if objective_quotient != 'none':
        assert objective_quotient == 'equal_xz' and (max_leaves, max_pump_checks) == (5, 7)
        from objective_equivalence import api as exact_objectives
        quotient_maps = {permission: exact_objectives.representative_map(permission, 'equal_xz')
                         for permission in ('real', 'full')}
        quotient_fingerprint = exact_objectives.fingerprint('equal_xz')

    @lru_cache(None)
    def extension_lookup(permission):
        return {(row['canonical_tree'], row['final_axis']): index
                for index, (_, row) in enumerate(catalog.candidate_records(max_leaves, max_pump_checks, permission))}

    def generate(p, depth=7, permissions='real', rmax=160):
        first = original_generate(p, depth, permissions, rmax)
        second = catalog.generate(p, permissions, max_leaves, max_pump_checks)
        return (np.concatenate((first[0], second[0])), np.concatenate((first[1], second[1])),
                np.concatenate((first[2], second[2])), first[3]+second[3])

    def generate_bounds(initial, permissions='real', depth=7, rmax=160):
        first = original_generate_bounds(initial, permissions, depth, rmax)
        if not hasattr(bounds, 'generate_bounds'):
            raise RuntimeError('Resource-model batched interval catalog must be available for an enlarged-class proof.')
        second = bounds.generate_bounds(initial, permissions, max_leaves, max_pump_checks)
        return tuple(tuple(np.concatenate((first[k][j], second[k][j])) for j in (0, 1)) for k in (0, 1))

    @lru_cache(None)
    def records(permissions='real', depth=7, rmax=160):
        first_memory, first_records = original_records(permissions, depth, rmax)
        second_records = [record for _, record in catalog.candidate_records(max_leaves, max_pump_checks, permissions)]
        second_memory = np.array([record['memory_upper_bound'] for record in second_records])
        return np.concatenate((first_memory, second_memory)), first_records+second_records

    def key(record):
        if record['family'] == 'explicit_binary_tree':
            if quotient_maps is not None:
                permission = 'real' if record['permissions'] == 'real' else 'full'
                index = extension_lookup(permission)[record['canonical_tree'], record['final_axis']]
                representative = int(quotient_maps[permission][index])
                return ('exact_equal_xz_objective', quotient_fingerprint['map_sha256'], permission, representative)
            return ('explicit_binary_tree', record['canonical_tree'], record['final_axis'],
                    record['retry_rule'], record['memory_upper_bound'])
        return original_key(record)

    def representatives(rr, symmetric=True):
        if not symmetric:
            return np.arange(len(rr))
        if quotient_maps is None:
            return original_representatives(rr, symmetric)
        boundary = next((i for i, row in enumerate(rr) if row['family'] == 'explicit_binary_tree'), len(rr))
        if boundary == len(rr):
            return original_representatives(rr, symmetric)
        first = original_representatives(rr[:boundary], symmetric)
        permission = 'real' if rr[boundary]['permissions'] == 'real' else 'full'
        extension = quotient_maps[permission]
        assert len(rr)-boundary == len(extension)
        assert all(row['family'] == 'explicit_binary_tree' for row in rr[boundary:])
        memory = np.array([row['memory_upper_bound'] for row in rr[boundary:]])
        assert np.all(memory[extension] <= memory)
        return np.concatenate((first, extension+boundary))

    def prepare(b, p, ts, record, rounds=0):
        if record['family'] != 'explicit_binary_tree':
            return original_prepare(b, p, ts, record, rounds)
        ts = np.asarray(ts)
        z = min(b, p)*ts
        initial = np.stack((1-b-p+z, b-z, z, p-z), -1)
        value = model.evaluate(model.tree_from_dict(record['tree']), initial, record['final_axis'])
        state, y = value['state'], value['yield_per_raw_pair']
        for _ in range(rounds):
            state, success = core.b_step(state)
            y = y*success/2
        return state, y

    def prepare_jets(b, p, intervals, record, rounds=0):
        if record['family'] == 'explicit_binary_tree':
            return bounds.prepare_jets(b, p, intervals, record, rounds)
        return original_jet_prepare(b, p, intervals, record, rounds)

    def prepare_intervals(b, p, intervals, record, rounds=0):
        if record['family'] == 'explicit_binary_tree':
            initial = certify_bb84_recursive.source(b, p, intervals)
            return bounds.prepare_intervals(initial, record, rounds)
        return original_interval_prepare(b, p, intervals, record, rounds)

    recurrence.generate = generate
    interval_library.generate_bounds = generate_bounds
    interval_library.records = records
    equivalence.key = key
    equivalence.representatives = representatives
    bb84_recursive.prepare = prepare
    interval_jets.prepare = prepare_jets
    certify_bb84_recursive.prepare = prepare_intervals
    return dict(generate=generate, generate_bounds=generate_bounds, records=records,
                prepare=prepare, prepare_jets=prepare_jets, prepare_intervals=prepare_intervals,
                coverage=catalog.coverage(max_leaves, max_pump_checks),
                objective_quotient=quotient_fingerprint)


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    project = HERE.parents[2]
    sys.path.insert(0, str(project/'research/revision_20260922'))
    import core
    core.ROOT = HERE/'ed_adapter_validation'
    core.DATA = core.ROOT/'data'
    core.DATA.mkdir(parents=True, exist_ok=True)
    api = install(core, max_leaves=4, max_pump_checks=4)
    from fractions import Fraction
    from certify_bb84_recursive import source
    from recurrence_equivalence import representatives
    initial = np.array([[.6,.1,.1,.2], [.5,.25,0.,.25]])
    point_bounds = (initial, initial)
    ps, ys, memory, rr = api['generate'](initial, depth=3, permissions='real', rmax=4)
    pp, yy = api['generate_bounds'](point_bounds, permissions='real', depth=3, rmax=4)
    mm, exact_records = api['records']('real', 3, 4)
    assert rr == exact_records and np.array_equal(memory, mm)
    assert np.all(ps >= pp[0]-3e-13) and np.all(ps <= pp[1]+3e-13)
    assert np.all(ys >= yy[0]-3e-13) and np.all(ys <= yy[1]+3e-13)
    rep = representatives(rr, True)
    extension_ids = [i for i, row in enumerate(rr) if row['family'] == 'explicit_binary_tree']
    assert all(rep[i] == i for i in extension_ids), 'No unproved tree quotient allowed.'
    probe = rr[-1]
    t = [Fraction(0), Fraction(1,7), Fraction(1)]
    fp, fy = api['prepare'](.23,.23,np.array(t,float),probe,2)
    ip, iy = api['prepare_intervals'](.23,.23,[(v,v) for v in t],probe,2)
    jp, jy = api['prepare_jets'](.23,.23,[(v,v) for v in t],probe,2)
    for lo, hi in (ip, jp.v):
        assert np.all(fp >= lo-3e-13) and np.all(fp <= hi+3e-13)
    for lo, hi in (iy, jy.v):
        assert np.all(fy >= lo-3e-13) and np.all(fy <= hi+3e-13)
    result = dict(status='passed', candidate_count=len(rr), extension_count=len(extension_ids),
                  floating_interval_values=int(ps.size+ys.size), dispatch_checks=int(fp.size+fy.size),
                  coverage=api['coverage'], source_hashes=fingerprint())
    core.atomic_json(core.DATA/'validation.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'source_hashes'}))
