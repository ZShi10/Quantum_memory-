"""Noise-independent, explicitly bounded coverage; no fidelity pruning."""
from itertools import product
from functools import lru_cache
from math import comb
import numpy as np
from .model import RAW, Node, PRIMITIVES, pumping_tree, structural_record, Evaluator, terminal_memory, terminal_plan, AXIS_PERMUTATIONS


def ordered_tree_layers(max_leaves=5, decoders=tuple(PRIMITIVES)):
    """All ordered full binary trees with <=max_leaves, all decoder labels.

    Both child orders are retained even on an IID source: asymmetric decoders
    and heterogeneous child laws make silent left/right quotienting unsafe.
    """
    if max_leaves < 1 or not decoders or len(set(decoders)) != len(decoders):
        raise ValueError('Positive leaf bound and distinct nonempty decoders are required')
    if any(d not in PRIMITIVES for d in decoders):
        raise ValueError('Unknown decoder')
    layers = {1: [RAW]}
    for n in range(2, max_leaves + 1):
        layers[n] = [Node(d, left, right) for k in range(1, n)
                     for left in layers[k] for right in layers[n-k] for d in decoders]
        assert len(layers[n]) == comb(2*n-2, n-1) // n * len(decoders)**(n-1)
    return layers


def variable_pumps(max_checks=7, decoders=tuple(PRIMITIVES)):
    """All chronological variable-decoder raw-pump words, including empty."""
    if max_checks < 0 or not decoders or any(d not in PRIMITIVES for d in decoders):
        raise ValueError('Nonnegative check bound and valid decoders are required')
    for n in range(max_checks + 1):
        for word in product(decoders, repeat=n):
            yield ''.join(word), pumping_tree(word)


@lru_cache(None)
def structural_catalog(max_leaves=5, max_pump_checks=7):
    """Deterministic structural union, reused by floating and interval scans."""
    families = {}
    for _, layer in ordered_tree_layers(max_leaves).items():
        for tree in layer:
            families.setdefault(tree, []).append('complete_ordered_tree_leaf_bound')
    for word, tree in variable_pumps(max_pump_checks):
        families.setdefault(tree, []).append('variable_decoder_raw_pump')
    return tuple((tree, tuple(memberships)) for tree, memberships in families.items())


def candidate_records(max_leaves=5, max_pump_checks=7, permissions='real'):
    """Distinct structural trees in the union, paired with allowed final axes."""
    if permissions not in ('real', 'enlarged_axes', 'full'):
        raise ValueError('Unknown permission class')
    axes = ('Z', 'X') if permissions == 'real' else ('Z', 'X', 'Y')
    normalized = 'real' if permissions == 'real' else 'enlarged_axes'
    for tree, memberships in structural_catalog(max_leaves, max_pump_checks):
        # Share the serialized immutable tree across axes, avoiding needless
        # copies of its ancestry in the large catalog held by the optimizer.
        base = structural_record(tree, axes[0], normalized)
        for axis in axes:
            yield tree, {**base, 'final_axis': axis, 'terminal_memory': terminal_memory(tree, axis),
                         'memory_upper_bound': terminal_memory(tree, axis),
                         'terminal_implementation': terminal_plan(tree, axis)[1],
                         'family_memberships': list(memberships)}


def generate(p, permissions='real', max_leaves=5, max_pump_checks=7):
    """Old-compatible (states,yields,memories,records), source shape (...,4).

    Catalog order matches generate_bounds exactly. `full` is an accepted
    compatibility alias for enlarged_axes, not full Clifford coverage.
    """
    evaluator = Evaluator(p)
    ps, ys, memories, records = [], [], [], []
    for tree, record in candidate_records(max_leaves, max_pump_checks, permissions):
        value = evaluator.retained(tree)
        ps.append(value['state'][..., list(AXIS_PERMUTATIONS[record['final_axis']])])
        ys.append(value['yield_per_raw_pair'])
        memories.append(record['memory_upper_bound'])
        records.append(record)
    return np.stack(ps), np.stack(ys), np.asarray(memories, int), records


def coverage(max_leaves=5, max_pump_checks=7):
    tree_layers = {n: comb(2*n-2, n-1) // n * 4**(n-1) for n in range(1, max_leaves+1)}
    pumps = {d: 4**d for d in range(max_pump_checks+1)}
    overlap = sum(4**d for d in range(min(max_leaves-1, max_pump_checks)+1))
    unique = sum(tree_layers.values()) + sum(pumps.values()) - overlap
    return dict(ordered_tree_leaf_bound=max_leaves, variable_pump_check_bound=max_pump_checks,
                all_labeled_ordered_trees_by_leaves=tree_layers,
                all_variable_pumps_by_checks=pumps, duplicate_trees_between_families=overlap,
                unique_structural_trees=unique, real_terminal_candidates=2*unique,
                enlarged_axis_terminal_candidates=3*unique,
                complete_within_declared_syntax=True, all_memory_bounded_protocols=False,
                excludes=['other Clifford primitives', 'nonbinary primitives', 'multiple quantum outputs',
                          'outcome-dependent decoder selection', 'failed-branch recycling',
                          'unbounded ancestry', 'cross-tree quantum pooling', 'arbitrary non-Clifford operations'])
