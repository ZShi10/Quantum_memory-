"""Binary ED trees: raw-pair cost, explicit scheduling, and terminal reductions.

Each parent prepares successful independent children. A failed child is retried
locally while the other successful child is held. A failed parent discards both
children and restarts the parent. This is a protocol definition, not a claim
that every memory-bounded Clifford protocol has this form.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import math
import numpy as np


@dataclass(frozen=True)
class Raw:
    """One IID raw pair. The incoming local qubit counts toward memory."""


RAW = Raw()


@dataclass(frozen=True)
class Node:
    decoder: str
    left: Raw | Node
    right: Raw | Node

    def __post_init__(self):
        if self.decoder not in PRIMITIVES:
            raise ValueError(f"Unknown decoder: {self.decoder}")
        if not isinstance(self.left, (Raw, Node)) or not isinstance(self.right, (Raw, Node)):
            raise TypeError("Children must be Raw or Node objects")


# The gates run chronologically, with qubit indices 0 and 1. Measure wire 1,
# accept equal outcomes, and retain wire 0. Operators below may differ from
# gate-conjugated logical operators by the accepted check.
PRIMITIVES = {
    'Z': dict(check='ZZ', logical_X='XX', logical_Z='ZI', gates=(('cx', 0, 1), ('cx', 1, 0))),
    'X': dict(check='XX', logical_X='ZZ', logical_Z='XI', gates=(('h', 0), ('h', 1), ('cx', 0, 1), ('cx', 1, 0))),
    'R': dict(check='YY', logical_X='ZX', logical_Z='XX', gates=(('cx', 0, 1), ('h', 0), ('cx', 0, 1), ('cx', 1, 0))),
    'C': dict(check='XZ', logical_X='ZX', logical_Z='XI', gates=(('h', 0), ('cx', 0, 1), ('cx', 1, 0))),
}
PAULI_BITS = {'I': (0, 0), 'X': (1, 0), 'Y': (1, 1), 'Z': (0, 1)}
BELL_BITS = tuple(PAULI_BITS[a] for a in ('I', 'X', 'Y', 'Z'))
AXIS_PERMUTATIONS = {'Z': (0, 1, 2, 3), 'X': (0, 3, 2, 1), 'Y': (0, 1, 3, 2)}
TERMINAL_REDUCTIONS = {('Z', 'Z'): ('Z', 'Z'), ('X', 'Z'): ('X', 'X'),
                       ('R', 'Y'): ('Y', 'Y'), ('C', 'Z'): ('X', 'Z')}
RETRY_RULE = 'retry_failed_child_locally;discard_both_children_and_restart_parent_on_parent_failure'


def parity(error, operator):
    return sum(x * PAULI_BITS[q][1] + z * PAULI_BITS[q][0]
               for (x, z), q in zip(error, operator)) % 2


def decoder_events(decoder):
    item = PRIMITIVES[decoder]
    result = []
    for i, a in enumerate(BELL_BITS):
        for j, b in enumerate(BELL_BITS):
            if parity((a, b), item['check']) == 0:
                out = (parity((a, b), item['logical_Z']), parity((a, b), item['logical_X']))
                result.append((i, j, BELL_BITS.index(out)))
    return tuple(result)


EVENTS = {d: decoder_events(d) for d in PRIMITIVES}


@lru_cache(None)
def canonical(tree):
    if isinstance(tree, Raw):
        return 'r'
    return f'{tree.decoder}({canonical(tree.left)},{canonical(tree.right)})'


def tree_id(tree):
    return 'T_' + hashlib.sha256(canonical(tree).encode('ascii')).hexdigest()[:20]


def tree_to_dict(tree):
    if isinstance(tree, Raw):
        return {'kind': 'raw'}
    return {'kind': 'node', 'decoder': tree.decoder, 'left': tree_to_dict(tree.left),
            'right': tree_to_dict(tree.right)}


def tree_from_dict(value):
    if value['kind'] == 'raw':
        return RAW
    if value['kind'] != 'node':
        raise ValueError('Unknown tree kind')
    return Node(value['decoder'], tree_from_dict(value['left']), tree_from_dict(value['right']))


def pumping_tree(word):
    """Chronological decoder word; left survivor, right fresh raw pair."""
    out = RAW
    for decoder in word:
        out = Node(decoder, out, RAW)
    return out


def homogeneous_tree(word):
    """Chronological homogeneous levels, with two identical child subtrees."""
    out = RAW
    for decoder in word:
        out = Node(decoder, out, out)
    return out


@lru_cache(None)
def ancestry(tree):
    return 1 if isinstance(tree, Raw) else ancestry(tree.left) + ancestry(tree.right)


@lru_cache(None)
def depth(tree):
    return 0 if isinstance(tree, Raw) else 1 + max(depth(tree.left), depth(tree.right))


@lru_cache(None)
def retained_plan(tree):
    if isinstance(tree, Raw):
        return (1, 'raw')
    ml, mr = retained_memory(tree.left), retained_memory(tree.right)
    left_first = max(ml, mr + 1, 2)
    right_first = max(mr, ml + 1, 2)
    return (left_first, 'left_then_right') if left_first <= right_first else (right_first, 'right_then_left')


def retained_memory(tree):
    """Minimum in the specified noninterleaved depth-first child schedule."""
    return retained_plan(tree)[0]


@lru_cache(None)
def terminal_plan(tree, axis):
    if axis not in AXIS_PERMUTATIONS:
        raise ValueError(f'Unknown terminal axis: {axis}')
    if isinstance(tree, Raw):
        return (1, 'immediate_measurement')
    targets = TERMINAL_REDUCTIONS.get((tree.decoder, axis))
    if targets is None:
        return (retained_memory(tree), 'retained_decoder_then_measurement')
    classical_memory = max(terminal_memory(tree.left, targets[0]), terminal_memory(tree.right, targets[1]))
    if classical_memory <= retained_memory(tree):
        return (classical_memory, 'classical_terminal_comparison')
    return (retained_memory(tree), 'retained_decoder_then_measurement')


def terminal_memory(tree, axis):
    return terminal_plan(tree, axis)[0]


def check_permissions(axis, permissions):
    if permissions not in ('real', 'enlarged_axes'):
        raise ValueError('permissions must be real or enlarged_axes')
    if axis not in ('X', 'Z') and not (permissions == 'enlarged_axes' and axis == 'Y'):
        raise ValueError(f'{axis} is unavailable under {permissions}')


def structural_record(tree, axis='Z', permissions='real'):
    check_permissions(axis, permissions)
    return dict(family='explicit_binary_tree', tree_id=tree_id(tree), tree=tree_to_dict(tree), canonical_tree=canonical(tree),
                final_axis=axis, permissions=permissions,
                successful_ancestry_inputs=ancestry(tree), depth=depth(tree),
                retained_memory=retained_memory(tree),
                terminal_memory=terminal_memory(tree, axis),
                memory_upper_bound=terminal_memory(tree, axis),
                retained_loading_order=retained_plan(tree)[1],
                terminal_implementation=terminal_plan(tree, axis)[1],
                retry_rule=RETRY_RULE, raw_cost_model='expected_all_consumed_raw_pairs_including_retries',
                memory_model='peak_local_live_qubits_including_input_port;no_external_quantum_buffer',
                memory_is_achievable_upper_bound=True, global_minimality_proved=False,
                memory_proof_kind='exact_in_registered_depth_first_and_terminal_reduction_schedule_class',
                depth_first_retained_schedule_minimal=True,
                terminal_classical_storage='unlimited_and_not_counted_as_quantum_memory',
                terminal_raw_cost_unchanged=True,
                pooled_B_substitution=False)


def executable_schedule(tree, axis=None):
    """Recursive instruction record; retries do not expand into infinite traces.

    axis=None requests a retained quantum output. A supplied axis requests the
    final classical instrument, and may release qubits before another child is
    prepared. The two outputs are never silently identified.
    """
    if isinstance(tree, Raw):
        return dict(action='receive_raw_pair', peak=1, final_axis=axis,
                    output='quantum_pair' if axis is None else 'classical_outcomes')
    if axis is not None and terminal_plan(tree, axis)[1] == 'classical_terminal_comparison':
        targets = TERMINAL_REDUCTIONS[(tree.decoder, axis)]
        return dict(action='repeat_until_equal_relative_check', decoder=tree.decoder,
                    implementation='classical_terminal_comparison', peak=terminal_memory(tree, axis),
                    output='classical_outcomes', child_axes=list(targets),
                    children=[executable_schedule(tree.left, targets[0]), executable_schedule(tree.right, targets[1])],
                    key_rule='left_child_outcome;known_affine_frame_if_required',
                    accept_rule='xor_of_relative_child_outcomes_is_zero',
                    public_communication='each_party_announces_only_its_local_XOR_of_the_two_child_outcomes',
                    private_records='individual_child_key_outcomes_remain_private',
                    basis_convention='Bob_uses_conjugate_basis;apply_known_Y_outcome_sign_correction',
                    failure_rule='discard_both_classical_outputs_and_restart_this_node',
                    pooling='none;one_successful_child_of_each_type_per_parent_attempt')
    order = retained_plan(tree)[1]
    children = [tree.left, tree.right] if order == 'left_then_right' else [tree.right, tree.left]
    return dict(action='repeat_until_accepted_decoder', decoder=tree.decoder,
                implementation='retained_quantum_decoder', peak=retained_memory(tree),
                prepare_order=order, children=[executable_schedule(c) for c in children],
                storage_rule='hold_first_successful_child_while_retrying_second_child',
                wire_assignment={'left': 0, 'right': 1}, gates=PRIMITIVES[tree.decoder]['gates'],
                measure_check_wire=1, accept_rule='equal_check_outcomes',
                public_communication='announce_only_check_wire_outcome;retain_key_wire_outcome_privately',
                failure_rule='discard_both_quantum_outputs_and_restart_this_node',
                final_axis=axis, output='quantum_pair' if axis is None else 'classical_outcomes')


def step(left, right, decoder):
    left, right = np.broadcast_arrays(np.asarray(left, float), np.asarray(right, float))
    raw = np.zeros_like(left)
    for i, j, k in EVENTS[decoder]:
        raw[..., k] += left[..., i] * right[..., j]
    success = raw.sum(axis=-1)
    output = np.divide(raw, success[..., None], out=np.zeros_like(raw), where=success[..., None] > 0)
    return output, success


class Evaluator:
    """Per-source cache shared by a catalog; p has shape (...,4), Bell I,X,Y,Z."""
    def __init__(self, p):
        self.p = np.asarray(p, float)
        if self.p.shape[-1:] != (4,) or np.any(self.p < 0) or not np.all(np.isfinite(self.p)):
            raise ValueError('A finite nonnegative Bell law of shape (...,4) is required')
        if not np.allclose(self.p.sum(axis=-1), 1, rtol=0, atol=2e-13):
            raise ValueError('Bell probabilities must sum to one')
        ones = np.ones(self.p.shape[:-1])
        self.cache = {RAW: dict(state=self.p, success=ones, expected_raw_cost=ones, yield_per_raw_pair=ones)}

    def retained(self, tree):
        if tree in self.cache:
            return self.cache[tree]
        left, right = self.retained(tree.left), self.retained(tree.right)
        state, success = step(left['state'], right['state'], tree.decoder)
        numerator = left['expected_raw_cost'] + right['expected_raw_cost']
        cost = np.divide(numerator, success, out=np.full_like(success, np.inf), where=success > 0)
        out = dict(state=state, success=success, expected_raw_cost=cost,
                   yield_per_raw_pair=np.divide(1., cost, out=np.zeros_like(cost), where=np.isfinite(cost)))
        self.cache[tree] = out
        return out

    def terminal(self, tree, axis='Z'):
        out = self.retained(tree)
        return {**out, 'state': out['state'][..., list(AXIS_PERMUTATIONS[axis])],
                'retained_memory': retained_memory(tree), 'memory_upper_bound': terminal_memory(tree, axis),
                'successful_ancestry_inputs': ancestry(tree)}


def evaluate(tree, p, axis=None):
    evaluator = Evaluator(p)
    return evaluator.retained(tree) if axis is None else evaluator.terminal(tree, axis)


def evaluate_exact(tree, p, axis=None):
    """Exact Fraction replay, intended for witnesses/tests rather than scans."""
    p = tuple(Fraction(x) for x in p)
    if len(p) != 4 or min(p) < 0 or sum(p) != 1:
        raise ValueError('Exact normalized Bell probabilities are required')
    @lru_cache(None)
    def visit(node):
        if isinstance(node, Raw):
            return p, Fraction(1), Fraction(1)
        a, ca, _ = visit(node.left)
        b, cb, _ = visit(node.right)
        if ca is None or cb is None:
            return (Fraction(0),) * 4, None, Fraction(0)
        raw = [Fraction(0) for _ in range(4)]
        for i, j, k in EVENTS[node.decoder]:
            raw[k] += a[i] * b[j]
        success = sum(raw)
        if not success:
            return (Fraction(0),) * 4, None, Fraction(0)
        return tuple(x / success for x in raw), (ca + cb) / success, success
    state, cost, success = visit(tree)
    if axis is not None:
        state = tuple(state[i] for i in AXIS_PERMUTATIONS[axis])
    return dict(state=state, expected_raw_cost=cost, success=success,
                yield_per_raw_pair=Fraction(0) if cost is None else 1 / cost)
