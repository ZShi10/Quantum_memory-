"""Independent gate, renewal, scheduling, exact arithmetic, and interval checks."""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from resource_model.model import *
from resource_model.catalog import *
from resource_model.bounds import *
from resource_model.bounds import _legacy_modules
from itertools import product
from fractions import Fraction as F
import json
import time


def dense_event_tables():
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.diag([1., -1.]).astype(complex)
    H = (X + Z) / np.sqrt(2)
    paul = [I, X, Y, Z]
    pair = {(i, j): np.kron(a, b) for i, a in enumerate(paul) for j, b in enumerate(paul)}
    result = {}
    for name, item in PRIMITIVES.items():
        unitary = np.eye(4, dtype=complex)
        for gate in item['gates']:
            if gate[0] == 'h':
                operator = np.kron(H, I) if gate[1] == 0 else np.kron(I, H)
            else:
                operator = np.zeros((4, 4), complex)
                for column in range(4):
                    bits = [column >> 1, column & 1]
                    bits[gate[2]] ^= bits[gate[1]]
                    operator[2*bits[0]+bits[1], column] = 1
            unitary = operator @ unitary
        events = {}
        for ij, error in pair.items():
            output = unitary @ error @ unitary.conj().T
            coefficients = {kl: np.trace(op.conj().T @ output)/4 for kl, op in pair.items()}
            kl = max(coefficients, key=lambda x: abs(coefficients[x]))
            assert abs(abs(coefficients[kl])-1) < 2e-15
            if BELL_BITS[kl[1]][0] == 0:
                events[ij] = kl[0]
        assert events == {(i, j): k for i, j, k in EVENTS[name]}, name
        result[name] = events
    return result


def terminal_identity_checks():
    count = 0
    for (decoder, axis), targets in TERMINAL_REDUCTIONS.items():
        accepted = {(i, j): k for i, j, k in EVENTS[decoder]}
        for i, a in enumerate(BELL_BITS):
            for j, b in enumerate(BELL_BITS):
                left, right = parity((a,), targets[0]), parity((b,), targets[1])
                assert ((i, j) in accepted) == (left == right)
                if left == right:
                    observed = parity((BELL_BITS[accepted[i, j]],), axis)
                    assert observed == left
                count += 1
    return count


def simulated_success_trace_peak(tree, axis=None, offset=0):
    """Construct a concrete no-failure execution trace with live-wire counts.

    Failed child/parent attempts revisit the same states and cannot exceed
    this trace's maximum. Offsets count previously stored sibling qubits.
    """
    if isinstance(tree, Raw):
        return offset + 1
    if axis is not None and terminal_plan(tree, axis)[1] == 'classical_terminal_comparison':
        a, b = TERMINAL_REDUCTIONS[tree.decoder, axis]
        return max(simulated_success_trace_peak(tree.left, a, offset),
                   simulated_success_trace_peak(tree.right, b, offset))
    first, second = (tree.left, tree.right) if retained_plan(tree)[1] == 'left_then_right' else (tree.right, tree.left)
    return max(simulated_success_trace_peak(first, None, offset),
               simulated_success_trace_peak(second, None, offset+1), offset+2)


@lru_cache(None)
def exhaustive_depth_first_peaks(tree):
    if isinstance(tree, Raw):
        return (1,)
    result = set()
    for a in exhaustive_depth_first_peaks(tree.left):
        for b in exhaustive_depth_first_peaks(tree.right):
            result.add(max(a, b+1, 2))
            result.add(max(b, a+1, 2))
    return tuple(sorted(result))


@dataclass(frozen=True)
class Dual:
    value: F
    derivative: F = F(0)
    def __add__(self, other):
        other = other if isinstance(other, Dual) else Dual(F(other))
        return Dual(self.value+other.value, self.derivative+other.derivative)
    __radd__ = __add__
    def __mul__(self, other):
        other = other if isinstance(other, Dual) else Dual(F(other))
        return Dual(self.value*other.value, self.derivative*other.value+self.value*other.derivative)
    __rmul__ = __mul__
    def __truediv__(self, other):
        other = other if isinstance(other, Dual) else Dual(F(other))
        return Dual(self.value/other.value, (self.derivative*other.value-self.value*other.derivative)/other.value**2)


def exact_jet_values(tree, b, p, t, axis):
    scale = min(b, p)
    raw_source = [Dual(v, scale*s) for v, s in zip((1-b-p+scale*t, b-scale*t, scale*t, p-scale*t), (1, -1, 1, -1))]
    raw, mass, den = unnormalized(tree, raw_source, Dual(F(1)), Dual(F(0)))
    return [raw[i]/mass for i in AXIS_PERMUTATIONS[axis]] + [mass/den]


def contains(bounds, value):
    return F(float(bounds[0])) <= value <= F(float(bounds[1]))


def main():
    started = time.time()
    tables = dense_event_tables()
    terminal_checks = terminal_identity_checks()
    layers = ordered_tree_layers(5)
    structural_checks = 0
    for layer in layers.values():
        for tree in layer:
            assert retained_memory(tree) == min(exhaustive_depth_first_peaks(tree))
            assert retained_memory(tree) == simulated_success_trace_peak(tree)
            assert tree_from_dict(tree_to_dict(tree)) == tree
            for axis in AXIS_PERMUTATIONS:
                assert terminal_memory(tree, axis) == simulated_success_trace_peak(tree, axis)
                structural_checks += 1
    pumps = list(variable_pumps(7))
    assert len({tree for _, tree in pumps}) == sum(4**d for d in range(8))
    assert all(retained_memory(tree) <= 2 for _, tree in pumps)
    actual_unique = len(set(tree for layer in layers.values() for tree in layer) | {t for _, t in pumps})
    assert actual_unique == coverage()['unique_structural_trees']

    old = ROOT.parents[1] / 'revision_20260922'
    sys.path.insert(0, str(old))
    import recurrence_v2 as legacy
    rng = np.random.default_rng(812964)
    laws = rng.dirichlet(np.ones(4), 9)
    comparisons = 0
    max_state_error = max_yield_error = 0.
    for n in range(4):
        for letters in product(PRIMITIVES, repeat=n):
            word = ''.join(letters)
            tree = homogeneous_tree(word)
            for axis in AXIS_PERMUTATIONS:
                record = dict(family='real_decoder_recurrence', path=word, final_axis=axis)
                oldp, oldy = legacy.prepare(laws, record)
                new = evaluate(tree, laws, axis)
                assert new['memory_upper_bound'] == legacy.memory(word, axis)
                max_state_error = max(max_state_error, float(np.max(np.abs(oldp-new['state']))))
                max_yield_error = max(max_yield_error, float(np.max(np.abs(oldy-new['yield_per_raw_pair']))))
                comparisons += len(laws)
    for d in PRIMITIVES:
        for r in range(1, 21):
            tree = pumping_tree(d*(r-1))
            for axis in AXIS_PERMUTATIONS:
                record = dict(family='real_decoder_pumping', primitive=d, r=r, final_axis=axis)
                oldp, oldy = legacy.prepare(laws, record)
                new = evaluate(tree, laws, axis)
                max_state_error = max(max_state_error, float(np.max(np.abs(oldp-new['state']))))
                max_yield_error = max(max_yield_error, float(np.max(np.abs(oldy-new['yield_per_raw_pair']))))
                comparisons += len(laws)
    assert max_state_error < 5e-13 and max_yield_error < 5e-13

    rational_p = tuple(map(F, ('61/100', '7/100', '11/100', '21/100')))
    selected = [t for _, t in variable_pumps(4)] + layers[5][::37]
    exact_checks = 0
    for tree in selected:
        exact = evaluate_exact(tree, rational_p)
        raw, mass, denominator = unnormalized(tree, rational_p, F(1), F(0))
        assert exact['state'] == tuple(v/mass for v in raw)
        assert exact['expected_raw_cost'] == denominator/mass
        assert exact['expected_raw_cost'] >= ancestry(tree)
        floating = evaluate(tree, np.array(rational_p, float))
        assert np.max(np.abs(floating['state']-np.array(exact['state'], float))) < 5e-14
        exact_checks += 1
    # Closed all-or-nothing raw-pump renewal formula, independent of C recurrence.
    pump_renewal_checks = 0
    for word, tree in variable_pumps(4):
        raw = rational_p
        prefix = F(1)
        mass = F(1)
        for decoder in word:
            prefix += mass
            nxt = [F(0)]*4
            for i, j, k in EVENTS[decoder]:
                nxt[k] += raw[i]*rational_p[j]
            raw = tuple(nxt)
            mass = sum(raw)
        # empty word consumes one input; after j checks D=1+sum_{i=0}^{j-1}P_i.
        numerator = F(1) if not word else prefix
        assert evaluate_exact(tree, rational_p)['expected_raw_cost'] == numerator/mass
        pump_renewal_checks += 1

    # Diverse asymmetric children, including deeper right child and zero-memory
    # terminal reductions, are evaluated with exact values AND derivatives.
    interval_trees = [RAW, pumping_tree('ZRXCZR'), homogeneous_tree('RXZ'),
                      Node('C', pumping_tree('ZR'), homogeneous_tree('RX')),
                      Node('R', homogeneous_tree('XZ'), pumping_tree('C'))] + layers[5][::503]
    intervals = [(F(0), F(1, 32)), (F(37, 100), F(19, 50)), (F(31, 32), F(1)), (F(1, 5), F(1, 5))]
    interval_checks = 0
    for b, p in ((F(1, 5), F(1, 5)), (F(3, 25), F(6, 25))):
        for tree in interval_trees:
            for axis in AXIS_PERMUTATIONS:
                record = structural_record(tree, axis, 'enlarged_axes')
                jets, _ = _legacy_modules()
                source = jets.source(float(b), float(p), intervals)
                state, y = prepare_jets(float(b), float(p), intervals, record)
                ivstate, ivy = prepare_intervals(source.v, record)
                for cell, (lo, hi) in enumerate(intervals):
                    for t in (lo, (lo+hi)/2, hi):
                        values = exact_jet_values(tree, b, p, t, axis)
                        for k, exact in enumerate(values):
                            val = tuple(v[cell, k] for v in state.v) if k < 4 else tuple(v[cell] for v in y.v)
                            der = tuple(v[cell, k] for v in state.d) if k < 4 else tuple(v[cell] for v in y.d)
                            val_iv = tuple(v[cell, k] for v in ivstate) if k < 4 else tuple(v[cell] for v in ivy)
                            assert contains(val, exact.value), (tree_id(tree), axis, cell, k, 'Jet value')
                            assert contains(der, exact.derivative), (tree_id(tree), axis, cell, k, 'Jet derivative')
                            assert contains(val_iv, exact.value), (tree_id(tree), axis, cell, k, 'IV value')
                            interval_checks += 3

    # Independent event-driven stochastic execution, never using the cost formula.
    simulations = []
    simlaw = np.array([.76, .06, .08, .10])
    for tree in (Node('C', RAW, homogeneous_tree('RX')), Node('R', pumping_tree('CZ'), Node('X', RAW, RAW))):
        def produce(node):
            if isinstance(node, Raw):
                return int(rng.choice(4, p=simlaw)), 1
            cost = 0
            while True:
                left, cl = produce(node.left)
                right, cr = produce(node.right)
                cost += cl + cr
                outcome = tables[node.decoder].get((left, right))
                if outcome is not None:
                    return outcome, cost
        runs = 20000
        draws = [produce(tree) for _ in range(runs)]
        costs = np.array([v[1] for v in draws], float)
        exact = evaluate(tree, simlaw)
        standard_error = float(costs.std(ddof=1)/np.sqrt(runs))
        assert abs(costs.mean()-exact['expected_raw_cost']) <= 6*standard_error
        frequencies = np.bincount([v[0] for v in draws], minlength=4)/runs
        sigma = np.sqrt(exact['state']*(1-exact['state'])/runs)
        assert np.all(np.abs(frequencies-exact['state']) < 6*sigma+1/runs)
        simulations.append(dict(tree_id=tree_id(tree), runs=runs, mean_raw_cost=float(costs.mean()),
                                predicted_raw_cost=float(exact['expected_raw_cost']), standard_error=standard_error,
                                observed_state=frequencies.tolist(), predicted_state=exact['state'].tolist()))

    # Complete default catalog batch order and value-bounds replay, including
    # the longest variable pumps; exact individual checks above certify the
    # point arithmetic independently rather than trusting floating agreement.
    batch_start = time.time()
    batch_laws = np.array([[.7, .1, .1, .1], [.6, .2, 0., .2], [.72, .04, .12, .12]])
    _, iv = _legacy_modules()
    lowers = np.array([[iv.rational(F(str(x)))[0] for x in row] for row in batch_laws])
    uppers = np.array([[iv.rational(F(str(x)))[1] for x in row] for row in batch_laws])
    fp, fy, fm, fr = generate(batch_laws, permissions='full')
    bp, by = generate_bounds((lowers, uppers), permissions='full')
    assert fp.shape[0] == coverage()['enlarged_axis_terminal_candidates']
    assert fp.shape == bp[0].shape and fy.shape == by[0].shape
    assert np.all(fp >= bp[0]-2e-13) and np.all(fp <= bp[1]+2e-13)
    assert np.all(fy >= by[0]-2e-13) and np.all(fy <= by[1]+2e-13)
    assert all(int(m) == r['terminal_memory'] for m, r in zip(fm, fr))
    batch_check = dict(records=len(fr), sources=len(batch_laws),
                       probability_and_yield_components=int(fp.size+fy.size),
                       seconds=time.time()-batch_start,
                       max_probability_bound_width=float(np.max(bp[1]-bp[0])),
                       max_yield_bound_width=float(np.max(by[1]-by[0])))

    example = Node('C', pumping_tree('ZR'), homogeneous_tree('RX'))
    sample = structural_record(example, 'Z')
    evaluated = evaluate(example, np.array([.7, .1, .1, .1]), 'Z')
    out = dict(status='passed', coverage=coverage(), dense_gate_error_events=64,
               terminal_classicalization_event_checks=terminal_checks,
               all_small_tree_schedule_checks=structural_checks,
               legacy_homogeneous_and_constant_pump_comparisons=comparisons,
               max_legacy_state_error=max_state_error, max_legacy_yield_error=max_yield_error,
               exact_fraction_tree_replays=exact_checks, variable_pump_renewal_checks=pump_renewal_checks,
               outward_value_and_exact_derivative_containments=interval_checks,
               independent_event_simulations=simulations,
               complete_default_batch_replay=batch_check,
               example=dict(record=sample, retained_schedule=executable_schedule(example),
                            terminal_schedule=executable_schedule(example, 'Z'),
                            Q=.2, state=evaluated['state'].tolist(),
                            expected_raw_cost=float(evaluated['expected_raw_cost']),
                            yield_per_raw_pair=float(evaluated['yield_per_raw_pair'])),
               seconds=time.time()-started,
               guarantee='Exact structure/renewal algebra plus implementation checks; no new QKD optimal-rate certificate is asserted by this file.')
    (ROOT/'validation.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ('example', 'independent_event_simulations')}, indent=2))


if __name__ == '__main__':
    main()
