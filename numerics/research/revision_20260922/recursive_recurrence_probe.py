"""Bounded real-Clifford recurrence probe and exact selected witnesses.

This is a new achievable family, not a replacement for the frozen finite-class
certificates.  Each node obtains two independent accepted children, retains the
first while retrying the second, and restarts the node if its own check fails.
All storage and gates are ideal.  Rates are bits per consumed raw pair.
"""
from pathlib import Path
from fractions import Fraction as F
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'joint_optimization'))
from streaming import rate
from interval_certificate import IV, LN2

H_PERM = [0, 3, 2, 1]
LABELS = [(0, 0), (1, 0), (1, 1), (0, 1)]


def map_float(p, basis):
    if basis == 'X':
        p = p[H_PERM]
    a, x, y, z = p
    raw = np.array([a*a+z*z, x*x+y*y, 2*x*y, 2*a*z])
    s = raw.sum()
    return raw/s, s


def exact_polynomial_map(p, basis):
    if basis == 'X':
        p = [p[i] for i in H_PERM]
    a, x, y, z = p
    raw = [a*a+z*z, x*x+y*y, 2*x*y, 2*a*z]
    s = sum(raw)
    return [v/s for v in raw], s


def exact_pair_enumeration(p, basis):
    """Independent enumeration of the 16 two-pair Pauli error events."""
    raw = [F(0)]*4
    for i, (x1, z1) in enumerate(LABELS):
        for j, (x2, z2) in enumerate(LABELS):
            if basis == 'X':
                cx, cz, tx, tz = z1, x1, z2, x2
            else:
                cx, cz, tx, tz = x1, z1, x2, z2
            # Bilateral CNOT: (cx,cz;tx,tz) -> (cx,cz^tz;cx^tx,tz).
            if cx == tx:
                raw[LABELS.index((cx, cz ^ tz))] += p[i]*p[j]
    s = sum(raw)
    return [v/s for v in raw], s


def successful_schedule(depth):
    live = peak = raw = 0

    def prepare(level):
        nonlocal live, peak, raw
        if level == 0:
            live += 1
            raw += 1
            peak = max(peak, live)
            return
        prepare(level-1)
        prepare(level-1)
        live -= 1  # measure/discard the target; retain the control

    prepare(depth)
    assert live == 1 and raw == 2**depth and peak == depth+1
    # Retrying a failed subtree starts at the same stack occupancy; a failure
    # only frees registers.  Thus this also bounds every failure trajectory.
    return dict(successful_ancestry_raw_pairs=raw, peak_live_qubits_per_party=peak,
                schedule='depth-first; retain accepted first child while retrying second',
                failure_policy='discard both children if parent check fails; retry that parent')


def probe(q, maximum_depth=7):
    states = [('', np.array([1-1.5*q, q/2, q/2, q/2]), 1.)]
    rows = []
    lengths = np.arange(1, 129)[None, :]
    for depth in range(1, maximum_depth+1):
        next_states = []
        for path, p, y in states:
            for basis in 'ZX':
                out, s = map_float(p, basis)
                next_states.append((path+basis, out, y*s/2))
        states = next_states
        best = None
        for final_basis in 'ZX':
            p = np.stack([v[1] if final_basis == 'Z' else v[1][H_PERM] for v in states])
            y = np.array([v[2] for v in states])
            eps = p[:, 0]+p[:, 3]-p[:, 1]-p[:, 2]
            phi0 = p[:, 3]/(p[:, 0]+p[:, 3])
            phi1 = p[:, 2]/(p[:, 1]+p[:, 2])
            shape = (len(states), len(lengths[0]))
            eps = np.broadcast_to(eps[:, None], shape)
            phi0 = np.broadcast_to(phi0[:, None], shape)
            phi1 = np.broadcast_to(phi1[:, None], shape)
            for k in range(17):
                values = y[:, None]*rate(eps, phi0, phi1, k, lengths)
                i, j = np.unravel_index(np.argmax(values), values.shape)
                record = dict(Q=q, depth=depth, memory_schedule_upper_bound=depth+1,
                              path=states[i][0], final_basis=final_basis,
                              classical_B_rounds=k, repetition_length=int(lengths[0, j]),
                              rate_float=float(values[i, j]),
                              ED_expected_raw_cost=1/states[i][2])
                if best is None or record['rate_float'] > best['rate_float']:
                    best = record
        rows.append(best)
    # These rows are EXACT-depth probes. The budget envelope must include all
    # shallower depths and other families; very small positive floats are not
    # evidence of positivity and must not be plotted as certified witnesses.
    return rows


def entropy_interval(p):
    return -sum((IV(v)*IV(v).ln()/LN2 for v in p if v), IV(0))


def selected_certificate(q, path, final_basis, classical_B_rounds, repetition_length):
    assert repetition_length == 2
    p = [1-3*q/2, q/2, q/2, q/2]
    cost = F(1)
    trace = []
    for level, basis in enumerate(path, 1):
        new, success = exact_pair_enumeration(p, basis)
        assert (new, success) == exact_polynomial_map(p, basis)
        # Each attempt consumes two independently prepared children. A failed
        # second-child attempt does not destroy the first accepted child.
        cost = 2*cost/success
        p = new
        trace.append(dict(level=level, basis=basis, success=float(success),
                          output_p=list(map(float, p)), expected_raw_cost=float(cost)))
    if final_basis == 'X':
        p = [p[i] for i in H_PERM]
    y = 1/cost
    # Once final key-basis measurement occurs, these B steps operate on raw
    # classical strings. Accepted bits are pooled before each subsequent step.
    # One last B step equals zero-syndrome repetition [2,1,2] here.
    for _ in range(classical_B_rounds+1):
        new, success = exact_pair_enumeration(p, 'Z')
        assert (new, success) == exact_polynomial_map(p, 'Z')
        p = new
        y *= success/2
    gain = IV(1)-entropy_interval(p)
    result = IV(y)*gain
    assert result.lo > 0
    return dict(Q=str(q), path=path, final_basis=final_basis,
                classical_B_rounds=classical_B_rounds,
                repetition_length=repetition_length,
                memory_is_schedule_upper_bound_not_minimum=True,
                **successful_schedule(len(path)),
                ED_expected_raw_cost=float(cost), quantum_trace=trace,
                final_virtual_Bell_probabilities=list(map(float, p)),
                secret_fraction_interval=[str(gain.lo), str(gain.hi)],
                rate_interval_bits_per_raw_pair=[str(result.lo), str(result.hi)],
                validation='Exact rational Pauli enumeration equals polynomial recurrence at every step; entropy enclosed by 60-digit directed rounding.',
                scope='Constructive achievable rate only; no claim of a global optimum or a new distillation primitive.')


def main():
    output = ROOT / 'data'
    output.mkdir(exist_ok=True)
    probes = [row for q in (.28, .30, .32) for row in probe(q)]
    witnesses = [selected_certificate(F(3, 10), 'ZXZXZXX', 'X', 0, 2),
                 selected_certificate(F(8, 25), 'ZXZXZXZ', 'X', 2, 2)]
    for witness in witnesses:
        q = float(F(witness['Q']))
        row = next(r for r in probes if r['Q'] == q and r['depth'] == 7)
        assert row['path'] == witness['path']
        exact = float(witness['rate_interval_bits_per_raw_pair'][0])
        assert abs(row['rate_float']-exact) <= 5e-12*exact
    result = dict(status='selected_witnesses_certified',
                  warnings=['Floating probes are not certificates, especially for near-zero rates.',
                            'Per-depth rows are not memory-budget envelopes.',
                            'Peak memory is an explicit upper bound, not a proved minimum.',
                            'Recurrence with quantum retry is outside the old bounded fixed-input-block catalog.',
                            'Gates and arbitrarily long waiting memory are ideal.'],
                  finite_probe_scope=dict(quantum_depth_max=7, bases=['Z','X'],
                                          classical_B_max=16, repetition_max=128),
                  selected_witnesses=witnesses, floating_probes=probes)
    (output/'recursive_recurrence_probe.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(status=result['status'], selected_witnesses=witnesses)), flush=True)


if __name__ == '__main__':
    main()
