"""Readable reference calculations for the paper's ED+AD resource model.

Bell probabilities always have order (I, X, Y, Z). Rates count secret bits
per consumed raw pair. These scalar routines explain and independently
check the formulas; the interval optimizers in numerics/ establish bounds.
"""
from fractions import Fraction
from itertools import product
from math import log2
import numpy as np

ERROR_LABELS = ((0, 0), (1, 0), (1, 1), (0, 1))


def entropy(probabilities):
    """Shannon entropy in bits, using 0 log(0) = 0."""
    return -sum(float(p) * log2(float(p)) for p in probabilities if p > 0)


def binary_entropy(p):
    return entropy((p, 1 - p))


def b_round(bell):
    """Return (conditional Bell law, accepted outputs per input bit pair).

    Two already measured bit pairs supply one output when local parities
    agree. Pool the successes before applying another B round.
    """
    i, x, y, z = bell
    unnormalized = np.array((i*i + z*z, x*x + y*y, 2*x*y, 2*i*z))
    success = float(unnormalized.sum())
    return unnormalized / success, success / 2


def watanabe_rate(bell):
    """Two-way reconciliation and PA coefficient per classical input pair."""
    i, x, y, z = map(float, bell)
    a, b = i + z, x + y
    correction = 0 if a*b == 0 else a*b*binary_entropy((i*x + z*y)/(a*b))
    after_b, b_yield = b_round(bell)
    return max(0.0, 1 - entropy(bell) + correction,
               b_yield * (1 - entropy(after_b)))


def classical_rate(bell, b_rounds=0, terminal=watanabe_rate):
    """Include every B-round selection factor exactly once."""
    law = np.array(bell, dtype=float)
    accepted_fraction = 1.0
    for _ in range(b_rounds):
        law, step_yield = b_round(law)
        accepted_fraction *= step_yield
    return accepted_fraction * terminal(law)


def z_ed_step(first, second):
    """ZZ ED: return the conditional output law and check acceptance."""
    p_i, p_x, p_y, p_z = first
    q_i, q_x, q_y, q_z = second
    output = np.array((p_i*q_i + p_z*q_z, p_x*q_x + p_y*q_y,
                       p_x*q_y + p_y*q_x, p_i*q_z + p_z*q_i))
    success = float(output.sum())
    return output / success, success


def combine_cost(first_cost, second_cost, acceptance):
    """Cost after independently preparing, retaining and combining two pairs."""
    return (first_cost + second_cost) / acceptance


def combine_memory(first_memory, second_memory):
    """Peak memory when the first successful output waits for the second."""
    return min(max(first_memory, 1 + second_memory),
               max(second_memory, 1 + first_memory))


def linear_code_rates(bell, generator, parity_check):
    """Compute clear-tag and charged-OTP coefficients by finite sums.

    generator has shape (n,k), parity_check has shape (n-k,n), and
    parity_check @ generator is zero modulo 2. This is the column-generator
    convention used in the manuscript (some optimized engines use G.T).
    Enumerating 4**n errors is appropriate for small teaching/check examples.
    """
    generator = np.asarray(generator, dtype=int)
    parity_check = np.asarray(parity_check, dtype=int)
    n, k = generator.shape
    assert parity_check.shape == (n-k, n)
    assert not np.any((parity_check @ generator) % 2)
    groups = {}
    for indices in product(range(4), repeat=n):
        probability = float(np.prod([bell[j] for j in indices]))
        if probability == 0:
            continue
        errors = np.array([ERROR_LABELS[j] for j in indices], dtype=int)
        bit_error, phase_error = errors[:, 0], errors[:, 1]
        syndrome = tuple((parity_check @ bit_error) % 2)
        logical_phase = tuple((generator.T @ phase_error) % 2)
        group = groups.setdefault(syndrome, {"mass": 0.0, "logical": {}, "full": []})
        group["mass"] += probability
        key = (tuple(bit_error), logical_phase)
        group["logical"][key] = group["logical"].get(key, 0.0) + probability
        group["full"].append(probability)
    clear = gross = 0.0
    for group in groups.values():
        mass = group["mass"]
        h_logical = entropy(p / mass for p in group["logical"].values())
        h_full = entropy(p / mass for p in group["full"])
        clear += mass * max(k - h_logical, 0) / n
        gross += mass * max(n - h_full, 0) / n
    syndrome_entropy = entropy(group["mass"] for group in groups.values())
    return {"clear": clear, "plain_otp": gross - (n-k)/n,
            "compressed_otp": gross - syndrome_entropy/n}


def graph_pauli_support(adjacency, coefficient):
    """Support of a product of graph stabilizers (bit-mask convention)."""
    z_mask = 0
    for vertex, neighbors in enumerate(adjacency):
        if coefficient >> vertex & 1:
            z_mask ^= neighbors
    return coefficient | z_mask


def graph_schedule(adjacency, logical_dual, order, q):
    """Exact memory, continuation probabilities and cost for Werner input.

    adjacency[i] is the bit mask of neighbors of vertex i; order lists the
    raw-input labels. A stabilizer combination is public when its binary
    inner product with logical_dual is zero. Zero relative syndrome is kept.
    q may be a Fraction, giving exact rational costs.
    """
    n = len(adjacency)
    assert sorted(order) == list(range(n)) and 0 < logical_dual < 2**n
    character = 1 - 2*Fraction(q)
    supports = [graph_pauli_support(adjacency, a) for a in range(2**n)]
    loaded = 0
    continuation, between_arrivals = [], []
    for vertex in list(order) + [None]:
        available = [a for a, support in enumerate(supports)
                     if support & ~loaded == 0]
        public = [a for a in available if (a & logical_dual).bit_count() % 2 == 0]
        survival = sum((character**supports[a].bit_count() for a in public),
                       Fraction(0)) / len(public)
        continuation.append(survival)
        if vertex is not None:
            dimension = len(available).bit_length() - 1
            assert 1 << dimension == len(available)
            between_arrivals.append(loaded.bit_count() - dimension)
            loaded |= 1 << vertex
    attempt_cost = sum(continuation[:-1], Fraction(0))
    acceptance = continuation[-1]
    return {"memory": 1 + max(between_arrivals),
            "continuation": continuation[:-1], "attempt_cost": attempt_cost,
            "acceptance": acceptance, "cost": attempt_cost / acceptance}
