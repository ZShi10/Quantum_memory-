"""Positive unnormalized recurrence with floating, rational, IV, or Jet scalars.

For a subtree T, u_T is its all-success unnormalized Bell vector, P_T=sum u_T,
and D_T=C_T P_T. At a node, u=conv(u_L,u_R), D=D_L P_R+D_R P_L.
There is no intermediate conditional normalization. Normalize only once at
the terminal, with p=u/P, y=P/D. This avoids unnecessary interval dependency.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
from .model import RAW, Raw, EVENTS, AXIS_PERMUTATIONS, tree_from_dict


def unnormalized(tree, components, one=1, zero=0):
    """Backend-generic exact algebra; components has four I,X,Y,Z scalars.

    A scalar can be an array, Fraction, interval, or first-derivative Jet.
    It must support + and *. Return (four raw weights, P, D).
    """
    initial = tuple(components)
    if len(initial) != 4:
        raise ValueError('Four Bell components are required')
    cache = {}
    def visit(node):
        if node in cache:
            return cache[node]
        if isinstance(node, Raw):
            result = initial, one, one
        else:
            a, pa, da = visit(node.left)
            b, pb, db = visit(node.right)
            raw = [zero for _ in range(4)]
            for i, j, k in EVENTS[node.decoder]:
                raw[k] = raw[k] + a[i] * b[j]
            mass = sum(raw, zero)
            result = tuple(raw), mass, da * pb + db * pa
        cache[node] = result
        return result
    return visit(tree)


def _legacy_modules():
    old = Path(__file__).resolve().parents[2] / 'revision_20260922'
    if str(old) not in sys.path:
        sys.path.insert(0, str(old))
    import interval_jets as jets
    import stable_intervals as iv
    return jets, iv


def prepare_jets(b, p, intervals, record, rounds=0):
    """Drop-in counterpart of interval_jets.prepare for explicit_binary_tree.

    Returns (state Jet, yield Jet). Additional rounds are pooled classical B
    rounds and hence multiply yield by s/2; the ED tree itself retains its
    specified retry semantics and is never substituted by a pooled schedule.
    """
    jets, _ = _legacy_modules()
    tree = tree_from_dict(record['tree'])
    initial = jets.source(b, p, intervals)
    one = jets.Jet((np.ones(len(intervals)), np.ones(len(intervals))))
    zero = jets.Jet((np.zeros(len(intervals)), np.zeros(len(intervals))))
    raw, mass, denominator = unnormalized(tree, [initial[..., i] for i in range(4)], one, zero)
    state = (jets.stack(raw) / mass[..., None]).clip()
    y = (mass / denominator).clip()
    state = state[..., list(AXIS_PERMUTATIONS[record['final_axis']])]
    for _ in range(rounds):
        state, success = jets.step(state, state, 'Z')
        y = (y * success / 2).clip()
    return state, y


class PositiveInterval:
    """Internal positive-only wrapper around the existing audited IV backend."""
    def __init__(self, bounds, backend):
        self.bounds = tuple(np.asarray(x, float) for x in bounds)
        self.iv = backend

    def __add__(self, other):
        if not isinstance(other, PositiveInterval):
            other = PositiveInterval((np.asarray(other), np.asarray(other)), self.iv)
        lo, hi = self.iv.add(self.bounds, other.bounds)
        return PositiveInterval((np.maximum(0, lo), hi), self.iv)
    __radd__ = __add__

    def __mul__(self, other):
        if not isinstance(other, PositiveInterval):
            if other < 0:
                raise ValueError('This wrapper is restricted to nonnegative scalars')
            other = PositiveInterval((np.asarray(other), np.asarray(other)), self.iv)
        lo, hi = self.iv.mul(self.bounds, other.bounds)
        return PositiveInterval((np.maximum(0, lo), hi), self.iv)
    __rmul__ = __mul__


def _interval_ratio(numerator, divisor, iv):
    good = divisor[0] > 0
    safe = (np.where(good, divisor[0], 1.), np.where(good, divisor[1], 1.))
    lo, hi = iv.div(numerator, safe)
    return np.where(good, np.maximum(0, lo), 0.), np.where(good, np.minimum(1, hi), 1.)


class IntervalEvaluator:
    """Shared positive-polynomial DAG evaluation for an entire candidate menu."""
    def __init__(self, initial):
        _, self.iv = _legacy_modules()
        self.initial = tuple(np.asarray(x, float) for x in initial)
        if self.initial[0].shape[-1:] != (4,) or self.initial[0].shape != self.initial[1].shape:
            raise ValueError('Expected equally shaped Bell bounds (...,4)')
        shape = self.initial[0].shape[:-1]
        self.one = PositiveInterval((np.ones(shape), np.ones(shape)), self.iv)
        self.zero = PositiveInterval((np.zeros(shape), np.zeros(shape)), self.iv)
        components = tuple(PositiveInterval((self.initial[0][..., i], self.initial[1][..., i]), self.iv) for i in range(4))
        self.raw_cache = {RAW: (components, self.one, self.one)}
        self.output_cache = {}

    def raw(self, tree):
        if tree in self.raw_cache:
            return self.raw_cache[tree]
        a, pa, da = self.raw(tree.left)
        b, pb, db = self.raw(tree.right)
        raw = [self.zero for _ in range(4)]
        for i, j, k in EVENTS[tree.decoder]:
            raw[k] = raw[k] + a[i]*b[j]
        raw = tuple(PositiveInterval((v.bounds[0], np.minimum(1., v.bounds[1])), self.iv) for v in raw)
        mass = sum(raw, self.zero)
        mass = PositiveInterval((mass.bounds[0], np.minimum(1., mass.bounds[1])), self.iv)
        result = raw, mass, da*pb+db*pa
        self.raw_cache[tree] = result
        return result

    def retained(self, tree):
        if tree in self.output_cache:
            return self.output_cache[tree]
        raw, mass, denominator = self.raw(tree)
        weights = tuple(np.stack([v.bounds[j] for v in raw], axis=-1) for j in (0, 1))
        state = _interval_ratio(weights, tuple(v[..., None] for v in mass.bounds), self.iv)
        y = _interval_ratio(mass.bounds, denominator.bounds, self.iv)
        self.output_cache[tree] = state, y
        return state, y

    def terminal(self, tree, axis):
        state, y = self.retained(tree)
        return tuple(v[..., list(AXIS_PERMUTATIONS[axis])] for v in state), y


def prepare_intervals(initial, record, rounds=0):
    """Input (lower,upper) arrays (...,4); output state/yield bound tuples.

    Uses value-only outward arithmetic, without derivative work. Wider cells
    with no positive denominator lower bound conservatively return [0,1].
    """
    tree = tree_from_dict(record['tree'])
    evaluator = IntervalEvaluator(initial)
    iv = evaluator.iv
    state, y = evaluator.terminal(tree, record['final_axis'])
    if rounds:
        from recurrence_interval_library import step
        from fractions import Fraction
        for _ in range(rounds):
            state, success = step(state, state, 'Z')
            y = iv.scale(iv.mul(y, success), Fraction(1, 2))
    return state, y


def generate_bounds(initial, permissions='real', max_leaves=5, max_pump_checks=7):
    """All candidate bounds, same order as catalog.generate, cached by subtree.

    Returns ((prob_lo,prob_hi),(yield_lo,yield_hi)); shapes match generate.
    Metadata and memory are supplied once by catalog.generate/candidate_records.
    """
    from .catalog import structural_catalog
    if permissions not in ('real', 'enlarged_axes', 'full'):
        raise ValueError('Unknown permission class')
    axes = ('Z', 'X') if permissions == 'real' else ('Z', 'X', 'Y')
    evaluator = IntervalEvaluator(initial)
    ps, ys = [], []
    for tree, _ in structural_catalog(max_leaves, max_pump_checks):
        for axis in axes:
            p, y = evaluator.terminal(tree, axis)
            ps.append(p)
            ys.append(y)
    return (tuple(np.stack([v[j] for v in ps]) for j in (0, 1)),
            tuple(np.stack([v[j] for v in ys]) for j in (0, 1)))
