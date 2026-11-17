"""Exact three-register counterexample to extrapolating the m<=4 result.

Run from the project root. This concerns the specified complete terminal
measurement, not an assertion that it maximizes a QKD rate.
"""
from pathlib import Path
import sys, json
from itertools import combinations, permutations
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
sys.path[:0] = [str(PROJECT / 'research/completion'),
               str(PROJECT / 'research/joint_optimization')]
from algebra import rref
from streaming_clifford import compile_measurement, cut_entropies
from compile_graph_library import exact_replay
from verify_streaming import dense_check


def run():
    m = 5
    rows = [(1 << j) | (1 << (m + (j-1) % m)) |
            (1 << (m + (j+1) % m)) for j in range(m)]
    cuts = cut_entropies(rows, m)
    pair_cuts = {','.join(map(str, pair)): cuts[sum(1 << j for j in pair)]
                 for pair in combinations(range(m), 2)}
    assert set(pair_cuts.values()) == {2}
    assert max(cuts) == 2
    all_peaks = [max(1 + cuts[sum(1 << j for j in order[:i])]
                     for i in range(m)) for order in permutations(range(m))]
    assert set(all_peaks) == {3}
    compiler = compile_measurement(rref(rows, 2*m), m, real=True)
    exact_replay(compiler, m)
    discrepancy = dense_check(compiler, m)
    assert compiler['M'] == compiler['M_min'] == 3
    assert compiler['minimum_proved']
    result = dict(
        status='validated',
        observable_rule='g_j = X_j Z_(j-1) Z_(j+1), indices modulo five',
        symplectic_generators=rows, pair_cut_ranks=pair_cuts,
        all_120_arrival_orders_have_peak=3, dense_operator_error=discrepancy,
        terminal_compilation=compiler,
        proof=('For every two-vertex subset A of C5, the two rows of the '
               'crossing adjacency matrix are nonzero and distinct, so its '
               'binary rank is two. Every arrival order has such a two-vertex '
               'prefix. Every cut rank is at most min(|A|,5-|A|)<=2. Thus '
               'the minimum, over arrival orders, of the maximum cut rank is '
               'exactly two. The exact complete-basis streaming measurement '
               'formula M=1+min_order max_prefix rank(Gamma_A,A^c) gives M=3, '
               'including the arriving input register. The saved real '
               'H/CNOT compiler attains this lower bound.'),
        scope=('Exact terminal measurement with arbitrary classical outcome '
               'reconstruction in the declared streaming model. This does '
               'not establish QKD-rate optimality or rule out a different '
               'protocol with the same scalar rate and fewer registers.'))
    target = HERE / 'data/five_cycle_example.json'
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(M=3, checked_orders=120,
                         dense_operator_error=discrepancy,
                         output=str(target))))


if __name__ == '__main__':
    run()
