"""Small independent checks of the manuscript's central formulas."""
from fractions import Fraction
from itertools import permutations
import json
import numpy as np
from qkd_core import (graph_schedule, linear_code_rates, entropy, b_round,
                      combine_cost, combine_memory, watanabe_rate)


def main():
    graph = [18, 5, 10, 4, 1]
    expected = {2: Fraction(2975, 173), 3: Fraction(11425, 692)}
    best = {}
    counts = {2: 0, 3: 0}
    for order in permutations(range(5)):
        result = graph_schedule(graph, 5, order, Fraction(1, 5))
        assert result["acceptance"] == Fraction(692, 3125)
        for memory in (2, 3):
            if result["memory"] <= memory:
                counts[memory] += 1
                best[memory] = min(best.get(memory, result["cost"]), result["cost"])
    assert best == expected and counts == {2: 8, 3: 120}
    assert best[2] / best[3] == Fraction(476, 457)
    assert combine_memory(2, 2) == 3
    assert combine_cost(Fraction(2), Fraction(2), Fraction(1, 2)) == 8
    pure = np.array([1.0, 0, 0, 0])
    assert watanabe_rate(pure) == 1
    rates = linear_code_rates(pure, [[1], [1]], [[1, 1]])
    assert rates["clear"] == .5 and rates["plain_otp"] == .5
    law = np.array([.7, .1, .1, .1])
    direct = linear_code_rates(law, [[1]], np.zeros((0, 1), int))
    assert abs(direct["clear"] - max(0, 1 - entropy(law))) < 1e-12
    transformed, y = b_round(law)
    assert abs(y - .34) < 1e-12 and abs(transformed.sum() - 1) < 1e-12
    print(json.dumps({"status": "passed", "five_input_orders": 120,
                      "feasible_orders": counts,
                      "C2": str(best[2]), "C3": str(best[3]),
                      "rate_ratio": str(best[2]/best[3]),
                      "AD_checks": "pure input, direct extraction, B-round yield"}))


if __name__ == "__main__":
    main()
