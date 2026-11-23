from math import comb, sqrt
from statistics import NormalDist

import pytest

from nn_window_ablation.analysis import AnalysisPolicy, PairedCounts, binomial_interval, sparse_difference_interval, summarize_pair


def test_p5_wilson_against_analytical_score_interval_and_endpoints():
    z = NormalDist().inv_cdf(0.975)
    for k in (0, 4, 10):
        p = k / 10
        center = (p + z**2 / 20) / (1 + z**2 / 10)
        spread = z * sqrt(p * (1 - p) / 10 + z**2 / 400) / (1 + z**2 / 10)
        assert binomial_interval(k, 10, confidence=0.95, method="wilson") == pytest.approx(
            (max(0, center - spread), min(1, center + spread)), abs=1e-14,
        )
    assert binomial_interval(0, 10, confidence=0.95, method="wilson") == pytest.approx((0, 0.2775328), abs=1e-7)
    assert binomial_interval(10, 10, confidence=0.95, method="wilson") == pytest.approx((0.7224672, 1), abs=1e-7)


@pytest.mark.parametrize(("k", "n", "confidence"), [(0, 0, 0.95), (-1, 4, 0.95), (5, 4, 0.95),
                                                        (1, 4, 0), (1, 4, 1), (1, 4, float("nan"))])
def test_p5_rejects_invalid_interval_inputs(k, n, confidence):
    with pytest.raises(ValueError, match="count|confidence"):
        binomial_interval(k, n, confidence=confidence, method="wilson")


def _reference_exact(k, n):
    def invert_tail(first):
        low, high = 0., 1.
        for _ in range(85):
            middle = (low + high) / 2
            tail = sum(comb(n, j) * middle**j * (1 - middle)**(n - j) for j in range(first, n + 1))
            if tail < 0.0125:
                low = middle
            else:
                high = middle
        return (low + high) / 2

    return (invert_tail(k) if k else 0), (1 - invert_tail(n - k) if k < n else 1)


def test_p5_exact_marginals_and_sparse_difference_against_binomial_tails():
    for k in range(11):
        assert binomial_interval(k, 10, confidence=0.975, method="exact") == pytest.approx(_reference_exact(k, 10), abs=1e-12)
    counts = PairedCounts(6, 3, 1, 0)
    lower10, upper10 = _reference_exact(1, 10)
    lower01, upper01 = _reference_exact(3, 10)
    assert sparse_difference_interval(counts) == pytest.approx((lower10 - upper01, upper10 - lower01), abs=1e-12)
    policy = AnalysisPolicy(10, 1, 0)
    zero = summarize_pair(PairedCounts(7, 0, 0, 3), policy, policy.rng())
    u = 1 - 0.0125**0.1
    assert zero["difference"] == 0 and zero["interval"] == pytest.approx((-u, u), abs=1e-12)
