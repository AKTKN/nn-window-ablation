import numpy as np
import pytest

from nn_window_ablation.analysis import AnalysisPolicy, PairedCounts, paired_counts, summarize_pair


def test_p5_counts_sign_reversal_and_three_condition_decomposition():
    ids = np.array([22, 23, 28, 77, 107, 342], dtype=np.int64)
    a, b, c = map(np.array, ([1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]))

    def count(x, y):
        return paired_counts(x, y, ids, ids)

    direct = count(a, b)
    assert (direct.n10, direct.n01, direct.difference) == (2, 1, 1 / 6)
    assert (count(b, a).n10, count(b, a).n01, count(b, a).difference) == (1, 2, -1 / 6)
    assert count(a, b).difference == count(a, c).difference + count(c, b).difference
    assert count(a, a).categories == (3, 0, 0, 3)
    assert count(a, 1 - a).categories == (0, 3, 3, 0)
    assert count(b, c).categories == (2, 2, 2, 0)  # Equal marginals can disagree on individual shots.


@pytest.mark.parametrize(("ids", "message"), [
    ([22, 23], "missing or extra"), ([22, 23, 23], "duplicate"),
    ([23, 22, 28], "differ or are reordered"), ([22, 23, 29], "differ or are reordered"),
])
def test_p5_detects_bad_pairing_before_aggregation(ids, message):
    a, b = np.array([0, 1, 1]), np.array([1, 0, 1])
    with pytest.raises(ValueError, match=message):
        paired_counts(a, b, np.array([22, 23, 28], dtype=np.int64), np.array(ids, dtype=np.int64))


def test_p5_ratio_is_explicit_and_does_not_drop_undefined_resamples():
    policy = AnalysisPolicy(500, 45, 1, ratio_min_denominator_failures=2)
    enough = summarize_pair(PairedCounts(0, 40, 10, 50), policy, policy.rng())
    assert enough["interval_method"] == "paired_bootstrap_percentile"
    assert enough["ratio"]["status"] == "resolved"
    assert enough["ratio"]["estimate"] == 60 / 90
    assert enough["ratio"]["interval"][0] < 60 / 90 < enough["ratio"]["interval"][1]
    not_enough = summarize_pair(PairedCounts(10, 1, 0, 0), policy, policy.rng())
    assert not_enough["ratio"]["reason"] == "insufficient_denominator_failures"
    sparse = summarize_pair(PairedCounts(48, 2, 0, 0), policy, policy.rng())
    assert sparse["ratio"]["reason"] == "zero_denominator_in_bootstrap"
    assert sparse["ratio"]["undefined_bootstrap_replicates"] > 0
    assert sparse["ratio"]["estimate"] is sparse["ratio"]["interval"] is None
    assert summarize_pair(PairedCounts(10, 0, 0, 0), AnalysisPolicy(10, 7, 0), policy.rng())["ratio"] is None
