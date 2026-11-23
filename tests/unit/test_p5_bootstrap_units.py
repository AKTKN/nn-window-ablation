from collections import Counter
from itertools import product
from math import factorial, prod

import numpy as np

from nn_window_ablation.analysis import bootstrap_counts, paired_counts


def test_p5_multinomial_equals_enumerated_whole_trajectory_bootstrap():
    # Five distinct trajectories, with the (0,0) outcome repeated.
    outcomes = np.array([[0, 0], [0, 0], [0, 1], [1, 0], [1, 1]])
    ids = np.arange(5, dtype=np.int64)
    modal_counts = paired_counts(*outcomes.T, ids, ids)
    trajectory_counts = Counter()
    for indices in product(range(5), repeat=5):
        pairs = [tuple(pair) for pair in outcomes[list(indices)]]
        trajectory_counts[sum(a - b for a, b in pairs)] += 1 / 5**5
    multinomial_counts = Counter()
    for counts in product(range(6), repeat=4):
        if sum(counts) != 5:
            continue
        coefficient = factorial(5) / prod(factorial(count) for count in counts)
        probability = coefficient * prod((count / 5)**replicas for count, replicas in zip((2, 1, 1, 1), counts, strict=True))
        multinomial_counts[counts[2] - counts[1]] += probability
    assert multinomial_counts.keys() == trajectory_counts.keys()
    np.testing.assert_allclose(list(map(multinomial_counts.get, sorted(trajectory_counts))),
                               list(map(trajectory_counts.get, sorted(trajectory_counts))), atol=1e-14, rtol=0)

    first = bootstrap_counts(modal_counts, 80, np.random.Generator(np.random.PCG64(11)))
    second = bootstrap_counts(modal_counts, 80, np.random.Generator(np.random.PCG64(11)))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, np.random.Generator(np.random.PCG64(11)).multinomial(5, [0.4, 0.2, 0.2, 0.2], 80))


def test_p5_shared_row_keeps_every_condition_and_core_together():
    # Both conditions always have even parity under intact rows.
    errors = np.array([[[0, 0], [1, 1]], [[1, 1], [0, 0]]])
    enumerate_rows = lambda: product(range(2), repeat=2)
    for selected in enumerate_rows():
        data = errors[list(selected)]
        np.testing.assert_array_equal(data[:, :, 0], data[:, :, 1])
        np.testing.assert_array_equal(data[:, 0, 0] + data[:, 1, 0], [1, 1])
        assert not (data.sum(axis=2) % 2).any()
    resample = lambda: np.random.Generator(np.random.PCG64(17)).integers(2, size=(9, 2))
    np.testing.assert_array_equal(errors[resample()], errors[resample()])

    wrong_flags = [
        int((errors[left, 0, 0] + errors[right, 0, 1]) % 2)
        for left, right in enumerate_rows()
    ]
    assert sum(wrong_flags) / len(wrong_flags) == 0.5  # Incorrect independent window resampling breaks cancellation.
