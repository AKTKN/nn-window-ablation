import numpy as np
import pytest

from nn_window_ablation.analysis import CoreDiagnostics, core_disagreements
from nn_window_ablation.evaluation import compose_core_decisions


def test_p5_xor_cancellation_with_nonzero_targets():
    targets = np.array([
        [1, 0, 1, 1], [0, 1, 1, 0], [0, 1, 0, 1], [1, 1, 1, 0],
    ], dtype=np.uint8)
    predictions = np.array([
        [1, 0, 1, 1], [1, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0],
    ], dtype=np.uint8)
    targets = np.pad(targets, ((0, 0), (0, 8)))
    predictions = np.pad(predictions, ((0, 0), (0, 8)))
    parity = lambda row: sum(int(bit) for bit in row) % 2
    global_labels = np.array([parity(row) for row in targets])
    expected_predictions = [parity(row) for row in predictions]
    expected = np.array([
        [int(left != right) for left, right in zip(a, b, strict=True)]
        for a, b in zip(predictions, targets, strict=True)
    ])
    errors = core_disagreements(predictions, targets, global_labels)
    np.testing.assert_array_equal(errors, expected)
    np.testing.assert_array_equal(compose_core_decisions(predictions), expected_predictions)
    assert [parity(row) for row in errors] == [0, 1, 0, 1]
    assert [int(a != b) for a, b in zip(expected_predictions, global_labels, strict=True)] == [0, 1, 0, 1]
    diagnostics = CoreDiagnostics()
    diagnostics.add(errors[:1])
    diagnostics.add(errors[1:])
    report = diagnostics.report()
    assert report["disagreement_count_histogram"] == [1, 1, 1, 1] + [0] * 9
    assert report["odd_disagreement_rate"] == 0.5
    assert report["per_core_counts"][:4] == [2, 2, 2, 0]
    assert report["adjacent_joint_counts"][:3] == [1, 2, 0]
    assert report["adjacent_correlations"][:3] == [0.0, 1.0, None]


@pytest.mark.parametrize("invalid", [
    np.zeros(12, dtype=bool), np.zeros((1, 11), dtype=bool), np.zeros((0, 12), dtype=bool),
    np.full((1, 12), 2), np.full((1, 12), -1), np.zeros((1, 12), dtype=float),
])
def test_p5_rejects_malformed_core_decisions(invalid):
    with pytest.raises(ValueError, match="complete row|binary"):
        compose_core_decisions(invalid)


def test_p5_rejects_misaligned_or_inconsistent_labels():
    bits = np.zeros((3, 12), dtype=bool)
    with pytest.raises(ValueError, match="matching trajectory and core dimensions"):
        core_disagreements(bits, bits[:2], np.zeros(2, dtype=bool))
    with pytest.raises(ValueError, match="disagree with the global label"):
        core_disagreements(bits, bits, np.ones(3, dtype=bool))
