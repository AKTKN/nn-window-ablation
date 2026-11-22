from math import exp, log1p

import numpy as np
import pytest
import torch

from nn_window_ablation.models.gnn_rnn import hard_decisions, probabilities
from nn_window_ablation.models.objective import query_loss, sample_core_indices


def _bce(label, logit):
    return max(logit, 0) - label * logit + log1p(exp(-abs(logit)))


def _derivative(label, logit):
    exponential = exp(-abs(logit))
    sigmoid = 1 / (1 + exponential) if logit >= 0 else exponential / (1 + exponential)
    return sigmoid - label


def _oracle(function, labels, logits):
    return np.array([function(y, z) for y, z in zip(labels.flat, logits.flat, strict=True)]).reshape(labels.shape)


def _double(values, requires_grad=False):
    return torch.tensor(values, dtype=torch.float64, requires_grad=requires_grad)


def _check(actual, expected):
    np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=1e-12, atol=1e-12)


def test_p3_bce_extreme_zero_and_ordinary_logits():
    global_values = np.array([-1000., 0., 1.5, 1000.])
    global_targets = np.array([1., 1., 0., 1.])
    core_values = np.array([[1000., 0.], [-1000., 2.], [0., -2.], [1000., -1000.]])
    core_targets = np.array([[0., 0.], [0., 1.], [1., 1.], [1., 1.]])
    global_logits = _double(global_values, requires_grad=True)
    core_logits = _double(core_values, requires_grad=True)
    result = query_loss(global_logits, _double(global_targets), core_logits, _double(core_targets))

    g = _oracle(_bce, global_targets, global_values).mean()
    c = _oracle(_bce, core_targets, core_values).mean()
    _check(result.global_term, g)
    _check(result.core_term, c)
    _check(result.total, (g + c) / 2)
    global_gradient, core_gradient = torch.autograd.grad(result.total, (global_logits, core_logits))
    _check(global_gradient, _oracle(_derivative, global_targets, global_values) / (2 * global_values.size))
    _check(core_gradient, _oracle(_derivative, core_targets, core_values) / (2 * core_values.size))
    assert hard_decisions(global_logits).tolist() == [False, True, True, True]
    assert probabilities(global_logits)[1].item() == 0.5


def test_p3_all_twelve_sampled_choices_match_full_loss_and_affine_gradients():
    g_x = np.array([[1., 2., -1.], [-0.5, -1.5, 2.]])
    c_x = np.arange(2 * 12 * 3).reshape(2, 12, 3) / 17 - 2
    theta = np.array([0.2, -0.3, 0.10])
    g_targets = np.array([1., 0.])
    c_targets = (np.arange(24).reshape(2, 12) % 3 != 0).astype(float)
    g_values = g_x @ theta
    c_values = c_x @ theta
    g_expected = _oracle(_derivative, g_targets, g_values) / 4
    c_expected = _oracle(_derivative, c_targets, c_values) / 48
    affine_expected = g_x.T @ g_expected + np.sum(c_x * c_expected[..., None], axis=(0, 1))
    loss_expected = (_oracle(_bce, g_targets, g_values).mean() + _oracle(_bce, c_targets, c_values).mean()) / 2

    parameters = _double(theta, requires_grad=True)
    g_logits = _double(g_x) @ parameters
    c_logits = _double(c_x) @ parameters
    labels = _double(c_targets)
    global_labels = _double(g_targets)
    full = query_loss(g_logits, global_labels, c_logits, labels).total
    _check(full, loss_expected)
    for gradient, expected in zip(
        torch.autograd.grad(full, (g_logits, c_logits, parameters), retain_graph=True),
        (g_expected, c_expected, affine_expected), strict=True,
    ):
        _check(gradient, expected)

    # Each trajectory independently visits every possible core in this
    # finite average. Linearity of the sum over trajectories suffices.
    choices = []
    for i in range(12):
        indices = torch.tensor([i, (i + 5) % 12])
        sampled = c_logits[torch.arange(2), indices]
        choices.append(query_loss(g_logits, global_labels, sampled, labels, core_indices=indices).total)
    enumerated = torch.stack(choices).mean()
    _check(enumerated, loss_expected)
    for gradient, expected in zip(
        torch.autograd.grad(enumerated, (g_logits, c_logits, parameters)),
        (g_expected, c_expected, affine_expected), strict=True,
    ):
        _check(gradient, expected)


def test_p3_checker_detects_wrong_normalization_core_sum_and_index():
    global_logits = _double([1., -2.])
    global_labels = _double([0., 1.])
    core_logits = _double([2., -1.])
    labels = _double(np.tile([0., 1.], (2, 6)))
    indices = torch.tensor([0, 11])
    terms = query_loss(global_logits, global_labels, core_logits, labels, core_indices=indices)

    expected_core = (_bce(0, 2) + _bce(1, -1)) / 2
    expected_global = (_bce(0, 1) + _bce(1, -2)) / 2
    expected = (expected_global + expected_core) / 2
    _check(terms.total, expected)
    with pytest.raises(AssertionError):
        _check(terms.global_term + terms.core_term, expected)
    with pytest.raises(AssertionError):
        _check((terms.global_term + 12 * terms.core_term) / 2, expected)
    with pytest.raises(AssertionError):
        rolled = query_loss(global_logits, global_labels, core_logits, labels, core_indices=(indices + 1) % 12)
        _check(rolled.total, expected)
    for invalid in (torch.tensor([-1, 1]), torch.tensor([1, 12])):
        with pytest.raises(ValueError, match="outside"):
            query_loss(global_logits, global_labels, core_logits, labels, core_indices=invalid)


def test_p3_uniform_indices_use_the_supplied_stream():
    rng = np.random.default_rng(820)
    independent = np.random.default_rng(820)
    for trajectories in (2, 32):
        result = sample_core_indices(trajectories, 12, rng=rng)
        expected = independent.integers(0, 12, size=trajectories, dtype=np.int64)
        assert result.dtype == np.int64
        np.testing.assert_array_equal(result, expected)
    with pytest.raises(ValueError, match="positive integers"):
        sample_core_indices(1, 0, rng=rng)
