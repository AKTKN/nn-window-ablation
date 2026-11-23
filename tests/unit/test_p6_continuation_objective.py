from dataclasses import replace
from math import exp, log1p

import numpy as np
import pytest
import torch

from nn_window_ablation.data import sample_chunk
from nn_window_ablation.features import attach_supervision
from nn_window_ablation.training import MemoryObjective
from nn_window_ablation.windows import Buffers
from tests.p4_checks import assert_exact, setup_plan
from tests.p6_checks import decoder_trainer, tensor_inventory
from tests.reference.p2_observations import PILOT_SCHEDULE


def _bce(logit, bit):
    return max(logit, 0) - bit * logit + log1p(exp(-abs(logit)))


class LiteralLogits(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = torch.tensor(values, dtype=torch.float64)
        self.batch = None

    def forward(self, batch):
        self.batch = batch
        return self.values


def test_p6_global_and_sampled_core_terms_against_independent_bce_and_mean():
    bundle, plan = setup_plan(distance=5, rounds=60, p=0, chunk=2)
    original = sample_chunk(plan, bundle, 0)
    assert not original.detectors.any()
    events = original.detectors.copy()
    events[1, [0, 500, 1100, 1430]] = True
    labels = np.zeros((2, 12), dtype=bool)
    labels[1, [0, 4, 11]] = True
    global_labels = np.array([False, True])
    shots = replace(original, detectors=events, global_labels=global_labels, core_labels=labels)
    global_logits = (-1.25, 2.0)
    core_logits = [[(-1)**i * (i + row + 1) / 4 for i in range(12)] for row in (0, 1)]
    global_mean = sum(_bce(z, int(y)) for z, y in zip(global_logits, global_labels, strict=True)) / 2
    core_mean = sum(_bce(core_logits[row][i], int(labels[row, i])) for row in range(2) for i in range(12)) / 24
    for context, sampled in (("full_closed", None), ("deployment_window", None), ("sampled_window", Buffers(0, 3))):
        objective = MemoryObjective(bundle, core_context=context, fixed_buffers=Buffers(5, 5), core_weight=1)
        dynamic_or_fixed = sampled or Buffers(5, 5)
        losses = []
        for i in range(12):
            indices = np.array([i, i], dtype=np.int64)
            model = LiteralLogits((*global_logits, core_logits[0][i], core_logits[1][i]))
            measured = objective(model, shots, indices, sampled).item()
            reference = (global_mean + sum(_bce(core_logits[row][i], int(labels[row, i])) for row in range(2)) / 2) / 2
            assert measured == pytest.approx(reference, rel=0, abs=1e-14)
            losses.append(measured)
            assert model.batch.sequence_lengths[:2] == (60, 60)
        assert sum(losses) / 12 == pytest.approx((global_mean + core_mean) / 2, rel=0, abs=1e-14)

        indices = np.array([0, 11], dtype=np.int64)
        features = objective.prepare(shots, indices, sampled)
        supervised = attach_supervision(features, shots)
        np.testing.assert_array_equal(supervised.labels, [False, True, False, True])
        assert [example.shot_id for example in features.examples] == [*shots.shot_ids, *shots.shot_ids]
        assert [example.query.target.core_index for example in features.examples] == [None, None, 0, 11]
        expected_global = PILOT_SCHEDULE.select()
        for example in features.examples[:2]:
            assert sorted(detector for slot in example.query.slots for detector in slot.detector_indices) == list(expected_global)
        for example in features.examples[2:]:
            pair = None if context == "full_closed" else (dynamic_or_fixed.left, dynamic_or_fixed.right)
            assert sorted(detector for slot in example.query.slots for detector in slot.detector_indices) == list(
                PILOT_SCHEDULE.select(example.query.target.core_index, pair),
            )
        # Dropping the no-event row or changing its sampled label alters the checked objective.
        sample_index = 0
        only_second = (_bce(global_logits[1], 1) + _bce(core_logits[1][sample_index], 1)) / 2
        assert abs(only_second - losses[sample_index]) > 1e-3
        wrong_target = (_bce(core_logits[0][0], 1) + _bce(core_logits[1][0], 1)) / 2
        assert abs((global_mean + wrong_target) / 2 - losses[sample_index]) > 1e-3


def test_p6_degenerate_d_equals_w_through_prepared_inputs_gradients_and_updates():
    bundle, base = setup_plan(distance=3, rounds=36, chunk=3)
    parent = decoder_trainer(bundle, base)
    parent.step(MemoryObjective(bundle, core_context="full_closed", fixed_buffers=Buffers(3, 3), core_weight=1))
    plan = replace(base, namespace=replace(base.namespace, phase="continuation"))
    w, d = decoder_trainer(bundle, plan), decoder_trainer(bundle, plan, dynamic=True)
    for trainer in (w, d):
        trainer.fork_from(parent.snapshot())
    assert tensor_inventory(w.model.state_dict()) == tensor_inventory(d.model.state_dict())
    original_torch = torch.get_rng_state().clone()
    d_torch = d.randomness.torch_state.clone()

    class FullBuffers:
        calls = 0

        def integers(self, high, *, size):
            assert (high, size) == (3, 2)
            self.calls += 1
            return np.array([2, 2], dtype=np.int64)

    draw = FullBuffers()
    d.randomness.buffer = draw
    captured = ([], [])
    traces = []
    for trainer, context, batches in zip((w, d), ("deployment_window", "sampled_window"), captured, strict=True):
        objective = MemoryObjective(bundle, core_context=context, fixed_buffers=Buffers(3, 3), core_weight=1)

        def capture(model, shots, cores, buffers):
            features = objective.prepare(shots, cores, buffers)
            batches.append((features.examples, features.to_arrays()))
            return objective(model, shots, cores, buffers)

        traces.append(trainer.step(capture))
    assert draw.calls == 1
    assert traces[0].buffers is None and traces[1].buffers == Buffers(3, 3)
    assert traces[0].loss == traces[1].loss
    assert traces[0].trajectories == traces[1].trajectories and traces[0].cores == traces[1].cores
    assert_exact(captured[0], captured[1])
    for left, right in zip(w.model.parameters(), d.model.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)
    assert_exact(w.model.state_dict(), d.model.state_dict())
    assert_exact(w.optimizer.state_dict(), d.optimizer.state_dict())
    assert_exact(w.scheduler.state_dict(), d.scheduler.state_dict())
    assert_exact(d.randomness.torch_state, d_torch)
    assert_exact(torch.get_rng_state(), original_torch)
