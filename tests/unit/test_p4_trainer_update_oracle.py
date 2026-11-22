import random

import pytest
import torch

from nn_window_ablation.checkpoints import load_checkpoint, save_checkpoint
from tests.p4_checks import assert_exact, convex_loss, scalar_trainer, setup_plan


def test_p4_update_and_uneven_accumulation_against_convex_recurrence():
    bundle, plan = setup_plan(chunk=3)
    accumulated = scalar_trainer(bundle, plan, steps=4, micro=2)
    whole = scalar_trainer(bundle, plan, steps=4, micro=3)
    expected = incorrect = -2.0
    for update in range(1, 5):
        left = accumulated.step(convex_loss)
        right = whole.step(convex_loss)
        assert left.trajectories == right.trajectories and left.cores == right.cores
        expected = (1 - 0.125) * expected + 0.125 * (14 / 3)
        incorrect = (1 - 0.125 / 3) * incorrect + (0.125 / 3) * (14 / 3)
        for trainer in (accumulated, whole):
            assert trainer.model.theta.item() == pytest.approx(expected, abs=1e-15, rel=1e-15)
            assert trainer.counters == {
                "parent_updates": 0, "phase_updates": update, "cumulative_updates": update,
                "trajectories": 3 * update, "query_examples": 6 * update,
            }
        with pytest.raises(AssertionError):
            assert accumulated.model.theta.item() == pytest.approx(incorrect, abs=1e-15, rel=1e-15)
    with pytest.raises(ValueError, match="declared update budget"):
        accumulated.step(convex_loss)


def test_p4_stochastic_resume_restores_torch_python_and_logical_buffer_draws(tmp_path):
    bundle, plan = setup_plan(chunk=3)

    def stochastic(log):
        def objective(model, shots, cores, buffers):
            log.append((len(shots.shot_ids), buffers))
            inputs = torch.arange(1, len(cores) + 1, dtype=torch.float64)
            activated = torch.nn.functional.dropout(inputs, p=0.35, training=model.training)
            target = activated + random.random()
            return ((model(len(target)) - target) ** 2).mean() / 2
        return objective

    trainer = scalar_trainer(bundle, plan, steps=4, adam=True, dynamic=True)
    log = []
    objective = stochastic(log)
    first, second = trainer.step(objective), trainer.step(objective)
    assert log == [(2, first.buffers), (1, first.buffers), (2, second.buffers), (1, second.buffers)]
    path = tmp_path / "rng.pt"
    save_checkpoint(path, trainer.snapshot())
    log.clear()
    intended = [trainer.step(objective), trainer.step(objective)]

    resumed = scalar_trainer(bundle, plan, steps=4, adam=True, dynamic=True)
    wrong_rng = resumed.randomness.record()
    resumed.restore(load_checkpoint(path))
    resumed_log = []
    resumed_objective = stochastic(resumed_log)
    assert [resumed.step(resumed_objective), resumed.step(resumed_objective)] == intended
    assert resumed_log == log
    assert_exact(resumed.snapshot(), trainer.snapshot())

    for role in ("torch_cpu", "python"):
        wrong = load_checkpoint(path)
        wrong["random_states"][role] = wrong_rng[role]
        restarted = scalar_trainer(bundle, plan, steps=4, adam=True, dynamic=True)
        restarted.restore(wrong)
        trace = restarted.step(stochastic([]))
        assert trace.trajectories == intended[0].trajectories and trace.cores == intended[0].cores
        with pytest.raises(AssertionError):
            assert trace.loss == intended[0].loss
