from copy import deepcopy

import numpy as np
import pytest
import torch

from nn_window_ablation.artifacts import ArtifactError
from nn_window_ablation.checkpoints import load_checkpoint, load_evaluation_model, save_checkpoint
from nn_window_ablation.config import resolve_config
from nn_window_ablation.data import SamplingPlan, StreamNamespace
from nn_window_ablation.simulation import CircuitSpec, build_circuit_bundle
from nn_window_ablation.streams import TrajectoryStream
from nn_window_ablation.training import MemoryObjective, Trainer, UpdateBudget, fresh_decoder
from nn_window_ablation.windows import Buffers
from tests.p4_checks import assert_exact


@pytest.fixture
def deterministic_cpu():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.get_num_threads() == torch.get_num_interop_threads() == 1
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def test_full_d5_model_replays_four_declared_updates_exactly(tmp_path, compose_settings, deterministic_cpu):
    config = resolve_config(compose_settings(
        "runtime=test_cpu", "data.split=train", "data.training_chunk_shots=3", "training.max_steps=4",
        "training.batch_size=2", "training.microbatch_size=1", "training.learning_rate=0.001",
        "training.optimizer=adam", "training.scheduler=step", "training.scheduler_step_size=1",
        "training.scheduler_gamma=0.9",
    ), invocation_dir=tmp_path)
    geometry, settings = config.geometry, config.training
    spec = CircuitSpec(geometry.distance, geometry.rounds, geometry.core_length, config.noise.probability)
    assert spec == CircuitSpec(5, 60, 5, 0.003)
    bundle = build_circuit_bundle(spec)
    namespace = StreamNamespace(config.seed, config.data.family_id, "train", 5, 0, "base")
    plan = SamplingPlan.from_circuit(bundle, namespace, chunk_rows=config.data.training_chunk_shots)
    objective = MemoryObjective(bundle, core_context="full_closed", fixed_buffers=Buffers(5, 5), core_weight=1.0)
    budget = UpdateBudget(settings.max_steps, settings.batch_size, settings.microbatch_size)

    def reconstruct():
        model = fresh_decoder(plan)
        assert (model.rnn.hidden_size, model.rnn.num_layers, model.rnn.dropout) == (256, 4, 0.0)
        adam = torch.optim.Adam(model.parameters(), lr=settings.learning_rate, foreach=False)
        scheduler = torch.optim.lr_scheduler.StepLR(adam, settings.scheduler_step_size, gamma=settings.scheduler_gamma)
        return Trainer(model, adam, scheduler, TrajectoryStream(plan, bundle), budget, architecture=model.architecture())

    uninterrupted = reconstruct()
    expected_traces = [uninterrupted.step(objective) for _ in range(4)]
    interrupted = reconstruct()
    observed_traces = [interrupted.step(objective) for _ in range(2)]
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, interrupted.snapshot())
    with pytest.raises(ArtifactError, match="fixed terminal update"):
        load_evaluation_model(checkpoint)

    resumed = reconstruct()
    middle = load_checkpoint(checkpoint)
    assert middle["contract"]["budget"]["max_steps"] == 4
    assert middle["cursor"] == {"chunk": 1, "row": 1}
    assert middle["optimizer_state"]["state"] and middle["scheduler_state"]["last_epoch"] == 2
    resumed.restore(middle)
    assert_exact(resumed.snapshot(), interrupted.snapshot())
    observed_traces += [resumed.step(objective) for _ in range(2)]
    assert observed_traces == expected_traces
    assert len({identity for trace in observed_traces for identity in trace.trajectories}) == 8
    assert all(len(trace.cores) == 2 and all(0 <= index < 12 for index in trace.cores) for trace in observed_traces)
    assert resumed.counters == {
        "parent_updates": 0, "phase_updates": 4, "cumulative_updates": 4, "trajectories": 8, "query_examples": 16,
    }
    assert_exact(resumed.snapshot(), uninterrupted.snapshot())
    save_checkpoint(checkpoint, resumed.snapshot())
    evaluator = load_evaluation_model(checkpoint)
    assert not evaluator.training
    assert_exact(evaluator.state_dict(), uninterrupted.model.state_dict())

    expected_rows = uninterrupted.stream.take(2)
    next_rows = resumed.stream.take(2)
    assert next_rows.identities == expected_rows.identities
    np.testing.assert_array_equal(next_rows.shots.detectors, expected_rows.shots.detectors)
    np.testing.assert_array_equal(
        resumed.randomness.core.integers(12, size=2), uninterrupted.randomness.core.integers(12, size=2),
    )

    wrong_cursor = {**middle, "cursor": {"chunk": 0, "row": 0}}
    with pytest.raises(ArtifactError, match="cursor disagrees with consumed trajectory count"):
        reconstruct().restore(wrong_cursor)
    wrong_moments = deepcopy(middle)
    for moments in wrong_moments["optimizer_state"]["state"].values():
        moments["exp_avg"].zero_()
    diverges = reconstruct()
    diverges.restore(wrong_moments)
    for _ in range(2):
        diverges.step(objective)
    assert not torch.equal(diverges.model.head.weight, uninterrupted.model.head.weight)
