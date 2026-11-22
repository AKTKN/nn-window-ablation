from dataclasses import replace

import numpy as np
import pytest
import stim

from nn_window_ablation.artifacts import ArtifactError
from nn_window_ablation.data import TrajectoryID, open_dataset, sample_chunk, write_dataset
from nn_window_ablation.models.objective import sample_core_indices
from nn_window_ablation.simulation import sample_shots
import nn_window_ablation.streams as stream_module
from nn_window_ablation.streams import ChunkCursor, TrajectoryStream
from nn_window_ablation.training import TrainingRandomness
from tests.p4_checks import assert_exact, reference_seed, setup_plan


def test_p4_chunk_records_independent_of_worker_assignment_and_restart(tmp_path):
    bundle, plan = setup_plan(chunk=3, total=8)
    alternate_assignment = {index: sample_chunk(plan, bundle, index) for index in (2, 0, 1)}
    for index, count in ((0, 3), (1, 3), (2, 2)):
        expected = sample_shots(
            bundle, count, seed=reference_seed(plan.namespace, "circuit", index), first_shot_id=plan.shot_prefix + index * 3,
        )
        for field in ("detectors", "global_labels", "core_labels", "shot_ids"):
            assert_exact(getattr(alternate_assignment[index], field), getattr(expected, field))
    stream = TrajectoryStream(plan, bundle)
    first = stream.take(4)
    random_streams = TrainingRandomness(plan.namespace)
    first_cores = sample_core_indices(4, plan.core_count, rng=random_streams.core)
    assert stream.cursor == ChunkCursor(1, 1)
    resumed = TrajectoryStream(plan, bundle)
    resumed.seek(stream.cursor)
    rest = resumed.take(4)
    restart_streams = TrainingRandomness(plan.namespace)
    restart_streams.restore(random_streams.record())
    rest_cores = sample_core_indices(4, plan.core_count, rng=restart_streams.core)
    expected_ids = [
        TrajectoryID(plan.dataset_id, chunk, int(row))
        for chunk, rows in ((0, 3), (1, 3), (2, 2))
        for row in np.random.Generator(np.random.PCG64(reference_seed(plan.namespace, "trajectory_order", chunk))).permutation(rows)
    ]
    assert first.identities + rest.identities == tuple(expected_ids)
    expected_core_rng = np.random.default_rng(reference_seed(plan.namespace, "core_query"))
    assert_exact(np.concatenate([first_cores, rest_cores]), expected_core_rng.integers(0, plan.core_count, size=8))
    assert_exact(rest.shots.detectors, stream.take(4).shots.detectors)
    with pytest.raises(ValueError, match="complete trajectory batch"):
        resumed.take(1)

    destination = tmp_path / "stored"
    write_dataset(destination, plan, bundle, provenance={})
    mapped = open_dataset(destination, expected_split="train")
    stored = TrajectoryStream(plan, bundle, mapped).take(8)
    assert stored.identities == tuple(expected_ids)
    for field in ("detectors", "global_labels", "core_labels", "shot_ids"):
        assert_exact(getattr(stored.shots, field), np.concatenate([
            getattr(first.shots, field), getattr(rest.shots, field),
        ]))


def test_p4_continuation_namespace_shared_across_arms_and_buffer_draws_are_independent():
    bundle, base = setup_plan(chunk=3)
    phase = replace(base.namespace, phase="continuation")
    continuation = replace(base, namespace=phase)

    def trace(plan, draws):
        stream = TrajectoryStream(plan, bundle)
        rngs = TrainingRandomness(plan.namespace)
        rngs.buffer.integers(0, 3, size=draws)
        return stream.take(5), sample_core_indices(5, plan.core_count, rng=rngs.core)

    # Three arms use one plan. Their context-specific buffer streams may be
    # at different positions before the physical/core selection.
    continued, core_expected = trace(continuation, 0)
    for buffer_draws in (1, 99):
        actual, cores = trace(continuation, buffer_draws)
        assert actual.identities == continued.identities
        assert_exact(actual.shots.detectors, continued.shots.detectors)
        assert_exact(cores, core_expected)
    assert_exact(core_expected, np.random.default_rng(reference_seed(phase, "core_query")).integers(0, 2, size=5))
    assert not set(trace(base, 0)[0].identities) & set(continued.identities)
    assert base.namespace.seed("circuit", 0) != phase.seed("circuit", 0)
    assert base.namespace.seed("core_query") != phase.seed("core_query")
    for wrong_plan in (base, replace(continuation, namespace=replace(phase, family="p4-fixture-D"))):
        with pytest.raises(AssertionError):
            assert trace(wrong_plan, 0)[0].identities == continued.identities


def test_p4_transient_next_chunk_failure_keeps_valid_previous_cache(monkeypatch):
    bundle, plan = setup_plan(chunk=3, total=5)
    stream = TrajectoryStream(plan, bundle)
    stream.take(2)
    expected = TrajectoryStream(plan, bundle)
    expected.seek(ChunkCursor(0, 2))
    real_sample = stream_module.sample_chunk

    def fail_next(plan, bundle, index):
        if index == 1:
            raise OSError("next chunk failed")
        return real_sample(plan, bundle, index)

    with monkeypatch.context() as patch:
        patch.setattr(stream_module, "sample_chunk", fail_next)
        with pytest.raises(OSError, match="next chunk failed"):
            stream.take(2)
    assert stream.cursor == ChunkCursor(0, 2)
    assert stream.take(2).identities == expected.take(2).identities


def test_p4_frozen_rows_are_readable_but_new_sampling_requires_the_recorded_stim(tmp_path, monkeypatch):
    bundle, plan = setup_plan(split="test", total=2)
    path = tmp_path / "test"
    write_dataset(path, plan, bundle, provenance={})
    original = sample_chunk(plan, bundle, 0)
    monkeypatch.setattr(stim, "__version__", "simulated-different-stim")

    loaded = open_dataset(path, expected_split="test")
    assert_exact(loaded.shards[0].read().shots.detectors, original.detectors)
    with pytest.raises(ArtifactError, match="sampling recipe requires Stim .*found simulated-different-stim"):
        sample_chunk(loaded.plan, loaded.bundle, 0)
