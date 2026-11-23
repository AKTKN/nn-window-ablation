from collections import Counter
from hashlib import sha256

import numpy as np
import pytest

import nn_window_ablation.execution as execution
from nn_window_ablation.artifacts import ArtifactError
from nn_window_ablation.checkpoints import load_checkpoint
from nn_window_ablation.data import SamplingPlan, open_dataset
from nn_window_ablation.evaluation import CheckpointProvenance
from nn_window_ablation.execution import generate_workflow, train_workflow
from tests.p4_checks import assert_exact, reference_seed
from tests.p6_checks import TEST_BUDGET, continuation_pilot, run_nnwa


def test_p6_d_resume_in_new_process_replays_remaining_updates_exactly(continuation_pilot, tmp_path):
    resolve, _ = continuation_pilot
    training = ("training=dynamic_window", "training.max_steps=3", *TEST_BUDGET)
    parent = "checkpoint.init_from=runs/base/checkpoint.pt"
    uninterrupted = train_workflow(resolve("run_name=uninterrupted", "data.split=train", *training, parent))
    initial_path = train_workflow(resolve("run_name=pause", "data.split=train", *training, parent), stop_after=0)
    initial = load_checkpoint(initial_path)
    assert not initial["optimizer_state"]["state"] and initial["scheduler_state"]["last_epoch"] == 0
    assert initial["optimizer_state"]["param_groups"][0]["lr"] == 0.002
    replay = "checkpoint.resume_from=runs/pause/checkpoint.pt"
    path = train_workflow(resolve("run_name=pause", "data.split=train", *training, replay), stop_after=1)
    partial = load_checkpoint(path)
    assert partial["counters"]["phase_updates"] == 1 and partial["scheduler_state"]["last_epoch"] == 1
    assert Counter(int(state["step"]) for state in partial["optimizer_state"]["state"].values()).keys() == {1}
    with pytest.raises(ArtifactError, match="resolved configuration"):
        train_workflow(resolve("run_name=pause", "data.split=train", *training, "training.learning_rate=0.003", replay))

    run_nnwa(tmp_path, "train", "runtime=test_cpu", "run_name=pause", *training, replay)
    expected, resumed = load_checkpoint(uninterrupted), load_checkpoint(path)
    for key in ("contract", "model", "optimizer_state", "scheduler_state", "counters", "cursor", "last_update", "random_states"):
        assert_exact(resumed[key], expected[key])
    left, right = resumed["continuation_audit"], expected["continuation_audit"]
    for key in ("controls", "controls_sha256", "input_sha256", "trace_prefix", "updates"):
        assert_exact(left[key], right[key])
    assert left["trace_prefix"][:1] == partial["continuation_audit"]["trace_prefix"]
    assert left["resources"]["detector_tokens"] == right["resources"]["detector_tokens"]
    assert resumed["counters"]["phase_updates"] == 3 and resumed["counters"]["cumulative_updates"] == 6
    assert resumed["scheduler_state"]["last_epoch"] == 3
    assert set(int(state["step"]) for state in resumed["optimizer_state"]["state"].values()) == {3}
    plan = SamplingPlan.from_record(left["controls"]["trainer"]["source"]["plan"])
    rng = np.random.default_rng(reference_seed(plan.namespace, "dynamic_buffer"))
    lengths = (0, 3, 5)
    picks = [tuple(lengths[i] for i in rng.integers(3, size=2)) for _ in range(3)]
    assert [tuple(trace["buffers"][side] for side in ("left", "right")) for trace in left["trace_prefix"]] == picks
    assert_exact(resumed["random_states"]["dynamic_buffer"], rng.bit_generator.state)
    digest = sha256(path.read_bytes()).hexdigest()
    run_nnwa(tmp_path, "train", "runtime=test_cpu", "run_name=pause", *training, replay)
    assert sha256(path.read_bytes()).hexdigest() == digest


def test_p6_terminal_validation_failure_retries_without_an_extra_update(continuation_pilot, tmp_path, monkeypatch):
    resolve, _ = continuation_pilot
    validation = generate_workflow(resolve("data.split=validation", "data.id=val", "data.shots=1", "data.shard_shots=1"))
    mapped = open_dataset(validation.parent, expected_split="validation")
    settings = (
        "run_name=validated", "data.split=train", "training=fixed_window", "training.max_steps=1",
        "training.validation_data_id=val", *TEST_BUDGET,
    )

    def interrupted(*_):
        raise OSError("validation interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(execution, "_terminal_validation", interrupted)
        with pytest.raises(OSError, match="validation interrupted"):
            train_workflow(resolve(*settings, "checkpoint.init_from=runs/base/checkpoint.pt"))
    path = tmp_path / "runs/validated/checkpoint.pt"
    old = load_checkpoint(path)
    assert old["counters"]["phase_updates"] == 1 and old["continuation_audit"]["validation"] is None
    with pytest.raises(ArtifactError, match="terminal continuation validation has not completed"):
        CheckpointProvenance.from_payload(old, sha256(path.read_bytes()).hexdigest(), mapped.plan)

    assert train_workflow(resolve(*settings, "checkpoint.resume_from=runs/validated/checkpoint.pt")) == path
    complete = load_checkpoint(path)
    for key in ("model", "optimizer_state", "scheduler_state", "counters", "random_states", "cursor"):
        assert_exact(complete[key], old[key])
    assert complete["continuation_audit"]["validation"]["shots"] == 1
