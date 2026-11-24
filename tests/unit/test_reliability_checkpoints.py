from dataclasses import replace
import json
import random

import pytest
import torch

from nn_window_ablation import artifacts, checkpoints, execution, training
from nn_window_ablation.artifacts import ArtifactError
from nn_window_ablation.checkpoints import load_checkpoint, save_checkpoint
from nn_window_ablation.config import ConfigError, resolve_config
from tests.p4_checks import assert_exact, convex_loss, scalar_trainer, setup_plan


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float64))
        self.register_buffer("scale", torch.tensor(1.5, dtype=torch.float64))

    def architecture(self):
        return {"test": "scalar-cadence"}


class SmallObjective:
    def __init__(self, bundle, *, core_context, fixed_buffers, core_weight):
        self.fixed_buffers = fixed_buffers
        self.core_weight = core_weight

    def detector_tokens(self, cores, buffers):
        return len(cores)

    def __call__(self, model, shots, cores, buffers):
        margin = 0 if buffers is None else buffers.left + 2 * buffers.right
        target = (sum(cores) + margin) * 0.1 + random.random() + torch.rand((), dtype=torch.float64)
        return (model.weight * model.scale - target).square()


@pytest.fixture
def cheap_training(tmp_path, compose_settings, monkeypatch):
    monkeypatch.setattr(training, "fresh_decoder", lambda _plan: SmallModel())
    monkeypatch.setattr(training, "MemoryObjective", SmallObjective)

    def compose(name, *overrides):
        return resolve_config(compose_settings(
            "runtime=test_cpu", "data.split=train", "code.distance=3", f"run_name={name}",
            "training.max_steps=5", "training.batch_size=2", "training.microbatch_size=1",
            "training.learning_rate=0.01", "training.optimizer=adam", "training.scheduler=step",
            "training.scheduler_step_size=1", "training.scheduler_gamma=0.8", "data.training_chunk_shots=3",
            *overrides,
        ), invocation_dir=tmp_path)

    base = execution.train_workflow(compose("base", "training.max_steps=2"))

    def continuation(name, *overrides, resume=False):
        source = (
            f"checkpoint.resume_from=runs/{name}/checkpoint.pt" if resume
            else f"checkpoint.init_from={base}"
        )
        return compose(name, "training=dynamic_window", source, *overrides)

    return continuation


def _assert_same_state(actual, expected):
    for key in ("model", "optimizer_state", "scheduler_state", "counters", "cursor", "random_states", "last_update"):
        assert_exact(actual[key], expected[key])
    for key in ("trace_prefix", "input_sha256", "controls", "controls_sha256"):
        assert_exact(actual["continuation_audit"][key], expected["continuation_audit"][key])


@pytest.mark.parametrize("pause", [1, 3])
def test_learning_rate_floor_survives_continuation_resume(cheap_training, pause):
    schedule = "training.scheduler_gamma=0.2"
    reference = load_checkpoint(execution.train_workflow(cheap_training("reference", schedule)))
    path = execution.train_workflow(cheap_training("pause", schedule), stop_after=0)
    initial = load_checkpoint(path)
    assert initial["optimizer_state"]["param_groups"][0]["lr"] == 0.01
    assert initial["scheduler_state"]["base_lrs"] == [0.01]

    execution.train_workflow(cheap_training("pause", schedule, resume=True), stop_after=pause)
    partial = load_checkpoint(path)
    expected_rate = 0.002 if pause == 1 else 0.001
    assert partial["optimizer_state"]["param_groups"][0]["lr"] == expected_rate
    assert partial["scheduler_state"]["_last_lr"] == [expected_rate]
    execution.train_workflow(cheap_training("pause", schedule, resume=True))
    resumed = load_checkpoint(path)
    assert resumed["optimizer_state"]["param_groups"][0]["lr"] == 0.001
    assert resumed["scheduler_state"]["_last_lr"] == [0.001]
    _assert_same_state(resumed, reference)


def test_sampled_update_logging_preserves_resume_and_checkpoint_state(cheap_training):
    reference = execution.train_workflow(cheap_training("reference"))
    reference_log, = reference.parent.glob("train-attempts/*/rank-0.jsonl")
    assert all(json.loads(line)["event"] != "update" for line in reference_log.read_text().splitlines())

    config = cheap_training("sampled", "training.log_every_updates=2")
    path = execution.train_workflow(config, stop_after=3)
    execution.train_workflow(cheap_training("sampled", "training.log_every_updates=4", resume=True))
    updates = []
    for log in path.parent.glob("train-attempts/*/rank-0.jsonl"):
        events = [json.loads(line) for line in log.read_text().splitlines()]
        assert events[0]["event"] == "training_start" and events[-1]["event"] == "training_end"
        updates.extend(event["update"] for event in events if event["event"] == "update")
    assert sorted(updates) == [2, 4]
    _assert_same_state(load_checkpoint(path), load_checkpoint(reference))


def test_cadences_produce_exact_terminal_state_and_matched_draws(cheap_training, monkeypatch):
    saves = []
    original = execution._save_training_checkpoint

    def record(path, trainer, run, audit):
        saves.append(trainer.phase_updates)
        original(path, trainer, run, audit)

    monkeypatch.setattr(execution, "_save_training_checkpoint", record)
    states = []
    for run, cadence, expected in (
        ("every", "1", [0, 1, 2, 3, 4, 5]), ("period", "2", [0, 2, 4, 5]),
        ("large", "100", [0, 5]), ("terminal", "null", [0, 5]),
    ):
        saves.clear()
        path = execution.train_workflow(cheap_training(run, f"checkpoint.save_every_updates={cadence}"))
        assert saves == expected
        assert [item.name for item in path.parent.glob("*.pt")] == ["checkpoint.pt"]
        states.append(load_checkpoint(path))
    for state in states[1:]:
        _assert_same_state(state, states[0])


def test_pause_lost_update_replay_and_terminal_metadata_retry(cheap_training, monkeypatch):
    interval = "checkpoint.save_every_updates=2"
    reference = load_checkpoint(execution.train_workflow(cheap_training("reference", interval)))
    path = execution.train_workflow(cheap_training("pause", interval), stop_after=0)
    assert load_checkpoint(path)["counters"]["phase_updates"] == 0
    execution.train_workflow(cheap_training("pause", interval, resume=True), stop_after=3)
    assert load_checkpoint(path)["counters"]["phase_updates"] == 3
    with pytest.raises(ArtifactError, match="resolved configuration"):
        execution.train_workflow(cheap_training("pause", "checkpoint.save_every_updates=3", resume=True))

    original_step = training.Trainer.step
    def interrupted(trainer, objective):
        if trainer.phase_updates == 3:
            raise RuntimeError("interrupted after one unsaved update")
        return original_step(trainer, objective)

    with monkeypatch.context() as patch:
        patch.setattr(training.Trainer, "step", interrupted)
        with pytest.raises(RuntimeError, match="unsaved update"):
            execution.train_workflow(cheap_training("lost", interval))
    lost = path.parent.parent / "lost/checkpoint.pt"
    assert load_checkpoint(lost)["counters"]["phase_updates"] == 2
    execution.train_workflow(cheap_training("lost", interval, resume=True))
    _assert_same_state(load_checkpoint(lost), reference)

    write = execution.write_json
    def fail_final(path, record):
        if record.get("status") == "complete":
            raise OSError("final metadata failed")
        write(path, record)
    with monkeypatch.context() as patch:
        patch.setattr(execution, "write_json", fail_final)
        with pytest.raises(OSError, match="final metadata"):
            execution.train_workflow(cheap_training("pause", interval, resume=True))
    completed = path.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(training.Trainer, "step", lambda *_: pytest.fail("extra optimizer step"))
        execution.train_workflow(cheap_training("pause", interval, resume=True))
    assert path.read_bytes() == completed
    _assert_same_state(load_checkpoint(path), reference)


@pytest.mark.parametrize("failure", ["serialize", "flush", "replace", "parent-sync"])
def test_checkpoint_failure_boundary_preserves_valid_old_or_new_state(tmp_path, monkeypatch, failure):
    bundle, plan = setup_plan()
    trainer = scalar_trainer(bundle, plan, steps=2, adam=True)
    path = tmp_path / "checkpoint.pt"
    trainer.step(convex_loss)
    save_checkpoint(path, trainer.snapshot())
    older = path.read_bytes()
    trainer.step(convex_loss)

    def fail(*_):
        raise OSError("injected publication failure")
    with monkeypatch.context() as patch:
        module, attribute = {
            "serialize": (checkpoints.torch, "save"), "flush": (artifacts.os, "fsync"),
            "replace": (artifacts.os, "replace"), "parent-sync": (artifacts, "_sync_directory"),
        }[failure]
        patch.setattr(module, attribute, fail)
        with pytest.raises(OSError, match="publication failure"):
            save_checkpoint(path, trainer.snapshot())
    assert load_checkpoint(path)["counters"]["phase_updates"] == (2 if failure == "parent-sync" else 1)
    assert (path.read_bytes() == older) == (failure != "parent-sync")
    assert list(tmp_path.iterdir()) == [path]
    recovered = scalar_trainer(bundle, plan, steps=2, adam=True)
    recovered.restore(load_checkpoint(path))
    if not recovered.complete:
        recovered.step(convex_loss)
    assert_exact(recovered.snapshot(), trainer.snapshot())


def test_failed_step_has_no_snapshot_and_prior_source_resume_is_explicit(cheap_training):
    interval = "checkpoint.save_every_updates=2"
    path = execution.train_workflow(cheap_training("old", interval), stop_after=1)
    old = load_checkpoint(path)
    old["run"]["replay_environment"]["package_source_sha256"] = "a" * 64
    save_checkpoint(path, old)
    with pytest.raises(ArtifactError, match="original checkout"):
        execution.train_workflow(cheap_training("old", interval, resume=True))

    bundle, plan = setup_plan()
    trainer = scalar_trainer(bundle, plan, steps=2)
    durable = path.with_name("fixture.pt")
    save_checkpoint(durable, trainer.snapshot())
    old_bytes = durable.read_bytes()
    def bad(*_):
        raise RuntimeError("failed objective")
    with pytest.raises(RuntimeError, match="failed objective"):
        trainer.step(bad)
    with pytest.raises(RuntimeError, match="failed training action"):
        save_checkpoint(durable, trainer.snapshot())
    assert durable.read_bytes() == old_bytes


@pytest.mark.parametrize(("value", "expected"), [("100", 100), ("1", 1), ("null", None)])
def test_checkpoint_cadence_is_typed_and_round_trips(tmp_path, compose_settings, value, expected):
    config = resolve_config(compose_settings(f"checkpoint.save_every_updates={value}"), invocation_dir=tmp_path)
    actual = config.to_record()["checkpoint"]["save_every_updates"]
    assert actual is expected or (type(actual) is int and actual == expected)


@pytest.mark.parametrize("value", ["0", "-2", "true", "2.5", "'2'"])
def test_invalid_checkpoints_cadences_fail(tmp_path, compose_settings, value):
    with pytest.raises(ConfigError, match="checkpoint.save_every_updates"):
        resolve_config(compose_settings(f"checkpoint.save_every_updates={value}"), invocation_dir=tmp_path)


@pytest.mark.resource_process
@pytest.mark.parametrize("boundary", ["before", "after"])
def test_checkpoint_sigkill_at_replace_retains_valid_snapshot(tmp_path, boundary):
    import select
    import subprocess
    import sys

    bundle, plan = setup_plan()
    trainer = scalar_trainer(bundle, plan, steps=2, adam=True)
    path = tmp_path / "checkpoint.pt"
    trainer.step(convex_loss)
    save_checkpoint(path, trainer.snapshot())
    previous = path.read_bytes()
    trainer.step(convex_loss)
    incoming = tmp_path / "update.pt"
    save_checkpoint(incoming, trainer.snapshot())
    program = """
from pathlib import Path
import sys
from nn_window_ablation import artifacts
from nn_window_ablation.checkpoints import load_checkpoint, save_checkpoint
original = artifacts.os.replace
def wait():
    print('READY', flush=True)
    sys.stdin.buffer.read(1)
def barrier(source, destination):
    if sys.argv[3] == 'before':
        wait()
    original(source, destination)
    if sys.argv[3] == 'after':
        wait()
state = load_checkpoint(Path(sys.argv[1]))
artifacts.os.replace = barrier
save_checkpoint(Path(sys.argv[2]), state)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", program, str(incoming), str(path), boundary],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    try:
        ready, _, _ = select.select([child.stdout], [], [], 45)
        assert ready and child.stdout.readline().strip() == b"READY"
    finally:
        child.kill()
        child.communicate(timeout=15)
    restored = load_checkpoint(path)
    assert restored["counters"]["phase_updates"] == (1 if boundary == "before" else 2)
    if boundary == "before":
        assert previous == path.read_bytes()
    replay = scalar_trainer(bundle, plan, steps=2, adam=True)
    replay.restore(restored)
    if not replay.complete:
        replay.step(convex_loss)
    assert_exact(replay.snapshot(), trainer.snapshot())
