"""Check the global update contract and the production Gloo lifecycle."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nn_window_ablation import ddp_launch, execution
from nn_window_ablation.artifacts import ArtifactError, file_integrity, read_json
from nn_window_ablation.checkpoints import load_checkpoint, load_evaluation_model
from nn_window_ablation.config import resolve_config
from nn_window_ablation.data import SamplingPlan, sample_chunk, write_dataset
from nn_window_ablation.paired_workflows import evaluate_workflow
from nn_window_ablation.pbs import _verified_artifact
from nn_window_ablation.training import Trainer
from tests.ddp_checks import capture_gradients, capture_rank, failing_exit_rank, recovery_rank, timeout_rank, toy_rank
from tests.integration.test_parallel_evaluation import _reap_adopted_processes, _reap_group
from tests.p4_checks import assert_exact
from tests.parallel_evaluation_checks import await_file


pytestmark = pytest.mark.resource_process
STATE_RTOL, STATE_ATOL = 1e-4, 3e-5
GRAD_RTOL, GRAD_ATOL = 2e-4, 1e-7


def _settings(tmp_path, compose_settings, *, distance=5):
    def resolved(*overrides):
        return resolve_config(
            compose_settings("runtime=test_cpu", f"code.distance={distance}", *overrides), invocation_dir=tmp_path,
        )
    return resolved


def _stored_inputs(resolved, *, total=12, chunk=3, events_at=None, name="train", phase="full_closed"):
    generation = resolved(
        f"training={phase}", "data.split=train", f"data.id={name}", f"data.shots={total}", f"data.shard_shots={chunk}",
    )
    if events_at is None:
        return execution.generate_workflow(generation)
    bundle, namespace = execution._circuit_and_namespace(generation)
    plan = SamplingPlan.from_circuit(bundle, namespace, chunk_rows=chunk, total_rows=total)
    count = (total + chunk - 1) // chunk
    order = []
    for index in range(count):
        _, rows = plan.chunk_bounds(index)
        rng = np.random.Generator(np.random.PCG64(namespace.seed("trajectory_order", index)))
        order.extend((index, int(row)) for row in rng.permutation(rows))
    event_ids = {order[position] for position in events_at}

    def chunks():
        for index in range(count):
            shots = sample_chunk(plan, bundle, index)
            for row in range(len(shots.shot_ids)):
                if (index, row) in event_ids:
                    assert shots.detectors[row].any()
                else:
                    shots.detectors[row] = shots.core_labels[row] = shots.global_labels[row] = False
            yield shots

    target = generation.paths.data_root / name
    target.parent.mkdir(exist_ok=True)
    write_dataset(target, plan, bundle, provenance={"fixture": "specified event-empty ordered rows"}, chunks=chunks())
    return target / "manifest.json"


def _options(name, world, *, steps=3, batch=4, shard_source="train", cadence=2):
    return (
        f"run_name={name}", "data.split=train", "data.train_source=stored_shards", f"data.id={shard_source}",
        f"training.max_steps={steps}", f"training.batch_size={batch}", "training.microbatch_size=1",
        "training.optimizer=adam", "training.learning_rate=0.001", "training.scheduler=step",
        "training.scheduler_step_size=1", "training.scheduler_gamma=0.75", f"checkpoint.save_every_updates={cadence}",
        "training.execution=single" if world == 0 else "training.execution=ddp",
        f"runtime.processes={max(1, world)}", f"runtime.allocated_cpus={max(1, world)}",
        "training.ddp_timeout_seconds=20",
    )


def _numerically_equal(expected, actual):
    for key, value in expected["weights"].items():
        torch.testing.assert_close(actual["weights"][key], value, atol=STATE_ATOL, rtol=STATE_RTOL, msg=key)
    assert expected["scheduler"] == actual["scheduler"]
    first, second = expected["optimizer"], actual["optimizer"]
    assert first["param_groups"] == second["param_groups"]
    assert first["state"].keys() == second["state"].keys()
    for key, left in first["state"].items():
        right = second["state"][key]
        assert_exact(left["step"], right["step"])
        torch.testing.assert_close(right["exp_avg"], left["exp_avg"], rtol=2e-4, atol=2e-7)
        torch.testing.assert_close(right["exp_avg_sq"], left["exp_avg_sq"], rtol=2e-4, atol=2e-9)
    for key, first_grad in expected["gradients"].items():
        second_grad = actual["gradients"][key]
        assert (first_grad is None) == (second_grad is None), key
        if first_grad is not None:
            torch.testing.assert_close(second_grad, first_grad, rtol=GRAD_RTOL, atol=GRAD_ATOL, msg=key)


def test_gloo_scalar_global_mean_with_partial_local_microbatch(tmp_path, compose_settings, monkeypatch):
    resolved = _settings(tmp_path, compose_settings, distance=3)
    _stored_inputs(resolved, total=6, chunk=4)
    config = resolved(*_options("toy", 2, steps=1, batch=6), "training.microbatch_size=2")
    config.paths.run_root.mkdir()
    monkeypatch.setattr(ddp_launch, "_rank_train", toy_rank)
    ddp_launch.launch_training(config)
    ranks = [read_json(config.paths.run_root / f"rank{index}.json") for index in range(2)]
    assert ranks[0] == ranks[1]
    assert ranks[0]["microbatches"] == [2, 1]
    gradient = -2.0 - (1 + 4 + 9) / 3
    assert ranks[0]["gradient"] == pytest.approx(gradient, abs=1e-14)
    assert ranks[0]["theta"] == pytest.approx(-2.0 - 0.125 * gradient, abs=1e-14)
    assert ranks[0]["lr"] == 0.0625 and ranks[0]["scheduler"]["last_epoch"] == 1
    assert ranks[0]["counters"]["trajectories"] == 6 and ranks[0]["counters"]["query_examples"] == 12


def test_real_d5_gradients_unused_adam_and_world_size_one(tmp_path, compose_settings, monkeypatch):
    resolved = _settings(tmp_path, compose_settings)
    # Update 1: only rank 0's earlier microbatch uses the GNN. Update 2:
    # globally empty. Update 3: only rank 1's synchronized microbatch uses it.
    dataset_path = _stored_inputs(resolved, events_at={0, 11})
    dirs = [tmp_path / "runs" / name for name in ("serial", "one", "two")]
    with monkeypatch.context() as patch:
        patch.setattr(Trainer, "_optimize", capture_gradients(dirs[0], 0))
        paths = [execution.train_workflow(resolved(*_options("serial", 0)))]
    monkeypatch.setattr(ddp_launch, "_rank_train", capture_rank)
    for world, name in enumerate(("one", "two"), 1):
        paths.append(execution.train_workflow(resolved(*_options(name, world))))
    states = list(map(load_checkpoint, paths))
    model = load_evaluation_model(paths[-1])
    assert model.rnn.hidden_size == 256 and model.rnn.num_layers == 4
    for state in states[1:]:
        assert state["training_inputs"] == states[0]["training_inputs"]
        assert state["cursor"] == states[0]["cursor"]
        assert state["counters"] == {
            "parent_updates": 0, "phase_updates": 3, "cumulative_updates": 3, "trajectories": 12, "query_examples": 24,
        }
    prefix = states[0]["training_inputs"]["trace_prefix"]
    assert len({(item["shard"], item["row"]) for step in prefix for item in step["trajectories"]}) == 12
    assert len({item["shard"] for item in prefix[0]["trajectories"]}) == 2
    assert states[2]["contract"]["source"]["dataset_manifest_sha256"] == file_integrity(dataset_path)["sha256"]
    conv_id = None
    prior = None
    for update in (1, 2, 3):
        serial, one, two = [torch.load(path / f"check-rank0-update{update}.pt", weights_only=True) for path in dirs]
        peer = torch.load(dirs[-1] / f"check-rank1-update{update}.pt", weights_only=True)
        assert_exact(one, serial)
        assert_exact(peer, two)
        _numerically_equal(serial, two)
        conv_name = next(name for name in serial["gradients"] if name.startswith("gnn.convs"))
        if conv_id is None:
            conv_id = list(serial["gradients"]).index(conv_name)
        if update == 2:
            assert two["gradients"][conv_name] is None
            assert_exact(two["optimizer"]["state"][conv_id], prior["optimizer"]["state"][conv_id])
            assert_exact(two["weights"][conv_name], prior["weights"][conv_name])
        else:
            assert two["gradients"][conv_name] is not None and two["gradients"][conv_name].abs().max() > 0
            assert int(two["optimizer"]["state"][conv_id]["step"]) == (1 if update == 1 else 2)
        prior = two
    receipt, runtime = _verified_artifact(SimpleNamespace(stage="train"), resolved(*_options("two", 2)), paths[-1])
    assert receipt["sha256"] == file_integrity(paths[-1])["sha256"] and runtime["world_size"] == 2
    execution.generate_workflow(resolved("data.split=test", "data.id=held", "data.shots=3", "data.shard_shots=2"))
    for count in (1, 2):
        manifest = evaluate_workflow(resolved(
            "data.split=test", "data.id=held", "evaluation=paired", f"run_name=eval{count}",
            f"evaluation.workers={count}", f"checkpoint.path={paths[-1]}",
        ))
        if count == 1:
            reference = read_json(manifest)
        else:
            assert read_json(manifest)["shards"] == reference["shards"]


def _resume(resolved, name, world=2):
    return resolved(*_options(name, world, steps=4), f"checkpoint.resume_from=runs/{name}/checkpoint.pt")


def _exact_replay(expected, actual):
    for key in ("model", "optimizer_state", "scheduler_state", "random_states", "distributed_state", "training_inputs",
                "cursor", "counters", "last_update"):
        assert_exact(actual[key], expected[key])


def test_same_topology_pause_and_abrupt_rank_death_replay(tmp_path, compose_settings, monkeypatch):
    resolved = _settings(tmp_path, compose_settings)
    _stored_inputs(resolved, total=16)
    monkeypatch.setattr(ddp_launch, "_rank_train", recovery_rank)
    reference = load_checkpoint(execution.train_workflow(resolved(*_options("reference", 2, steps=4))))
    states = reference["distributed_state"]["rank_rng"]
    assert not torch.equal(states[0]["torch_cpu"], states[1]["torch_cpu"])
    assert states[0]["python"] != states[1]["python"]
    pause = execution.train_workflow(resolved(*_options("pause", 2, steps=4)), stop_after=1)
    paused_receipt = file_integrity(pause)
    assert read_json(pause.with_name("run.json"))["status"] == "paused"
    with pytest.raises(ArtifactError, match="rank .* exited"):
        execution.train_workflow(_resume(resolved, "pause", world=1))
    assert file_integrity(pause) == paused_receipt
    _exact_replay(reference, load_checkpoint(execution.train_workflow(_resume(resolved, "pause"))))

    crashed = tmp_path / "runs/crashed/checkpoint.pt"
    repo = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(repo), "NNWA_DDP_TEST_FAULT": "pause"}
    with _reap_adopted_processes(), (tmp_path / "crash.log").open("w+") as output:
        child = subprocess.Popen(
            [sys.executable, "-m", "tests.ddp_checks", "train", "runtime=test_cpu", "code.distance=5",
             *_options("crashed", 2, steps=4)],
            cwd=tmp_path, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            ready = tmp_path / "rank1-ready"
            await_file(ready, timeout=60)
            receipt = file_integrity(crashed)
            assert load_checkpoint(crashed)["counters"]["phase_updates"] == 2
            os.kill(int(ready.read_text()), signal.SIGKILL)
            assert child.wait(timeout=15) == 1
            output.seek(0)
            assert "rank 1" in output.read()
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=15)
            _reap_group(child.pid)
    assert file_integrity(crashed) == receipt and read_json(crashed.with_name("run.json"))["status"] != "complete"
    _exact_replay(reference, load_checkpoint(execution.train_workflow(_resume(resolved, "crashed"))))


def test_rank_exception_leaves_initial_checkpoint_and_collective_timeout_is_bounded(tmp_path, compose_settings, monkeypatch):
    resolved = _settings(tmp_path, compose_settings, distance=3)
    _stored_inputs(resolved, total=4)
    monkeypatch.setattr(ddp_launch, "_rank_train", recovery_rank)
    monkeypatch.setenv("NNWA_DDP_TEST_FAULT", "exception")
    with pytest.raises(ArtifactError, match="rank .* exited"):
        execution.train_workflow(resolved(*_options("failed", 2, steps=1)))
    path = tmp_path / "runs/failed/checkpoint.pt"
    assert load_checkpoint(path)["counters"]["phase_updates"] == 0
    assert read_json(path.with_name("run.json"))["status"] != "complete"
    monkeypatch.setattr(ddp_launch, "_rank_train", timeout_rank)
    started = monotonic()
    with pytest.raises(ArtifactError, match="rank 0 .* exited"):
        ddp_launch.launch_training(resolved(*_options("timeout", 2, steps=1), "training.ddp_timeout_seconds=3"))
    assert (tmp_path / "runs/timeout-ready").exists()
    assert monotonic() - started < 25


def test_finite_source_and_peer_exit_gate(tmp_path, compose_settings, monkeypatch):
    resolved = _settings(tmp_path, compose_settings, distance=3)
    dataset = _stored_inputs(resolved, total=4, chunk=2)
    with pytest.raises(ArtifactError, match="rank .* exited"):
        execution.train_workflow(resolved(*_options("too-large", 2, steps=2)))
    assert not (tmp_path / "runs/too-large").exists()

    with monkeypatch.context() as patch:
        patch.setattr(ddp_launch, "_rank_train", failing_exit_rank)
        with pytest.raises(ArtifactError, match="rank 1 .* code 17"):
            execution.train_workflow(resolved(*_options("late-failure", 2, steps=1)))
    run = tmp_path / "runs/late-failure"
    assert read_json(run / "run.json")["status"] == "running"
    assert load_checkpoint(run / "checkpoint.pt")["counters"]["phase_updates"] == 1

    payload = dataset.parent / "shards/1/detectors.npy"
    raw = payload.read_bytes()
    payload.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ArtifactError, match="rank .* exited"):
        execution.train_workflow(resolved(*_options("source-failure", 2, steps=1)))
    run = tmp_path / "runs/source-failure"
    assert load_checkpoint(run / "checkpoint.pt")["counters"]["phase_updates"] == 0
    assert read_json(run / "run.json")["status"] == "running"


def test_ddp_terminal_parent_and_matched_continuations_with_validation(tmp_path, compose_settings):
    resolved = _settings(tmp_path, compose_settings)
    _stored_inputs(resolved, total=4)
    base = execution.train_workflow(resolved(*_options("base", 2, steps=1)))
    _stored_inputs(resolved, total=4, name="new", phase="continued_full")
    execution.generate_workflow(resolved(
        "data.split=validation", "data.id=validation", "data.shots=1", "data.shard_shots=1",
    ))
    checkpoints = []
    for preset in ("continued_full", "fixed_window", "dynamic_window"):
        config = resolved(
            f"training={preset}", *_options(preset, 2, steps=1, shard_source="new"),
            f"checkpoint.init_from={base}", "training.validation_data_id=validation",
        )
        path = execution.train_workflow(config)
        checkpoint = load_checkpoint(path)
        log_dir, = (path.parent / "train-attempts").iterdir()
        for rank in (0, 1):
            events = [event["event"] for event in map(
                json.loads, (log_dir / f"rank-{rank}.jsonl").read_text().splitlines(),
            )]
            assert "group_closed" in events
            if rank == 0:
                assert events.index("group_closed") < events.index("terminal_validation")
            else:
                assert "terminal_validation" not in events
        checkpoints.append(checkpoint)
    audits = [state["continuation_audit"] for state in checkpoints]
    assert len({audit["input_sha256"] for audit in audits}) == 1
    assert len({audit["controls_sha256"] for audit in audits}) == 1
    assert all(audit["controls"]["parent_sha256"] == file_integrity(base)["sha256"] for audit in audits)
    assert all(audit["validation"]["shots"] == 1 for audit in audits)
    original_inputs = checkpoints[0]["training_inputs"]["trace_prefix"][0]
    for state in checkpoints:
        trace = state["training_inputs"]["trace_prefix"][0]
        assert trace["trajectories"] == original_inputs["trajectories"] and trace["cores"] == original_inputs["cores"]
        assert state["counters"] == {
            "parent_updates": 1, "phase_updates": 1, "cumulative_updates": 2, "trajectories": 4, "query_examples": 8,
        }
        assert {int(value["step"]) for value in state["optimizer_state"]["state"].values()} == {1}
    assert [state["last_update"]["buffers"] for state in checkpoints[:2]] == [None, None]
    plan = SamplingPlan.from_record(checkpoints[-1]["contract"]["source"]["plan"])
    rng = np.random.Generator(np.random.PCG64(plan.namespace.seed("dynamic_buffer")))
    expected = [int((0, 3, 5)[index]) for index in rng.integers(3, size=2)]
    assert checkpoints[-1]["last_update"]["buffers"] == dict(zip(("left", "right"), expected, strict=True))
    serial = execution.train_workflow(resolved(
        "training=dynamic_window", *_options("serial-dynamic", 0, steps=1, shard_source="new"),
        f"checkpoint.init_from={base}",
    ))
    assert load_checkpoint(serial)["training_inputs"] == checkpoints[-1]["training_inputs"]
