"""Exercise real spawned inference, out-of-order commits and worker loss."""

from contextlib import contextmanager
import ctypes
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import monotonic, sleep
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nn_window_ablation import paired_workflows as workflows
from nn_window_ablation.artifacts import ArtifactError, file_integrity, read_json
from nn_window_ablation.config import resolve_config
from nn_window_ablation.data import open_dataset, sample_chunk, write_dataset
from nn_window_ablation.execution import generate_workflow, train_workflow
from nn_window_ablation.export import export_bundle, reanalyze_bundle
from nn_window_ablation.archive import extract_bundle
from nn_window_ablation.pbs import _verified_artifact
from tests.p4_checks import setup_plan
from tests.parallel_evaluation_checks import actual_model_loads, await_file, controlled_initialize, controlled_task


pytestmark = pytest.mark.resource_process


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _log(root, pattern="coordinator.jsonl"):
    return [event for path in root.glob(f"attempts/*/{pattern}") for event in _lines(path)]


def _events(records, name):
    return [event for event in records if event["event"] == name]


def _inputs(tmp_path, compose_settings, *, genuine):
    distance = 5 if genuine else 3
    noise = 0.003 if genuine else 0
    preset = ("runtime=test_cpu", f"code.distance={distance}", f"noise.probability={noise}")
    common = ("data.id=held", "data.split=test", "checkpoint.path=runs/base/checkpoint.pt", "evaluation=paired")

    def resolve(*overrides):
        return resolve_config(compose_settings(*preset, *overrides), invocation_dir=tmp_path)

    train = generate_workflow(resolve("data.id=training", "data.split=train", "data.shots=1", "data.shard_shots=1"))
    assert train.is_file()
    train_workflow(resolve(
        "run_name=base", "data.split=train", "data.train_source=stored_shards", "data.id=training",
        "training.max_steps=1", "training.batch_size=1", "training.optimizer=sgd", "training.scheduler=none",
        "training.learning_rate=0.001",
    ))
    held = tmp_path / "data/held"
    if genuine:
        bundle, plan = setup_plan(split="test", total=9, chunk=2, distance=5, rounds=60)
        assert plan.spec.probability == noise

        def chunks():
            for index in range(5):
                shots = sample_chunk(plan, bundle, index)
                if index == 0:
                    shots.detectors[0] = shots.core_labels[0] = shots.global_labels[0] = False
                yield shots

        write_dataset(held, plan, bundle, chunks=chunks(), provenance={"fixture": "d5 with one explicit zero-event shot"})
    else:
        generate_workflow(resolve("data.id=held", "data.split=test", "data.shots=7", "data.shard_shots=1"))

    def args(name, workers, *extras, resume=False):
        return (
            f"run_name={name}", f"evaluation.workers={workers}", f"evaluation.resume={str(resume).lower()}",
            *common, *extras,
        )

    return resolve, preset, args, held


@pytest.fixture
def controlled(tmp_path, compose_settings, monkeypatch):
    inputs = _inputs(tmp_path, compose_settings, genuine=False)
    monkeypatch.setattr(workflows, "_initialize_evaluation_worker", controlled_initialize)
    monkeypatch.setattr(workflows, "_run_worker_task", controlled_task)
    return inputs


def _report(resolve, path, name):
    return workflows.analyze_workflow(resolve(
        f"run_name={name}", "data.id=held", "data.split=test", f"analysis.predictions=[{path}]",
        "analysis.bootstrap_seed=3", "analysis.bootstrap_replicates=19", "analysis.sparse_rule=1",
    ))


def test_saved_d5_serial_two_and_four_workers_are_identical(tmp_path, compose_settings, monkeypatch):
    resolve, _, args, held = _inputs(tmp_path, compose_settings, genuine=True)
    monkeypatch.setattr(workflows, "_initialize_evaluation_worker", actual_model_loads)
    torch.use_deterministic_algorithms(True)
    try:
        manifests = [
            workflows.evaluate_workflow(resolve(*args(
                f"eval{count}", count, "evaluation.matching=true", "evaluation.batch_size=2",
                f"runtime.allocated_cpus={max(1, count)}",
            ))) for count in (1, 2, 4)
        ]
    finally:
        torch.use_deterministic_algorithms(False)
    records = [read_json(path) for path in manifests]
    assert [entry["rows"] for entry in records[0]["shards"]] == [2, 2, 2, 2, 1]
    assert all(record["contract"] == records[0]["contract"] for record in records)
    assert all(record["shards"] == records[0]["shards"] for record in records)
    mapped = open_dataset(held)
    failures = {mode: [] for mode in ("G", "F", "S", "MWPM")}
    for entry in records[0]["shards"]:
        index = entry["index"]
        paths = [path.parent / "shards" / str(index) for path in manifests]
        with mapped.open_shard(index) as source:
            expected_ids = source.shot_ids
            labels, core = source.read_labels()
            if index == 0:
                assert not source.read(slice(0, 1)).shots.detectors.any()
        for name in entry["files"]:
            assert all((path / name).read_bytes() == (paths[0] / name).read_bytes() for path in paths)
        np.testing.assert_array_equal(np.load(paths[0] / "shot_ids.npy"), expected_ids)
        for mode in failures:
            width = 12 if mode in ("F", "S") else 1
            bits = np.unpackbits(np.load(paths[0] / f"{mode}.decisions.npy"), axis=1, count=width, bitorder="little").astype(bool)
            if mode != "MWPM":
                np.testing.assert_array_equal(bits, np.load(paths[0] / f"{mode}.logits.npy") >= 0)
            final = bits[:, 0] if width == 1 else np.bitwise_xor.reduce(bits, axis=1)
            if width == 12:
                np.testing.assert_array_equal(final, np.load(paths[0] / f"{mode}.final.npy")[:, 0])
                np.testing.assert_array_equal(final ^ labels, np.bitwise_xor.reduce(bits ^ core, axis=1))
            failures[mode].extend(final ^ labels)

    reports = [_report(resolve, path, f"stats{index}") for index, path in enumerate(manifests)]
    expected = read_json(reports[0])
    failures = {mode: np.array(rows) for mode, rows in failures.items()}
    for mode, rows in failures.items():
        assert expected["rates"][mode]["failures"] == np.count_nonzero(rows)
    for row in expected["contrasts"].values():
        first, second = failures[row["first"]], failures[row["second"]]
        assert (row["n10"], row["n01"]) == (np.count_nonzero(first & ~second), np.count_nonzero(second & ~first))
    for report in reports[1:]:
        for name in ("rates", "contrasts", "primary_decomposition"):
            assert read_json(report)[name] == expected[name]

    for workers, path in zip((2, 4), manifests[1:], strict=True):
        records = _log(path.parent, "worker-*.jsonl")
        initialized = _events(records, "worker_ready")
        loads = _events(records, "actual_model_load")
        assert len(initialized) == len(loads) == workers
        assert {item["pid"] for item in initialized} == {item["pid"] for item in _events(records, "worker_exit")}
        assert len({tuple(item["process"]["affinity"]) for item in initialized}) == workers
        tasks = _events(records, "task_done")
        assert len(tasks) == 5
        assert len({task["pid"] for task in tasks}) < len(tasks), "at least one persistent worker must process two shards"
        for item in initialized:
            assert item["runtime"]["torch_intraop"] == item["runtime"]["torch_interop"] == 1
            assert set(item["native_env"].values()) == {"1"}
            assert len(item["process"]["affinity"]) == 1
        for task in tasks:
            assert task["active_mappings"] == 4
            before, after = task["before"]["fd_count"], task["after"]["fd_count"]
            assert task["active_fd_count"] == before + 4 and before == after
            assert not task["dataset_active_after"]
        assert not (path.parent / ".evaluation-staging").exists()
    config = resolve(*args("eval4", 4, "evaluation.matching=true", "evaluation.batch_size=2"))
    receipt, _ = _verified_artifact(SimpleNamespace(stage="evaluate"), config, manifests[-1])
    assert receipt["sha256"] == file_integrity(manifests[-1])["sha256"]
    archive = export_bundle(held, reports[-1], tmp_path / "compact.tar.gz")
    imported = extract_bundle(archive, tmp_path / "portable")
    (tmp_path / "data").rename(tmp_path / "hidden-data")
    (tmp_path / "runs").rename(tmp_path / "hidden-runs")
    replay = read_json(reanalyze_bundle(imported, tmp_path / "replayed"))
    assert replay["contrasts"] == expected["contrasts"] and replay["rates"] == expected["rates"]


def test_replenishes_after_later_shard_commit_while_first_worker_is_blocked(tmp_path, controlled, monkeypatch):
    resolve, _, args, _ = controlled
    root = tmp_path / "runs/candidate"
    monkeypatch.setenv("NNWA_TEST_BEHAVIOR", "reorder")
    result = workflows.evaluate_workflow(resolve(*args("candidate", 2, "evaluation.max_in_flight=2")))
    assert (root / "test-barriers/started-2").exists()
    log = _log(root)
    commits = [event["index"] for event in _events(log, "task_committed")]
    assert commits[0] == 1 and sorted(commits) == list(range(7))
    assert [entry["index"] for entry in read_json(result)["shards"]] == list(range(7))
    submitted = _events(log, "task_submitted")
    assert max(event["pending"] for event in submitted) == 2
    first_commit = next(i for i, event in enumerate(log) if event["event"] == "task_committed")
    assert [event["index"] for event in _events(log[:first_commit], "task_submitted")] == [0, 1]
    assert all(event["task_bytes"] < 512 and event["result_bytes"] < 1024 for event in _events(_log(root, "worker-*.jsonl"), "ipc"))


@contextmanager
def _reap_adopted_processes():
    # Reap the deliberately orphaned workers and resource tracker even under a
    # test host whose PID 1 does not reap promptly. Restore the xdist process.
    libc = ctypes.CDLL(None, use_errno=True)
    prior = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(prior), 0, 0, 0) == 0  # PR_GET_CHILD_SUBREAPER
    assert libc.prctl(36, 1, 0, 0, 0) == 0  # PR_SET_CHILD_SUBREAPER
    try:
        yield
    finally:
        assert libc.prctl(36, prior.value, 0, 0, 0) == 0


def _reap_group(session_pid):
    deadline = monotonic() + 15
    while monotonic() < deadline:
        try:
            while os.waitpid(-session_pid, os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            pass
        try:
            os.killpg(session_pid, 0)
        except ProcessLookupError:
            return
        sleep(0.02)
    os.killpg(session_pid, signal.SIGKILL)
    pytest.fail(f"owned evaluation process group {session_pid} did not exit")


@pytest.mark.parametrize("interrupt", ["worker-kill", "coordinator-term", "coordinator-kill"])
def test_interruption_preserves_noncontiguous_commit_and_resumes(tmp_path, controlled, interrupt):
    resolve, preset, args, held = controlled
    root = tmp_path / "runs/candidate"
    barrier = root / "test-barriers"
    program = Path(__file__).resolve().parents[2]
    env = {**os.environ, "NNWA_TEST_BEHAVIOR": "pause", "PYTHONPATH": str(program)}
    # The guarded spawnable CLI pauses its worker after a partial staged write.
    with _reap_adopted_processes(), (tmp_path / "driver.log").open("w+") as output:
        child = subprocess.Popen(
            [sys.executable, "-m", "tests.parallel_evaluation_checks", "evaluate", *preset,
             *args("candidate", 2, "evaluation.max_in_flight=2")],
            cwd=tmp_path, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            await_file(barrier / "started-2")
            committed = root / "shards/1/receipt.json"
            intact = file_integrity(committed)
            assert {path.name for path in (root / "shards").iterdir()} == {"1"}
            pid = int((barrier / "ready-0").read_text())
            assert len(list(root.glob(".evaluation-staging/*/0/shot_ids.npy"))) == 1
            with pytest.raises(ArtifactError, match="another process"):
                workflows.evaluate_workflow(resolve(*args("candidate", 3, resume=True)))
            action = signal.SIGTERM if interrupt == "coordinator-term" else signal.SIGKILL
            os.kill(pid if interrupt == "worker-kill" else child.pid, action)
            assert child.wait(timeout=15) != 0
            output.seek(0)
            stdout = output.read()
            if interrupt == "worker-kill":
                assert "EvaluationTask(index=" in stdout and "terminated abruptly" in stdout
            if interrupt == "coordinator-term":
                assert "interrupted by SIGTERM" in stdout
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=15)
            _reap_group(child.pid)
    assert not (root / "manifest.json").exists()
    with pytest.raises(ArtifactError, match="manifest"):
        _verified_artifact(SimpleNamespace(stage="evaluate"), resolve(*args("candidate", 2)), root / "manifest.json")
    with pytest.raises(ArtifactError, match="manifest"):
        _report(resolve, root / "manifest.json", "incomplete")
    if interrupt == "coordinator-kill":
        stage = next(root.glob(".evaluation-staging/*"))
        with (stage / "writers.lock").open("rb") as lease:
            fcntl.flock(lease, fcntl.LOCK_SH)
            with pytest.raises(ArtifactError, match="still has live writers"):
                workflows.evaluate_workflow(resolve(*args("candidate", 3, resume=True)))
            assert (stage / "0/shot_ids.npy").exists()
    else:
        assert not (root / ".evaluation-staging").exists()
    result = workflows.evaluate_workflow(resolve(*args("candidate", 3, "runtime.allocated_cpus=3", resume=True)))
    assert file_integrity(committed) == intact
    assert not (root / ".evaluation-staging").exists()
    assert sorted(event["index"] for event in _events(_log(root), "task_committed")) == list(range(7))
    last_attempt = next(path for path in root.glob("attempts/*/coordinator.jsonl") if _lines(path)[0]["workers"] == 3)
    assert [event["index"] for event in _events(_lines(last_attempt), "task_submitted")] == [0, 2, 3, 4, 5, 6]
    verified, _ = _verified_artifact(SimpleNamespace(stage="evaluate"), resolve(*args("candidate", 3)), result)
    assert verified["sha256"] == file_integrity(result)["sha256"]
    report = _report(resolve, result, "resumed-report")
    assert read_json(report)["rates"]["S"]["shots"] == 7
    if interrupt == "coordinator-kill":
        archive = export_bundle(held, report, tmp_path / "resumed.tar.gz")
        imported = extract_bundle(archive, tmp_path / "restored")
        assert read_json(reanalyze_bundle(imported, tmp_path / "resumed-replay"))["rates"] == read_json(report)["rates"]
    original_manifest = file_integrity(result)
    assert workflows.evaluate_workflow(resolve(*args("candidate", 1, resume=True))) == result
    assert file_integrity(result) == original_manifest
    validated = [path for path in root.glob("attempts/*/coordinator.jsonl") if _lines(path)[0]["workers"] == 1]
    assert len(validated) == 1
    assert not _events(_lines(validated[0]), "task_submitted")
    assert _lines(validated[0])[-1]["status"] == "validated_complete"


@pytest.mark.parametrize("boundary", ["before", "after"])
def test_coordinator_commit_fault_preserves_atomically_promoted_output(tmp_path, controlled, monkeypatch, boundary):
    resolve, _, args, _ = controlled
    original = workflows.publish_directory
    index = None

    def fail(stage, destination):
        nonlocal index
        index = int(destination.name)
        if boundary == "after":
            original(stage, destination)
        raise OSError(f"{boundary} promotion")

    with monkeypatch.context() as patch:
        patch.setattr(workflows, "publish_directory", fail)
        with pytest.raises(OSError, match=f"{boundary} promotion"):
            workflows.evaluate_workflow(resolve(*args("candidate", 2)))
    root = tmp_path / "runs/candidate"
    committed = root / "shards" / str(index) / "receipt.json"
    before = file_integrity(committed) if boundary == "after" else None
    assert committed.exists() == (boundary == "after")
    result = workflows.evaluate_workflow(resolve(*args("candidate", 3, resume=True)))
    assert len(read_json(result)["shards"]) == 7
    if before is not None:
        assert file_integrity(committed) == before
        resumed = [event for event in _events(_log(root), "task_submitted") if event["index"] == index]
        assert len(resumed) == 1


@pytest.mark.parametrize(("behavior", "message"), [
    ("wrong-task", "invalid receipt or duplicate"),
    ("wrong-contract", "invalid receipt or duplicate"),
    ("wrong-receipt", "receipt differs"),
    ("wrong-payload", "size or checksum"),
    ("write-error", "evaluation task .*injected write failure"),
    ("mutate", "model parameters or buffers"),
])
def test_spawned_evaluation_rejects_bad_receipts_and_task_failures(tmp_path, controlled, monkeypatch, behavior, message):
    resolve, _, args, _ = controlled
    monkeypatch.setenv("NNWA_TEST_BEHAVIOR", behavior)
    root = tmp_path / "runs/failed"
    with pytest.raises(ArtifactError, match=message):
        workflows.evaluate_workflow(resolve(*args("failed", 2, "evaluation.max_in_flight=1")))
    assert not (root / "manifest.json").exists()
    assert list((root / "shards").iterdir()) == []
    assert not (root / ".evaluation-staging").exists()
    if behavior in {"write-error", "mutate"}:
        events = _log(root, "worker-*.jsonl")
        failed, = _events(events, "task_failed")
        started, = _events(events, "task_start")
        assert not failed["dataset_active"]
        assert failed["process"]["fd_count"] == started["process"]["fd_count"]
