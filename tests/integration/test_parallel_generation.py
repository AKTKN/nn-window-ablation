"""Run the Linux pool on real stored Stim shots and partial-write barriers."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from nn_window_ablation import generation
from nn_window_ablation.artifacts import ArtifactError, read_json
from nn_window_ablation.config import resolve_config
from nn_window_ablation.data import open_dataset
from nn_window_ablation.execution import generate_workflow
from nn_window_ablation.pbs import _verified_artifact
from tests.generation_checks import audited_initializer, controlled_generation
from tests.integration.test_parallel_evaluation import _reap_adopted_processes, _reap_group
from tests.parallel_evaluation_checks import await_file


pytestmark = pytest.mark.resource_process


@pytest.fixture
def settings(tmp_path, compose_settings):
    def args(name, workers, *extra, total=17, shard=2):
        return (
            "runtime=test_cpu", f"runtime.allocated_cpus={workers}", f"data.sampling_processes={workers}",
            f"data.id={name}", f"data.shots={total}", f"data.shard_shots={shard}", "data.split=test", *extra,
        )

    def resolve(*values):
        return resolve_config(compose_settings(*values), invocation_dir=tmp_path)

    return args, resolve


def _events(root, name, pattern="coordinator.jsonl"):
    return [
        json.loads(line) for path in root.glob(f".generation-attempts/{name}/*/{pattern}")
        for line in path.read_text().splitlines()
    ]


def _assert_same(paths):
    records = [read_json(path) for path in paths]
    assert all(generation.dataset_semantics(record) == generation.dataset_semantics(records[0]) for record in records)
    assert len({path.read_bytes() for path in paths}) == len(paths)
    files = [path.relative_to(paths[0].parent) for path in paths[0].parent.rglob("*") if path.is_file() and path.name != "manifest.json"]
    for relative in files:
        assert all((path.parent / relative).read_bytes() == (paths[0].parent / relative).read_bytes() for path in paths)
    for path in paths:
        mapped = open_dataset(path.parent)
        ids = []
        for entry in mapped.shards:
            with mapped.open_shard(entry.index) as shard:
                shots = shard.read().shots
                np.testing.assert_array_equal(np.logical_xor.reduce(shots.core_labels, axis=1), shots.global_labels)
                ids.extend(map(int, shots.shot_ids))
        assert ids == list(range(mapped.plan.shot_prefix, mapped.plan.shot_prefix + mapped.plan.total_rows))


def test_distance_five_serial_and_two_four_workers_are_byte_identical(tmp_path, settings, monkeypatch):
    args, resolve = settings
    monkeypatch.setattr(generation, "_initialize_sampler", audited_initializer)
    outputs = [generate_workflow(resolve(*args(f"layout{count}", count))) for count in (1, 2, 4)]
    _assert_same(outputs)
    mapped = open_dataset(outputs[0].parent)
    assert (mapped.plan.spec.distance, mapped.plan.spec.rounds, mapped.plan.spec.probability) == (5, 60, 0.003)
    assert [shard.rows for shard in mapped.shards] == [2] * 8 + [1]
    config = resolve(*args("layout4", 4))
    _, observed = _verified_artifact(SimpleNamespace(stage="generate"), config, outputs[-1])
    assert observed["sampling_threads"] == 1 and observed["sampling_processes"] == 4
    for count in (2, 4):
        events = _events(tmp_path / "data", f"layout{count}", "worker-*.jsonl")
        built = [event for event in events if event["event"] == "worker_ready"]
        assert len(built) == count and all(event["real_circuit_builds"] == 1 for event in built)
        assert all(len(event["process"]["affinity"]) == 1 for event in built)
        assert {event["pid"] for event in built} == {event["pid"] for event in events if event["event"] == "worker_exit"}
        work = [event for event in events if event["event"] == "task_done"]
        assert len(work) == 9 and max(sum(event["pid"] == w["pid"] for w in work) for event in built) > 1
        for pid in {event["pid"] for event in work}:
            assert len({event["process"]["fd_count"] for event in work if event["pid"] == pid}) == 1


@pytest.mark.parametrize(("total", "shard"), [(3, 2), (1, 2)])
def test_pool_with_more_workers_than_shards_preserves_zero_events(tmp_path, settings, total, shard):
    args, resolve = settings
    common = ("noise.probability=0", "data.split=validation")
    path1 = generate_workflow(resolve(*args("reference", 1, *common, total=total, shard=shard)))
    path4 = generate_workflow(resolve(*args("small", 4, *common, total=total, shard=shard)))
    _assert_same([path1, path4])
    mapped = open_dataset(path4.parent)
    for desc in mapped.shards:
        with mapped.open_shard(desc.index) as handle:
            shots = handle.read().shots
            assert not shots.detectors.any() and not shots.core_labels.any()
    events = _events(tmp_path / "data", "small", "worker-*.jsonl")
    assert len([event for event in events if event["event"] == "worker_ready"]) == len(mapped.shards)


def test_completion_driven_submission_is_bounded_and_inventory_is_canonical(tmp_path, settings, monkeypatch):
    args, resolve = settings
    monkeypatch.setattr(generation, "_run_sampler", controlled_generation)
    monkeypatch.setenv("NNWA_GENERATION_TEST", "reorder")
    serial = generate_workflow(resolve(*args("serial", 1)))
    result = generate_workflow(resolve(*args("parallel", 2, "data.sampling_max_in_flight=2")))
    _assert_same([serial, result])
    assert (tmp_path / "data/generation-test-barriers/replenished").exists()
    events = _events(tmp_path / "data", "parallel")
    committed = [event["index"] for event in events if event["event"] == "shard_verified"]
    assert committed[0] == 1 and sorted(committed) == list(range(9))
    assert [entry["index"] for entry in read_json(result)["shards"]] == list(range(9))
    assert max(event["pending"] for event in events if event["event"] == "task_submitted") == 2
    assert all(
        event["result_bytes"] < 1024 for event in _events(tmp_path / "data", "parallel", "worker-*.jsonl") if event["event"] == "ipc"
    )


@pytest.mark.parametrize("kill", ["worker", "coordinator"])
def test_sigkill_during_partial_write_has_no_published_dataset_and_retry_matches(tmp_path, settings, kill):
    args, resolve = settings
    serial = generate_workflow(resolve(*args("serial", 1)))
    repo = Path(__file__).resolve().parents[2]
    root = tmp_path / "data"
    output = root / "killed"
    staging = root / ".killed.generation"
    env = {**os.environ, "NNWA_GENERATION_TEST": "pause", "PYTHONPATH": str(repo)}
    with _reap_adopted_processes(), (tmp_path / "generation.log").open("w+") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "tests.generation_checks", "generate", *args("killed", 2, "data.sampling_max_in_flight=2")],
            cwd=tmp_path, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            ready = root / "generation-test-barriers/before-write-publication"
            await_file(ready)
            marker = read_json(ready)
            assert len(list(Path(marker["path"]).parent.glob(".detectors.npy.*"))) == 1
            assert not output.exists()
            with pytest.raises(ArtifactError, match="another process"):
                generate_workflow(resolve(*args("killed", 2)))
            os.kill(marker["pid"] if kill == "worker" else process.pid, signal.SIGKILL)
            assert process.wait(timeout=15) != 0
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=15)
            _reap_group(process.pid)
    assert not output.exists()
    assert staging.exists() == (kill == "coordinator")
    retry = generate_workflow(resolve(*args("killed", 4)))
    _assert_same([serial, retry])
    assert not staging.exists()


def test_generation_preserves_split_and_continuation_namespaces(settings):
    args, resolve = settings
    volumes = (
        ("base", "train", "full_closed"), ("continuation", "train", "continued_full"),
        ("validation", "validation", "full_closed"), ("test", "test", "full_closed"),
    )
    namespaces, prefixes, ids, seeds = set(), set(), set(), set()
    for name, split, preset in volumes:
        output = generate_workflow(resolve(*args(
            name, 2, f"data.split={split}", f"training={preset}", "noise.probability=0", total=1,
        )))
        dataset = open_dataset(output.parent)
        with dataset.open_shard(0) as shard:
            assert not shard.read().shots.detectors.any()
            ids.update(map(int, shard.shot_ids))
        plan = dataset.plan
        assert plan.namespace.split == split
        assert plan.namespace.phase == (name if split == "train" else "held_out")
        namespaces.add(plan.dataset_id)
        prefixes.add(plan.shot_prefix)
        seeds.add(plan.namespace.seed("circuit", 0))
    assert all(len(values) == 4 for values in (namespaces, prefixes, ids, seeds))
