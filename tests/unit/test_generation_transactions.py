from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nn_window_ablation import generation
from nn_window_ablation.artifacts import ArtifactError, file_integrity, read_json, write_json
from nn_window_ablation.config import ConfigError, resolve_config
from nn_window_ablation.execution import generate_workflow
from tests.p4_checks import setup_plan


@pytest.mark.parametrize("damage", ["plan", "seed", "start", "rows", "hash", "symlink", "extra", "missing", "shape", "padding", "ids"])
def test_coordinator_checks_written_generation_shard(tmp_path, damage):
    bundle, plan = setup_plan(total=2, chunk=2)
    shards = tmp_path / "shards"
    shards.mkdir()
    sampler = generation._Sampler(plan, bundle, shards, "attempt", tmp_path / "log")
    task = generation._task(plan, 0, "attempt")
    result = sampler.run(task)
    path = shards / "0"
    if damage in ("plan", "seed", "start", "rows"):
        name, value = {"plan": ("dataset_id", "other"), "seed": ("seed", task.seed + 1), "start": ("start", 1), "rows": ("rows", 3)}[damage]
        altered = replace(task, **{name: value})
        with pytest.raises(ArtifactError, match="plan, interval, seed"):
            sampler.run(altered)
        result = replace(result, task=altered)
    elif damage == "hash":
        result = replace(result, manifest={**result.manifest, "sha256": "0" * 64})
    elif damage == "symlink":
        path.rename(tmp_path / "outside")
        path.symlink_to(tmp_path / "outside", target_is_directory=True)
    elif damage == "extra":
        (path / "extra").touch()
    elif damage == "missing":
        (path / "shot_ids.npy").unlink()
    else:
        filename, role = ("shot_ids.npy", "shot_ids") if damage == "ids" else ("core_labels.npy", "core_labels")
        values = np.load(path / filename)
        if damage == "ids":
            values[1] = values[0]
        elif damage == "padding":
            values[0, -1] |= 128
        else:
            values = values[:1]
        np.save(path / filename, values, allow_pickle=False)
        record = read_json(path / "shard.json")
        record["arrays"][role].update(file_integrity(path / filename))
        record["arrays"][role]["shape"] = list(values.shape)
        write_json(path / "shard.json", record)
        result = replace(result, manifest=file_integrity(path / "shard.json"))
    with pytest.raises(ArtifactError):
        generation._verify_result(shards, plan, task, result)


def test_same_task_can_be_recomputed_with_exact_bytes(tmp_path):
    bundle, plan = setup_plan(total=3, chunk=2)
    task = generation._task(plan, 1, "same-attempt")
    roots = [tmp_path / path for path in ("first", "retry")]
    for root in roots:
        root.mkdir()
    receipts = [generation._Sampler(plan, bundle, root, task.attempt, tmp_path / "log").run(task) for root in roots]
    assert receipts[0].manifest == receipts[1].manifest
    assert all(file.read_bytes() == (roots[1] / "1" / file.name).read_bytes() for file in (roots[0] / "1").iterdir())


@pytest.mark.parametrize("when", ["before", "after"])
def test_dataset_publication_fault_keeps_only_durable_artifact(tmp_path, compose_settings, monkeypatch, when):
    def resolve():
        return resolve_config(compose_settings(
            "runtime=test_cpu", "data.id=fault", "data.shots=3", "data.shard_shots=2",
        ), invocation_dir=tmp_path)

    original = generation.publish_directory

    def fail(source, destination):
        if when == "after":
            original(source, destination)
        raise OSError("simulated publication failure")

    with monkeypatch.context() as patch:
        patch.setattr(generation, "publish_directory", fail)
        with pytest.raises(OSError, match="simulated publication"):
            generate_workflow(resolve())
    assert not (tmp_path / "data/.fault.generation").exists()
    expected = tmp_path / "data/fault/manifest.json"
    assert expected.is_file() == (when == "after")
    if when == "after":
        with pytest.raises(FileExistsError):
            generate_workflow(resolve())
    else:
        assert generate_workflow(resolve()) == expected


@pytest.mark.parametrize(("commands", "message"), [
    (("data.sampling_processes=3", "runtime.allocated_cpus=2"), "exceeding runtime.allocated"),
    (("data.sampling_processes=2", "data.num_workers=1"), "zero loader"),
    (("data.sampling_processes=2", "runtime.intraop_threads=2"), "one intra-op"),
    (("data.sampling_max_in_flight=2",), "serial generation"),
])
def test_generation_budget_fails_before_reserving_output(tmp_path, compose_settings, commands, message):
    with pytest.raises(ConfigError, match=message):
        config = resolve_config(compose_settings(
            "runtime=test_cpu", "data.shots=3", "data.shard_shots=2", *commands,
        ), invocation_dir=tmp_path)
        generate_workflow(config)
    assert list(tmp_path.iterdir()) == []
