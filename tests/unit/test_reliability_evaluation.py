"""Fault boundaries and semantic validation of published prediction shards."""

from collections import Counter
from pathlib import Path
import shutil

import numpy as np
import pytest

from nn_window_ablation import artifacts, paired_workflows as workflows
from nn_window_ablation.artifacts import ArtifactError, file_integrity, read_json, write_json
from nn_window_ablation.checkpoints import load_checkpoint, save_checkpoint
from nn_window_ablation.config import ConfigError, resolve_config
from nn_window_ablation.evaluation import FrozenEvaluator
from nn_window_ablation.execution import generate_workflow, train_workflow


@pytest.fixture
def evaluation_config(tmp_path, compose_settings):
    def resolve(*overrides):
        return resolve_config(compose_settings("runtime=test_cpu", "code.distance=3", *overrides), invocation_dir=tmp_path)

    generate_workflow(resolve("data.split=test", "data.id=held", "data.shots=5", "data.shard_shots=2", "noise.probability=0"))
    train_workflow(resolve(
        "run_name=base", "data.split=train", "training.max_steps=1", "training.batch_size=1",
        "training.optimizer=sgd", "training.scheduler=none", "training.learning_rate=0.001",
        "data.training_chunk_shots=1", "noise.probability=0",
    ))

    def evaluate(name, *overrides, resume=False):
        return resolve(
            f"run_name={name}", "data.split=test", "data.id=held", "evaluation=paired",
            "checkpoint.path=runs/base/checkpoint.pt", "noise.probability=0",
            f"evaluation.resume={str(resume).lower()}", *overrides,
        )

    return evaluate


@pytest.mark.parametrize("boundary", [
    "before-payload", "between-writes", "before-commit", "after-commit", "before-final", "replace-final", "after-final",
])
def test_evaluation_commit_failures_and_verified_idempotence(tmp_path, evaluation_config, monkeypatch, boundary):
    desired = workflows.evaluate_workflow(evaluation_config("reference"))
    directory = tmp_path / "runs/candidate"
    original = {
        "payload": workflows._write_array, "publish": workflows.publish_directory,
        "json": workflows.write_json, "replace": artifacts.os.replace, "sync": artifacts._sync_directory,
    }
    count = 0

    def fail_payload(path, array):
        nonlocal count
        count += 1
        if count == (1 if boundary == "before-payload" else 3):
            raise OSError(boundary)
        return original["payload"](path, array)

    def fail_commit(stage, target):
        if boundary == "before-commit":
            assert (stage / "receipt.json").exists()
            raise OSError(boundary)
        original["publish"](stage, target)
        raise OSError(boundary)

    def fail_final(path, record):
        if path == directory / "manifest.json":
            raise OSError(boundary)
        original["json"](path, record)

    def fail_replace(source, path):
        if Path(path) == directory / "manifest.json":
            assert read_json(Path(source))["status"] == "complete"
            raise OSError(boundary)
        original["replace"](source, path)

    def fail_final_sync(path):
        if path == directory and (directory / "manifest.json").exists():
            raise OSError(boundary)
        original["sync"](path)

    with monkeypatch.context() as patch:
        if boundary in ("before-payload", "between-writes"):
            patch.setattr(workflows, "_write_array", fail_payload)
        elif boundary in ("before-commit", "after-commit"):
            patch.setattr(workflows, "publish_directory", fail_commit)
        elif boundary == "before-final":
            patch.setattr(workflows, "write_json", fail_final)
        elif boundary == "after-final":
            patch.setattr(artifacts, "_sync_directory", fail_final_sync)
        else:
            patch.setattr(artifacts.os, "replace", fail_replace)
        with pytest.raises(OSError, match=boundary):
            workflows.evaluate_workflow(evaluation_config("candidate"))

    expected_committed = {
        "before-payload": 0, "between-writes": 0, "before-commit": 0,
        "after-commit": 1, "before-final": 3, "replace-final": 3, "after-final": 3,
    }[boundary]
    assert (directory / "manifest.json").exists() == (boundary == "after-final")
    assert sorted(p.name for p in (directory / "shards").iterdir()) == [str(i) for i in range(expected_committed)]
    with pytest.raises(FileExistsError):
        workflows.evaluate_workflow(evaluation_config("candidate"))
    evaluated = Counter()
    original_logits = FrozenEvaluator.logits

    def predict(self, mode, detectors, ids, **kwargs):
        evaluated[int(ids[0]) & 0xFFFFFFFF] += 1
        return original_logits(self, mode, detectors, ids, **kwargs)

    monkeypatch.setattr(FrozenEvaluator, "logits", predict)
    resumed = workflows.evaluate_workflow(evaluation_config("candidate", resume=True))
    assert evaluated == Counter({index: 3 for index in range(min(5, 2 * expected_committed), 5)})
    evaluated.clear()
    final_hash = file_integrity(resumed)
    assert workflows.evaluate_workflow(evaluation_config("candidate", resume=True)) == resumed
    assert not evaluated and file_integrity(resumed) == final_hash
    reference, result = read_json(desired), read_json(resumed)
    assert result["contract"] == reference["contract"]
    assert result["shards"] == reference["shards"]
    for entry in result["shards"]:
        for name in entry["files"]:
            suffix = f"shards/{entry['index']}/{name}"
            assert (resumed.parent / suffix).read_bytes() == (desired.parent / suffix).read_bytes()


@pytest.mark.parametrize(("damage", "message"), [
    ("missing", "missing or unexpected"), ("truncated", "size or checksum"),
    ("changed", "size or checksum"), ("ids", "shot identities differ or are reordered"),
    ("duplicate-ids", "duplicate shot identities"), ("dtype", "shape, dtype"),
    ("shape", "shape, dtype"), ("padding", "padding"), ("nonfinite", "finite logits"),
    ("xor", "core-decision XOR"), ("source", "size or checksum"),
    ("receipt", "incompatible source or run identity"), ("extra", "out-of-range"),
])
def test_committed_shard_damage_is_fatal_even_when_resealed(tmp_path, evaluation_config, monkeypatch, damage, message):
    path = workflows.evaluate_workflow(evaluation_config("target"))
    directory = path.parent / "shards/0"
    receipt = read_json(directory / "receipt.json")
    filename = {
        "ids": "shot_ids.npy", "duplicate-ids": "shot_ids.npy", "padding": "F.decisions.npy",
        "nonfinite": "S.logits.npy", "xor": "F.final.npy",
    }.get(damage, "G.logits.npy")
    file = directory / filename
    if damage == "missing":
        file.unlink()
    elif damage in ("changed", "truncated", "source"):
        if damage == "source":
            file = tmp_path / "data/held/shards/0/detectors.npy"
        raw = file.read_bytes()
        file.write_bytes(raw[:-1] if damage == "truncated" else raw[:-1] + bytes([raw[-1] ^ 1]))
    elif damage == "receipt":
        receipt["source"]["manifest"]["sha256"] = "0" * 64
        write_json(directory / "receipt.json", receipt)
    elif damage == "extra":
        (directory.parent / "3").mkdir()
    else:
        rows = np.load(file, allow_pickle=False)
        if damage == "ids":
            rows = rows[::-1]
        elif damage == "duplicate-ids":
            rows[0] = rows[1]
        elif damage == "shape":
            rows = rows[:, :0]
        elif damage == "dtype":
            rows = rows.astype(np.float64)
        elif damage == "padding":
            rows[0, 1] |= 128
        elif damage == "nonfinite":
            rows[0, 0] = np.nan
        else:
            rows[0, 0] ^= 1
        np.save(file, rows, allow_pickle=False)
        receipt["files"][filename].update(file_integrity(file))
        write_json(directory / "receipt.json", receipt)
    monkeypatch.setattr(FrozenEvaluator, "logits", lambda *_: pytest.fail("committed shard was re-inferred"))
    with pytest.raises(ArtifactError, match=message):
        workflows.evaluate_workflow(evaluation_config("target", resume=True))


def test_resume_contract_is_content_based_and_rejects_policy_changes(tmp_path, evaluation_config, monkeypatch):
    original = workflows.evaluate_workflow(evaluation_config("original"))
    monkeypatch.setattr(FrozenEvaluator, "logits", lambda *_: pytest.fail("verified shard was re-inferred"))
    for change in (
        "evaluation.batch_size=2", "evaluation=G", "window.left_per_distance=0.5",
        "evaluation.matching=true",
    ):
        with pytest.raises(ArtifactError, match="contract differs"):
            workflows.evaluate_workflow(evaluation_config("original", change, resume=True))
    with pytest.raises(ArtifactError, match="contract differs"):
        workflows.evaluate_workflow(evaluation_config(
            "original", "evaluation.workers=2", "evaluation.worker_threads=2", resume=True,
        ))
    assert workflows.evaluate_workflow(evaluation_config(
        "original", "runtime.allocated_cpus=7", "evaluation.workers=2",
        "evaluation.max_in_flight=1", resume=True,
    )) == original
    with pytest.raises(ConfigError, match="threshold"):
        workflows.evaluate_workflow(evaluation_config("original", "evaluation.threshold=0.51", resume=True))

    checkpoint = tmp_path / "runs/base/checkpoint.pt"
    unrelated = tmp_path / "other.pt"
    payload = load_checkpoint(checkpoint)
    payload["nonsemantic_perturbation"] = 1
    save_checkpoint(unrelated, payload)
    with pytest.raises(ArtifactError, match="contract differs"):
        workflows.evaluate_workflow(evaluation_config("original", f"checkpoint.path={unrelated}", resume=True))

    shutil.copyfile(checkpoint, tmp_path / "copy.pt")
    shutil.copytree(tmp_path / "data/held", tmp_path / "data/alias")
    renamed = original.parent.with_name("moved")
    original.parent.rename(renamed)
    assert workflows.evaluate_workflow(evaluation_config(
        "moved", "data.id=alias", f"checkpoint.path={tmp_path / 'copy.pt'}", resume=True,
    )) == renamed / "manifest.json"
    dataset_manifest = tmp_path / "data/alias/manifest.json"
    altered = read_json(dataset_manifest)
    altered["provenance"]["attempt"] = 2
    write_json(dataset_manifest, altered)
    with pytest.raises(ArtifactError, match="contract differs"):
        workflows.evaluate_workflow(evaluation_config("moved", "data.id=alias", resume=True))
    contract = renamed / "contract.json"
    record = read_json(contract)
    record["schema_version"] += 1
    write_json(contract, record)
    with pytest.raises(ArtifactError, match="contract differs"):
        workflows.evaluate_workflow(evaluation_config("moved", resume=True))


def test_old_terminal_checkpoint_and_completed_predictions_remain_readable(tmp_path, evaluation_config):
    checkpoint = tmp_path / "runs/base/checkpoint.pt"
    payload = load_checkpoint(checkpoint)
    for key in ("resume_contract", "resolved_config"):
        del payload["run"][key]["checkpoint"]["save_every_updates"]
        payload["run"][key]["evaluation"].pop("resume", None)
    save_checkpoint(checkpoint, payload)
    result = workflows.evaluate_workflow(evaluation_config("legacy"))
    old = read_json(result)
    old["schema_version"] = 1
    for key in ("contract", "contract_sha256"):
        del old[key]
    del old["resolved_config"]["evaluation"]["resume"]
    for entry in old["shards"]:
        del entry["source"]
    write_json(result, old)
    (result.parent / "contract.json").unlink()
    mapped = workflows._dataset(evaluation_config("legacy"))[1]
    # Historical complete outputs remain usable by the prediction reader.
    manifest = workflows._read_manifest(result, mapped)
    with mapped.open_shard(0, labels_only=True) as labels:
        assert {row[0] for row in workflows._read_shard(result, manifest, labels)} == {"G", "F", "S"}
    with pytest.raises(ArtifactError, match="new run name for legacy"):
        workflows.evaluate_workflow(evaluation_config("legacy", resume=True))


def test_schema_two_completed_manifest_is_readable_but_legacy_run_cannot_resume(evaluation_config):
    config = evaluation_config("legacy")
    path = workflows.evaluate_workflow(config)
    record = read_json(path)
    record["schema_version"] = record["contract"]["prediction_schema"] = 2
    for option in ("workers", "worker_threads", "max_in_flight"):
        record["resolved_config"]["evaluation"].pop(option)
    record["contract"]["request"] = workflows._evaluation_request(record["resolved_config"], schema=2)
    record["contract_sha256"] = workflows._semantic_digest(record["contract"])
    write_json(path, record)
    mapped = workflows._dataset(config)[1]
    assert workflows._read_manifest(path, mapped) == record
    contract = path.with_name("contract.json")
    write_json(contract, {"schema_version": 1, "semantic": record["contract"]})
    path.unlink()
    with pytest.raises(ArtifactError, match="legacy schema"):
        workflows.evaluate_workflow(evaluation_config("legacy", resume=True))


@pytest.mark.parametrize(("options", "message"), [
    (("evaluation.workers=2", "runtime.intraop_threads=4"), "parallel evaluation requires"),
    (("evaluation.workers=2", "evaluation.worker_threads=5"), "require 10 CPUs"),
    (("evaluation.max_in_flight=2",), "inline evaluation requires"),
    (("evaluation.workers=2", "data.num_workers=1"), "single process"),
    (("evaluation.workers=2", "data.sampling_processes=2"), "single process"),
    (("evaluation.workers=0",), "must be positive"),
    (("evaluation.worker_threads=0",), "must be positive"),
    (("evaluation.max_in_flight=true",), "must be an integer"),
    (("evaluation.workers=2",), "inherited OPENBLAS_NUM_THREADS"),
])
def test_parallel_budget_is_checked_before_opening_artifacts(tmp_path, compose_settings, monkeypatch, options, message):
    if "inherited" in message:
        monkeypatch.setenv("OPENBLAS_NUM_THREADS", "16")
    with pytest.raises(ConfigError, match=message):
        config = resolve_config(compose_settings(
            "runtime=test_cpu", "data.split=test", "data.id=missing", "checkpoint.path=missing.pt", *options,
        ), invocation_dir=tmp_path)
        workflows.evaluate_workflow(config)
    assert list(tmp_path.iterdir()) == []


def test_repeated_interrupted_attempts_use_unique_owned_staging_paths(tmp_path, evaluation_config, monkeypatch):
    original = workflows.publish_directory
    root = tmp_path / "runs/candidate"
    stages = set()

    def interrupt(stage, target):
        if target.name == "1":
            assert stage.name == "1" and stage not in stages
            stages.add(stage)
            raise OSError("repeat interruption")
        return original(stage, target)

    with monkeypatch.context() as patch:
        patch.setattr(workflows, "publish_directory", interrupt)
        for attempt in range(3):
            with pytest.raises(OSError, match="repeat interruption"):
                workflows.evaluate_workflow(evaluation_config("candidate", resume=attempt > 0))
            assert [p.name for p in (root / "shards").iterdir()] == ["0"]
            assert len(list(root.glob("**/receipt.json"))) == 1
            assert not (root / ".evaluation-staging").exists()
    path = workflows.evaluate_workflow(evaluation_config("candidate", resume=True))
    assert len(read_json(path)["shards"]) == 3
