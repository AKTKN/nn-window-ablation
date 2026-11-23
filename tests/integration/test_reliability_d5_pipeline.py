"""Compose finite stored training, crash recovery, and sealed paired results."""

from pathlib import Path
import select
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from nn_window_ablation.artifacts import ArtifactError, file_integrity, read_json
from nn_window_ablation.checkpoints import load_checkpoint
from nn_window_ablation.config import resolve_config
from nn_window_ablation.data import open_labels
from nn_window_ablation.evaluation import FrozenEvaluator
from nn_window_ablation.execution import generate_workflow, train_workflow
from nn_window_ablation.export import export_bundle
from nn_window_ablation.paired_workflows import analyze_workflow, evaluate_workflow
from nn_window_ablation.pbs import _verified_artifact
from tests.p4_checks import assert_exact
from tests.p7_checks import cli


@pytest.mark.resource_process
def test_d5_stored_training_killed_evaluation_and_replayed_statistics(tmp_path, compose_settings, monkeypatch):
    def resolve(*overrides):
        return resolve_config(compose_settings("runtime=test_cpu", *overrides), invocation_dir=tmp_path)

    generate_workflow(resolve("data.split=train", "data.id=training", "data.shots=7", "data.shard_shots=3"))
    training = (
        "data.split=train", "data.id=training", "data.train_source=stored_shards", "training.max_steps=3",
        "training.batch_size=2", "training.microbatch_size=1", "training.optimizer=adam",
        "training.scheduler=step", "training.scheduler_step_size=1", "training.scheduler_gamma=0.9",
        "training.learning_rate=0.001", "checkpoint.save_every_updates=2",
    )
    with pytest.raises(ValueError, match="cannot supply"):
        train_workflow(resolve("run_name=exhausted", *training, "training.max_steps=4"))
    with pytest.raises(ArtifactError, match="namespace"):
        train_workflow(resolve("run_name=wrong-seed", *training, "seed=22"))
    reference = train_workflow(resolve("run_name=uninterrupted", *training))
    checkpoint = train_workflow(resolve("run_name=base", *training), stop_after=1)
    assert load_checkpoint(checkpoint)["counters"]["phase_updates"] == 1
    cli(tmp_path, "train", "runtime=test_cpu", "run_name=base", *training, "checkpoint.resume_from=runs/base/checkpoint.pt")
    latest = load_checkpoint(checkpoint)
    full = load_checkpoint(reference)
    for key in ("model", "random_states", "optimizer_state", "scheduler_state", "cursor", "last_update", "counters"):
        assert_exact(latest[key], full[key])
    assert latest["model"]["architecture"]["hidden_size"] == 256
    assert latest["model"]["architecture"]["gru_layers"] == 4
    assert latest["counters"]["phase_updates"] == 3 and latest["counters"]["trajectories"] == 6

    held = generate_workflow(resolve("data.split=test", "data.id=held", "data.shots=3", "data.shard_shots=2")).parent
    settings = ("data.split=test", "data.id=held", "evaluation=paired", "checkpoint.path=runs/base/checkpoint.pt")

    def evaluation(name, *, resume=False):
        return resolve(f"run_name={name}", *settings, f"evaluation.resume={str(resume).lower()}")

    expected = evaluate_workflow(evaluation("reference"))
    destination = tmp_path / "runs/crash"
    program = """
from pathlib import Path
import sys
from nn_window_ablation import paired_workflows
from nn_window_ablation.cli import main
original = paired_workflows._write_array
def pause(path, values):
    receipt = original(path, values)
    if path.name == 'shot_ids.npy' and (Path('runs/crash/shards/0') / 'receipt.json').exists():
        print('P7_SHARD_BARRIER', flush=True)
        sys.stdin.buffer.read(1)
    return receipt
paired_workflows._write_array = pause
sys.argv[0:2] = ['nnwa', 'evaluate']
raise SystemExit(main())
"""
    child = subprocess.Popen(
        [sys.executable, "-c", program, "discard", "runtime=test_cpu", "run_name=crash", *settings],
        cwd=tmp_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    try:
        ready, _, _ = select.select([child.stdout], [], [], 60)
        assert ready, "child never reached the durable shard barrier"
        assert child.stdout.readline().strip() == b"P7_SHARD_BARRIER"
        committed = destination / "shards/0/receipt.json"
        observed = file_integrity(committed)
        assert (destination / "shards/.pending/shot_ids.npy").is_file()
        assert not (destination / "manifest.json").exists()
        with pytest.raises(ArtifactError, match="another process"):
            evaluate_workflow(evaluation("crash", resume=True))
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=15)
    assert child.returncode != 0
    assert (destination / "shards/.pending").exists()  # SIGKILL skipped Python cleanup.

    def analysis(name, source):
        return resolve(
            f"run_name={name}", "data.split=test", "data.id=held",
            f"analysis.predictions=[{source}]", "analysis.bootstrap_replicates=41", "analysis.bootstrap_seed=19",
            "analysis.sparse_rule=1",
        )

    with pytest.raises(ArtifactError, match="manifest"):
        analyze_workflow(analysis("early", "runs/crash/manifest.json"))
    with pytest.raises(ArtifactError, match="manifest"):
        _verified_artifact(SimpleNamespace(stage="evaluate"), evaluation("crash"), destination / "manifest.json")
    report_reference = analyze_workflow(analysis("summary-ref", "runs/reference/manifest.json"))
    incomplete = read_json(report_reference)
    # Bind the current test report to the missing final manifest, so export must reject it.
    incomplete["inputs"][0]["path"] = str(destination / "manifest.json")
    from nn_window_ablation.artifacts import write_json

    write_json(report_reference.with_name("incomplete.json"), incomplete)
    with pytest.raises(ArtifactError, match="missing or unreadable"):
        export_bundle(held, report_reference.with_name("incomplete.json"), tmp_path / "too-early.tar.gz")

    predictions = []
    original = FrozenEvaluator.logits
    def inspect(self, mode, detectors, ids):
        predictions.append((mode, tuple(int(shot) & 0xFFFFFFFF for shot in ids)))
        return original(self, mode, detectors, ids)
    monkeypatch.setattr(FrozenEvaluator, "logits", inspect)
    repaired = evaluate_workflow(evaluation("crash", resume=True))
    assert predictions == [("G", (2,)), ("F", (2,)), ("S", (2,))]
    assert file_integrity(committed) == observed
    assert sorted(path.name for path in (destination / "shards").iterdir()) == ["0", "1"]
    predictions.clear()
    evaluate_workflow(evaluation("crash", resume=True))
    assert not predictions
    receipt, _ = _verified_artifact(SimpleNamespace(stage="evaluate"), evaluation("crash", resume=True), repaired)
    assert receipt["sha256"] == file_integrity(repaired)["sha256"]

    failures = {mode: [] for mode in ("G", "F", "S")}
    mapped = open_labels(held)
    for descriptor, entry in zip(mapped.shards, read_json(repaired)["shards"], strict=True):
        with mapped.open_shard(descriptor.index) as shard:
            labels, core_labels = shard.read_labels()
            ids = shard.shot_ids
        assert np.array_equal(np.logical_xor.reduce(core_labels, axis=1), labels)
        for name in entry["files"]:
            suffix = f"shards/{descriptor.index}/{name}"
            assert (repaired.parent / suffix).read_bytes() == (expected.parent / suffix).read_bytes()
        np.testing.assert_array_equal(np.load(repaired.parent / f"shards/{descriptor.index}/shot_ids.npy"), ids)
        for mode in failures:
            values = np.load(repaired.parent / f"shards/{descriptor.index}/{mode}.decisions.npy")
            bits = np.unpackbits(values, count=1 if mode == "G" else 12, axis=1, bitorder="little")
            errors = [bool(sum(map(int, bits[row])) % 2) != bool(labels[row]) for row in range(len(labels))]
            failures[mode].extend(errors)

    report = analyze_workflow(analysis("summary", "runs/crash/manifest.json"))
    data = read_json(report)
    for mode, bits in failures.items():
        assert data["rates"][mode]["shots"] == 3
        assert data["rates"][mode]["failures"] == sum(bits)
    for contrast in data["contrasts"].values():
        pairs = list(zip(failures[contrast["first"]], failures[contrast["second"]], strict=True))
        discordant = [pairs.count((True, False)), pairs.count((False, True))]
        assert (contrast["n10"], contrast["n01"]) == tuple(discordant)
        assert contrast["difference"] == (discordant[0] - discordant[1]) / 3
    assert data["contrasts"] == read_json(report_reference)["contrasts"]

    archive = export_bundle(held, report, tmp_path / "result.tar.gz")
    cli(tmp_path, "extract", archive, "imported")
    cli(tmp_path, "reanalyze", "--bundle", "imported", "--output", "replayed")
    restored = read_json(tmp_path / "replayed/report.json")
    assert restored["rates"] == data["rates"] and restored["contrasts"] == data["contrasts"]
