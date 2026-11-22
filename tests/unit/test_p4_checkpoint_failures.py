from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from nn_window_ablation import artifacts, data
from nn_window_ablation.artifacts import ArtifactError
from nn_window_ablation.checkpoints import load_checkpoint, save_checkpoint
from tests.p4_checks import assert_exact, convex_loss, scalar_trainer, setup_plan


def test_failed_checkpoint_publication_preserves_previous_update(tmp_path, monkeypatch):
    bundle, plan = setup_plan()
    trainer = scalar_trainer(bundle, plan, steps=2, adam=True)
    path = tmp_path / "checkpoint.pt"
    trainer.step(convex_loss)
    save_checkpoint(path, trainer.snapshot())
    original = path.read_bytes()
    previous = load_checkpoint(path)
    trainer.step(convex_loss)
    actual_replace = artifacts.os.replace

    def interrupt(source, target):
        if Path(target) == path:
            assert load_checkpoint(Path(source))["counters"]["phase_updates"] == 2
            raise OSError("interrupted checkpoint publication")
        actual_replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(artifacts.os, "replace", interrupt)
        with pytest.raises(OSError, match="interrupted checkpoint publication"):
            save_checkpoint(path, trainer.snapshot())

    assert list(tmp_path.iterdir()) == [path]
    assert path.read_bytes() == original
    assert_exact(load_checkpoint(path), previous)
    save_checkpoint(path, trainer.snapshot())
    assert_exact(load_checkpoint(path), trainer.snapshot())


@pytest.mark.parametrize(("fault", "message"), [
    ("absent_manifest", "incomplete or has unexpected members"),
    ("future_schema", "incompatible schema"),
    ("bad_state", "state checksum or length differs"),
])
def test_checkpoint_rejects_incomplete_or_corrupt_envelopes(tmp_path, fault, message):
    bundle, plan = setup_plan()
    trainer = scalar_trainer(bundle, plan, steps=2)
    path = tmp_path / "complete.pt"
    save_checkpoint(path, trainer.snapshot())
    with ZipFile(path) as original:
        manifest = json.loads(original.read("manifest.json"))
        state = original.read("state.pt")
    if fault == "future_schema":
        manifest["schema_version"] += 1
    if fault == "bad_state":
        state = state[:-1]
    candidate = tmp_path / "incomplete.pt"
    with ZipFile(candidate, "w") as changed:
        changed.writestr("state.pt", state)
        if fault != "absent_manifest":
            changed.writestr("manifest.json", json.dumps(manifest))

    with pytest.raises(ArtifactError, match=message):
        load_checkpoint(candidate)
    assert_exact(load_checkpoint(path), trainer.snapshot())


def test_dataset_manifest_is_published_after_all_shards(tmp_path, monkeypatch):
    bundle, plan = setup_plan(split="validation", total=4)
    path = tmp_path / "dataset"
    actual_write = data.write_json

    def interrupt(target, record):
        if target.name == "manifest.json":
            assert set(target.parent.glob("shards/*/shard.json")) == {
                target.parent / "shards" / str(entry["index"]) / "shard.json" for entry in record["shards"]
            }
            with pytest.raises(ArtifactError, match="manifest"):
                data.open_dataset(target.parent)
            assert not path.exists()
            raise OSError("interrupted dataset finalization")
        actual_write(target, record)

    with monkeypatch.context() as patch:
        patch.setattr(data, "write_json", interrupt)
        with pytest.raises(OSError, match="interrupted dataset finalization"):
            data.write_dataset(path, plan, bundle, provenance={"fixture": True})

    assert not list(tmp_path.iterdir())
    data.write_dataset(path, plan, bundle, provenance={"fixture": True})
    assert data.open_dataset(path, expected_split="validation").plan == plan
    manifest = path / "manifest.json"
    record = json.loads(manifest.read_text())
    del record["provenance"]
    manifest.write_text(json.dumps(record))
    with pytest.raises(ArtifactError, match="manifest lacks a provenance record"):
        data.open_dataset(path)


def test_fork_starts_fresh_and_independent_resume_restores_phase_state(tmp_path):
    bundle, base = setup_plan()
    continuation = replace(base, namespace=replace(base.namespace, phase="continuation"))
    parent = scalar_trainer(bundle, base, steps=7, adam=True)
    child = scalar_trainer(bundle, continuation, steps=4, adam=True)
    with pytest.raises(ValueError, match="completed base checkpoint first"):
        child.step(convex_loss)
    parent.step(convex_loss)
    with pytest.raises(ArtifactError, match="fixed terminal update"):
        child.fork_from(parent.snapshot())
    for _ in range(6):
        parent.step(convex_loss)
    terminal = parent.snapshot()
    parent_as_child = scalar_trainer(bundle, base, steps=4, adam=True)
    with pytest.raises(ArtifactError, match="distinct continuation namespace"):
        parent_as_child.fork_from(terminal)

    initial_rng = child.randomness.record()
    initial_scheduler = deepcopy(child.scheduler.state_dict())
    child.fork_from(terminal)
    assert_exact(child.model.state_dict(), terminal["model"]["state_dict"])
    assert_exact(child.randomness.record(), initial_rng)
    assert_exact(child.scheduler.state_dict(), initial_scheduler)
    assert not child.optimizer.state
    child.step(convex_loss)
    assert child.optimizer.state and child.scheduler.last_epoch == 1
    path = tmp_path / "ongoing.pt"
    save_checkpoint(path, child.snapshot())
    payload = load_checkpoint(path)

    incompatible = scalar_trainer(bundle, continuation, steps=3, adam=True)
    with pytest.raises(ArtifactError, match="training contract or model schema differs"):
        incompatible.restore(payload)
    wrong_optimizer = scalar_trainer(bundle, continuation, steps=4)
    with pytest.raises(ArtifactError, match="training contract or model schema differs"):
        wrong_optimizer.restore(payload)
    resumed = scalar_trainer(bundle, continuation, steps=4, adam=True)
    resumed.restore(payload)
    assert_exact(resumed.snapshot(), child.snapshot())
    for _ in range(3):
        assert resumed.step(convex_loss) == child.step(convex_loss)
    assert_exact(resumed.snapshot(), child.snapshot())
    assert resumed.counters == {
        "phase_updates": 4, "cumulative_updates": 11, "parent_updates": 7, "trajectories": 12, "query_examples": 24,
    }
    with pytest.raises(ValueError, match="declared update budget"):
        resumed.step(convex_loss)
