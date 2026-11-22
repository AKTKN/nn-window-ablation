from dataclasses import replace
from hashlib import sha256
import json

import numpy as np
import pytest

from nn_window_ablation.artifacts import ArtifactError
from nn_window_ablation.data import TrajectoryID, open_dataset, sample_chunk, write_dataset
from nn_window_ablation.simulation import ShotBatch
from tests.p1_checks import sample_fault_signature
from tests.p4_checks import assert_exact, setup_plan
from tests.reference.p1_pauli_frame import Fault, oracle_signature
from tests.reference.p1_schedule import SMALL


def _example_dataset(path):
    bundle, plan = setup_plan(chunk=3, total=4)
    detectors = np.zeros((3, 48), dtype=np.bool_)
    detectors[1, [0, 7, 8, 47]] = True
    detectors[2, [1, 9]] = True
    first = ShotBatch(
        detectors, np.array([0, 1, 1], dtype=bool), np.array([[0, 0], [1, 0], [0, 1]], dtype=bool),
        np.arange(plan.shot_prefix, plan.shot_prefix + 3, dtype=np.int64), None,
    )
    last = sample_chunk(plan, bundle, 1)
    write_dataset(path, plan, bundle, chunks=(first, last), provenance={"fixture": "hand bytes"})
    return plan, bundle, first, last


def _integrity(path):
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": sha256(payload).hexdigest()}


def _amend_shard(path, mutate):
    shard_path = path / "shards/0/shard.json"
    shard = json.loads(shard_path.read_text())
    mutate(shard["arrays"])
    shard_path.write_text(json.dumps(shard))
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["shards"][0]["manifest"] = _integrity(shard_path)
    manifest_path.write_text(json.dumps(manifest))


def test_p4_packed_bytes_mmap_and_row_identity(tmp_path):
    path = tmp_path / "dataset"
    plan, bundle, initial, final = _example_dataset(path)
    dataset = open_dataset(path, expected_split="train")
    assert dataset.plan == plan and dataset.bundle == bundle
    first, last = dataset.shards
    assert isinstance(first.detectors, np.memmap) and not first.detectors.flags.writeable
    assert_exact(first.detectors, np.array([
        [0, 0, 0, 0, 0, 0], [129, 1, 0, 0, 0, 128], [2, 2, 0, 0, 0, 0],
    ], dtype=np.uint8))
    assert_exact(first.global_labels, np.array([[0], [1], [1]], dtype=np.uint8))
    assert_exact(first.core_labels, np.array([[0], [1], [2]], dtype=np.uint8))
    assert_exact(np.unpackbits(first.core_labels, axis=1, count=2, bitorder="little").astype(bool), initial.core_labels)
    selected = np.array([2, 0, 1], dtype=np.int64)
    shuffled = first.read(selected)
    for field in ("detectors", "global_labels", "core_labels", "shot_ids"):
        assert_exact(getattr(shuffled.shots, field), getattr(initial, field)[selected])
        assert_exact(getattr(last.read().shots, field), getattr(final, field))
    assert shuffled.identities == tuple(TrajectoryID(plan.dataset_id, 0, i) for i in [2, 0, 1])
    assert last.read().identities == (TrajectoryID(plan.dataset_id, 1, 0),)
    assert not shuffled.shots.detectors[1].any()  # Zero-event row retained.
    with pytest.raises(FileExistsError):
        write_dataset(path, plan, bundle, provenance={})


def test_p4_split_ids_disjoint_and_zero_event_rows_retained(tmp_path):
    numeric_ids, physical_ids = [], []
    for split in ("train", "validation", "test"):
        bundle, plan = setup_plan(split=split, p=0, total=3, chunk=2)
        target = tmp_path / split
        write_dataset(target, plan, bundle, provenance={})
        mapped = open_dataset(target, expected_split=split)
        batches = [shard.read() for shard in mapped.shards]
        assert all(not batch.shots.detectors.any() for batch in batches)
        numeric_ids.append({int(shot_id) for batch in batches for shot_id in batch.shots.shot_ids})
        physical_ids.append({identity for batch in batches for identity in batch.identities})
        assert len(numeric_ids[-1]) == len(physical_ids[-1]) == 3
    assert all(not ids[i] & ids[j] for ids in (numeric_ids, physical_ids) for i, j in ((0, 1), (0, 2), (1, 2)))
    with pytest.raises(ArtifactError, match="expected train dataset, found test"):
        open_dataset(tmp_path / "test", expected_split="train")


@pytest.mark.parametrize("damage", ["byte", "truncated", "shape", "duplicate", "swapped", "reassigned", "padding"])
def test_p4_rejects_damaged_shard_or_artifact_identity(tmp_path, damage):
    path = tmp_path / "dataset"
    _example_dataset(path)
    shard = path / "shards/0"
    detectors = shard / "detectors.npy"
    core = shard / "core_labels.npy"
    global_ = shard / "global_observables.npy"
    ids = shard / "shot_ids.npy"
    if damage == "byte":
        content = bytearray(detectors.read_bytes())
        content[-1] ^= 1
        detectors.write_bytes(content)
        expected = "checksum"
    elif damage == "truncated":
        detectors.write_bytes(detectors.read_bytes()[:-1])
        expected = "checksum"
    elif damage == "shape":
        np.save(detectors, np.zeros((3, 7), dtype=np.uint8))
        _amend_shard(path, lambda arrays: arrays["detectors"].update(_integrity(detectors)))
        expected = "wrong shape, dtype or valid-bit count for detectors"
    elif damage == "duplicate":
        values = np.load(ids)
        values[1] = values[0]
        np.save(ids, values)
        _amend_shard(path, lambda arrays: arrays["shot_ids"].update(_integrity(ids)))
        expected = "duplicate or reordered shot IDs"
    elif damage == "swapped":
        saved = core.read_bytes()
        core.write_bytes(global_.read_bytes())
        global_.write_bytes(saved)
        expected = "checksum"
    elif damage == "reassigned":
        _amend_shard(path, lambda arrays: arrays["global_labels"].update(filename="core_labels.npy"))
        expected = "reassigned artifact identity"
    else:
        values = np.load(global_)
        values[0, 0] |= 128
        np.save(global_, values)
        _amend_shard(path, lambda arrays: arrays["global_labels"].update(_integrity(global_)))
        expected = "nonzero padding bits"
    with pytest.raises(ArtifactError, match=expected):
        open_dataset(path)


def test_p4_independent_fault_labels_detect_structurally_valid_pairing_swap(tmp_path):
    bundle, plan = setup_plan(total=2, chunk=2)
    physical = bundle.original.to_stim().flattened()
    faults = (Fault(100, 0, "GAP", (1,), "X"), Fault(101, 0, "GAP", (1,), "X"))
    samples = [sample_fault_signature(bundle, fault) for fault in faults]
    oracle = tuple(oracle_signature(physical, SMALL, (fault,)).cores for fault in faults)
    assert oracle == ((1, 0), (0, 1))
    batch = ShotBatch(
        np.array([sample.detectors for sample in samples], dtype=bool),
        np.array([sample.global_label for sample in samples], dtype=bool),
        np.array([sample.cores for sample in samples], dtype=bool),
        np.array([plan.shot_prefix, plan.shot_prefix + 1], dtype=np.int64), None,
    )
    good = tmp_path / "aligned"
    bad = tmp_path / "same-parity-swapped"
    write_dataset(good, plan, bundle, chunks=(batch,), provenance={})
    write_dataset(bad, plan, bundle, chunks=(replace(batch, core_labels=batch.core_labels[::-1]),), provenance={})

    def check_known_pairing(path):
        loaded = open_dataset(path).shards[0].read()
        assert loaded.identities == (TrajectoryID(plan.dataset_id, 0, 0), TrajectoryID(plan.dataset_id, 0, 1))
        np.testing.assert_array_equal(loaded.shots.detectors, batch.detectors)
        np.testing.assert_array_equal(loaded.shots.core_labels, oracle, err_msg="independent label pairing")

    check_known_pairing(good)
    with pytest.raises(AssertionError, match="independent label pairing"):
        check_known_pairing(bad)
