"""Lifecycle checks for metadata-only dataset opens and owned readers."""

from contextlib import contextmanager
import json
import subprocess
import sys

import numpy as np
import pytest

from nn_window_ablation import data
from nn_window_ablation.artifacts import ArtifactError, file_integrity
from nn_window_ablation.streams import TrajectoryStream
from tests.p4_checks import assert_exact, setup_plan


def _dataset(tmp_path, *, rows=9, chunk=4):
    bundle, plan = setup_plan(total=rows, chunk=chunk, p=0)
    root = tmp_path / "data"
    data.write_dataset(root, plan, bundle, provenance={})
    return bundle, plan, root


def test_metadata_open_and_one_cached_shard_per_stream(tmp_path, monkeypatch):
    bundle, plan, root = _dataset(tmp_path)
    original = data._map_array
    active = set()
    calls = []

    @contextmanager
    def observe(spec):
        calls.append(spec.path)
        with original(spec) as array:
            active.add(spec.path)
            assert len({path.parent for path in active}) == 1
            try:
                yield array
            finally:
                active.remove(spec.path)

    monkeypatch.setattr(data, "_map_array", observe)
    mapped = data.open_dataset(root)
    assert [desc.rows for desc in mapped.shards] == [4, 4, 1]
    assert not calls
    with TrajectoryStream(plan, bundle, mapped) as stored, TrajectoryStream(plan, bundle) as generated:
        for count in (1, 1, 3, 1):
            actual = stored.take(count)
            expected = generated.take(count)
            assert actual.identities == expected.identities
            for role in ("shot_ids", "detectors", "global_labels", "core_labels"):
                assert_exact(getattr(actual.shots, role), getattr(expected.shots, role))
        assert len(active) == 4
        assert len(calls) == 8 and len(set(calls)) == 8
    assert not active
    with pytest.raises(RuntimeError, match="stream is closed"):
        stored.take(1)

    with mapped.open_shard(0) as handle:
        batch = handle.read(np.array([3, 0, 2], dtype=np.int64))
        assert batch.shots.shot_ids.flags.owndata
        assert batch.shots.core_labels.flags.owndata
        with pytest.raises(RuntimeError, match="close the active"):
            with mapped.open_shard(1):
                pytest.fail("nested open")
        handle.close()
        handle.close()
        with pytest.raises(RuntimeError, match="handle is closed"):
            handle.read()
    with mapped.open_shard(2):
        pass
    assert not active
    assert not batch.shots.detectors.any()
    assert [row.row for row in batch.identities] == [3, 0, 2]


def test_partial_mapping_failure_and_labels_only_reader(tmp_path, monkeypatch):
    _, _, root = _dataset(tmp_path)
    mapped = data.open_dataset(root)
    original = data._map_array
    events = []

    @contextmanager
    def observe(spec):
        if spec.role == "core_labels":
            raise ArtifactError("third mapping fails")
        with original(spec) as array:
            events.append(("open", spec.role))
            try:
                yield array
            finally:
                events.append(("close", spec.role))

    monkeypatch.setattr(data, "_map_array", observe)
    with pytest.raises(ArtifactError, match="third mapping"):
        with mapped.open_shard(0):
            pytest.fail("partially opened")
    assert events == [
        ("open", "detectors"), ("open", "global_labels"), ("close", "global_labels"), ("close", "detectors"),
    ]
    monkeypatch.undo()
    with pytest.raises(RuntimeError, match="body error"):
        with mapped.open_shard(0):
            raise RuntimeError("body error")
    for desc in mapped.shards:
        (desc.directory / "detectors.npy").unlink()
    for name in data.PHYSICAL_FILES:
        (root / name).unlink()
    labels = data.open_labels(root)
    labels.verify_payloads()
    with labels.open_shard(0) as shard:
        original_labels = shard.read_labels()
        with pytest.raises(ValueError, match="labels-only"):
            shard.read()
    assert not any(array.any() for array in original_labels)


def test_resealed_bad_npy_header_fails_at_payload_boundary(tmp_path):
    _, _, root = _dataset(tmp_path)
    damaged = root / "shards/0/core_labels.npy"
    damaged.write_bytes(b"this is not an NPY header")
    shard_path = damaged.with_name("shard.json")
    record = json.loads(shard_path.read_text())
    record["arrays"]["core_labels"].update(file_integrity(damaged))
    shard_path.write_text(json.dumps(record))
    manifest = root / "manifest.json"
    record = json.loads(manifest.read_text())
    record["shards"][0]["manifest"] = file_integrity(shard_path)
    manifest.write_text(json.dumps(record))
    dataset = data.open_dataset(root)
    with pytest.raises(ArtifactError, match="invalid NPY payload"):
        with dataset.open_shard(0):
            pytest.fail("corrupt payload accepted")
    with dataset.open_shard(1) as shard:
        assert len(shard.shot_ids) == 4


@pytest.mark.resource_process
def test_many_shards_under_low_linux_fd_limit(tmp_path):
    _, _, root = _dataset(tmp_path, rows=96, chunk=1)
    program = """
import json
from pathlib import Path
import resource
import sys

from nn_window_ablation.data import open_dataset, open_labels

root = Path(sys.argv[1])
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 24), hard))
def counts():
    return (len(list(Path('/proc/self/fd').iterdir())),
            sum(str(root) in line and '.npy' in line for line in Path('/proc/self/maps').read_text().splitlines()))

baseline = counts()
observed = {}
for opener, width in ((open_dataset, 4), (open_labels, 3)):
    dataset = opener(root)
    assert counts() == baseline
    peak = baseline
    for descriptor in dataset.shards:
        with dataset.open_shard(descriptor.index) as shard:
            now = counts()
            assert now[0] <= baseline[0] + width and now[1] == width
            assert len(shard.shot_ids) == 1
            peak = tuple(max(a, b) for a, b in zip(peak, now))
        assert counts() == baseline
    observed[opener.__name__] = peak
print(json.dumps({'baseline': baseline, 'peaks': observed}))
"""
    result = subprocess.run([sys.executable, "-c", program, str(root)], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    counts = json.loads(result.stdout)
    assert counts["peaks"]["open_dataset"][1] == 4 and counts["peaks"]["open_labels"][1] == 3
    (tmp_path / "fd-counts.json").write_text(result.stdout)
