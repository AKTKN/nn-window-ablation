import json
import tarfile

import numpy as np
import pytest

from nn_window_ablation.artifacts import ArtifactError
from nn_window_ablation.export import export_bundle
from tests.p7_checks import digest, truth_artifacts, write_json


def test_p7_export_inventory_binds_exact_artifact_bytes(truth_artifacts, tmp_path):
    dataset, prediction, checkpoint, report, failures = truth_artifacts
    log = tmp_path / "stdout.log"
    log.write_text("fixture job finished\n")
    output = export_bundle(dataset, report, tmp_path / "stats.tar.gz", logs=(log,))
    with tarfile.open(output, "r:gz") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        assert set(archive.getnames()) == set(manifest["files"]) | {"manifest.json"}
        assert manifest["capabilities"] == {
            "statistics": True, "redecode": False, "detectors": False, "checkpoints": False, "training_shards": False,
        }
        from hashlib import sha256
        from io import BytesIO

        for name, receipt in manifest["files"].items():
            payload = archive.extractfile(name).read()
            assert receipt == {"bytes": len(payload), "sha256": sha256(payload).hexdigest()}
            assert not name.endswith(("checkpoint.pt", "detectors.npy", ".stim"))
        label = np.load(BytesIO(archive.extractfile("dataset/shards/0/global_observables.npy").read()), allow_pickle=False)
        np.testing.assert_array_equal(label[:, 0], [1, 0, 1, 0])
        packed = np.load(BytesIO(archive.extractfile("predictions/0/shards/0/S.decisions.npy").read()), allow_pickle=False)
        assert packed.shape == (4, 2)
        assert archive.extractfile(manifest["logs"][0]).read() == log.read_bytes()
        assert manifest["runs"][0]["checkpoint_sha256"] == "a" * 64
        assert manifest["runs"][0]["checkpoint"] is None
        saved = json.load(archive.extractfile(manifest["report"]))
        assert saved["rates"]["S"]["failures"] == int(failures["S"].sum())
    assert report.is_file() and prediction.is_file() and dataset.is_dir() and not checkpoint.exists()


@pytest.mark.parametrize(("damage", "message"), [
    ("missing", "missing or unreadable"), ("truncated", "size or checksum mismatch"),
    ("changed", "size or checksum mismatch"), ("labels", "label/feature schema differs"),
    ("ids", "shot identities differ or are reordered"), ("schema", "incompatible prediction manifest"),
    ("unfinished", "incomplete, failed, or incompatible run record"),
])
def test_p7_rejects_incomplete_or_scientifically_incompatible_sources(truth_artifacts, tmp_path, damage, message):
    dataset, prediction, checkpoint, report, _ = truth_artifacts
    labels = dataset / "shards/0/core_labels.npy"
    if damage == "missing":
        labels.unlink()
    elif damage == "truncated":
        labels.write_bytes(labels.read_bytes()[:-7])
    elif damage == "changed":
        contents = labels.read_bytes()
        labels.write_bytes(contents[:-1] + bytes([contents[-1] ^ 1]))
    elif damage == "unfinished":
        write_json(checkpoint.with_name("run.json"), {"schema_version": 1, "status": "running"})
    else:
        manifest = json.loads(prediction.read_text())
        if damage == "labels":
            manifest["data"]["label_schema"] = "not-the-original-labels"
        elif damage == "schema":
            manifest["schema_version"] = 99
        else:
            ids = prediction.parent / "shards/0/shot_ids.npy"
            np.save(ids, np.load(ids, allow_pickle=False)[::-1], allow_pickle=False)
            manifest["shards"][0]["files"]["shot_ids.npy"].update(digest(ids))
        write_json(prediction, manifest)
        source = json.loads(report.read_text())
        source["inputs"][0].update(digest(prediction))
        write_json(report, source)
    with pytest.raises(ArtifactError, match=message):
        export_bundle(dataset, report, tmp_path / "bad.tar.gz")
    assert not (tmp_path / "bad.tar.gz").exists()
