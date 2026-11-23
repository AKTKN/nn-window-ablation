from hashlib import sha256
from io import BytesIO
import json
import tarfile

import pytest

from nn_window_ablation.archive import extract_bundle, pack_bundle
from nn_window_ablation.artifacts import ArtifactError


def _record(content):
    return {"schema_version": 1, "status": "complete", "files": {
        "labels/tiny.npy": {"bytes": len(content), "sha256": sha256(content).hexdigest()},
    }}


def test_p7_archive_roundtrip_is_relative_and_preserves_source(tmp_path):
    content = b"some label bytes"
    source = tmp_path / "cluster name"
    source.mkdir()
    (source / "labels").mkdir()
    (source / "labels/tiny.npy").write_bytes(content)
    record = _record(content)
    (source / "manifest.json").write_text(json.dumps(record))
    archive = pack_bundle(source, tmp_path / "transfer.tar.gz")
    with tarfile.open(archive, "r:gz") as reader:
        assert reader.getnames() == ["manifest.json", "labels/tiny.npy"]
        assert json.load(reader.extractfile("manifest.json")) == record
    local = extract_bundle(archive, tmp_path / "different local root")
    assert (local / "labels/tiny.npy").read_bytes() == content
    assert (source / "labels/tiny.npy").read_bytes() == content
    truncated = tmp_path / "truncated.tar.gz"
    truncated.write_bytes(archive.read_bytes()[:-4])
    with pytest.raises(ArtifactError, match="truncated|corrupt"):
        extract_bundle(truncated, tmp_path / "partial")
    assert not (tmp_path / "partial").exists()


@pytest.mark.parametrize(("name", "kind", "message"), [
    ("../escape", tarfile.REGTYPE, "unsafe"),
    ("/absolute", tarfile.REGTYPE, "unsafe"),
    ("labels/../escape", tarfile.REGTYPE, "unsafe"),
    ("labels//tiny.npy", tarfile.REGTYPE, "unsafe"),
    ("labels/tiny.npy", tarfile.REGTYPE, "duplicate"),
    ("labels/symlink", tarfile.SYMTYPE, "unsupported archive member"),
    ("labels/hardlink", tarfile.LNKTYPE, "unsupported archive member"),
])
def test_p7_rejects_unsafe_archive_members_without_publishing(tmp_path, name, kind, message):
    def add(archive, path, value, type_=tarfile.REGTYPE):
        item = tarfile.TarInfo(path)
        item.type = type_
        item.linkname = "../../escape" if type_ != tarfile.REGTYPE else ""
        item.size = len(value)
        archive.addfile(item, BytesIO(value))

    source = tmp_path / "bad.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        add(archive, "manifest.json", json.dumps(_record(b"truth")).encode())
        add(archive, "labels/tiny.npy", b"truth")
        add(archive, name, b"truth" if kind == tarfile.REGTYPE else b"", kind)
    before = set(tmp_path.iterdir())
    with pytest.raises(ArtifactError, match=message):
        extract_bundle(source, tmp_path / "failed")
    assert set(tmp_path.iterdir()) == before
