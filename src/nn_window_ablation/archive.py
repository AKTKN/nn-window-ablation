"""Verified packaging and staged extraction of portable analysis bundles."""

from contextlib import nullcontext
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
from tempfile import NamedTemporaryFile

from .artifacts import ArtifactError, _sync_directory, atomic_file, read_json, staged_directory, verify_file
from .continuation import is_digest


BUNDLE_SCHEMA = 1
MANIFEST = "manifest.json"
BLOCK_SIZE = 1 << 20


def member_path(name: str) -> PurePosixPath:
    """Accept only canonical, relative paths shared across ordinary filesystems."""
    if (
        type(name) is not str or not name or "\\" in name or ":" in name
        or any(ord(char) < 32 for char in name)
        or any(part in ("", ".", "..") for part in name.split("/"))
    ):
        raise ArtifactError(f"unsafe bundle member path {name!r}")
    return PurePosixPath(name)


def inventory(record: dict) -> dict:
    if (type(record.get("schema_version")) is not int or record["schema_version"] != BUNDLE_SCHEMA
            or record.get("status") != "complete"):
        raise ArtifactError("incomplete bundle manifest or unknown bundle schema")
    files = record.get("files")
    if not isinstance(files, dict) or not files or MANIFEST in files:
        raise ArtifactError("incomplete bundle file inventory")
    paths = set(files)
    for name, receipt in files.items():
        path = member_path(name)
        if any(parent.as_posix() in paths for parent in path.parents):
            raise ArtifactError("bundle member conflicts with a parent file")
        if (not isinstance(receipt, dict) or set(receipt) != {"bytes", "sha256"}
                or type(receipt["bytes"]) is not int or receipt["bytes"] < 0 or not is_digest(receipt["sha256"])):
            raise ArtifactError(f"invalid integrity receipt for {name}")
    return files


def verify_bundle(root: Path) -> dict:
    if (root / MANIFEST).is_symlink():
        raise ArtifactError("bundle contains a symbolic link")
    bundle = read_json(root / MANIFEST)
    files = inventory(bundle)
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ArtifactError("bundle contains a link or a nonregular file")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(files) | {MANIFEST}:
        raise ArtifactError("bundle contains missing or unexpected files")
    for name, receipt in files.items():
        verify_file(root / member_path(name), receipt)
    return bundle


def _regular(member: tarfile.TarInfo) -> None:
    member_path(member.name)
    if member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE) or member.issparse():
        raise ArtifactError(f"unsupported archive member type for {member.name}")


def _scan_archive(source, destination: Path | None = None) -> dict:
    try:
        with gzip.GzipFile(fileobj=source, mode="rb") as compressed:
            with tarfile.open(fileobj=compressed, mode="r|") as archive:
                first = archive.next()
                if first is None:
                    raise ArtifactError("archive is empty")
                _regular(first)
                if first.name != MANIFEST or first.size > 64 * BLOCK_SIZE:
                    raise ArtifactError("archive must begin with the bundle manifest")
                raw = archive.extractfile(first).read()
                bundle = json.loads(raw)
                if not isinstance(bundle, dict):
                    raise ArtifactError("invalid bundle manifest")
                files = inventory(bundle)
                if destination is not None:
                    with atomic_file(destination / MANIFEST) as file:
                        file.write(raw)
                seen = {MANIFEST}
                for member in iter(archive.next, None):
                    _regular(member)
                    name = member.name
                    if name in seen:
                        raise ArtifactError(f"duplicate archive member {name}")
                    seen.add(name)
                    if name not in files or member.size != files[name]["bytes"]:
                        raise ArtifactError(f"unexpected archive member or size for {name}")
                    path = None if destination is None else destination / member_path(name)
                    if path is not None:
                        path.parent.mkdir(parents=True, exist_ok=True)
                    digest = sha256()
                    size = 0
                    with atomic_file(path) if path is not None else nullcontext(None) as output:
                        with archive.extractfile(member) as payload:
                            for block in iter(lambda: payload.read(BLOCK_SIZE), b""):
                                size += len(block)
                                digest.update(block)
                                if output is not None:
                                    output.write(block)
                        if {"bytes": size, "sha256": digest.hexdigest()} != files[name]:
                            raise ArtifactError(f"size or checksum mismatch for {name}")
                if seen != set(files) | {MANIFEST}:
                    raise ArtifactError("archive is missing declared files")
            # Tar parsing can stop before gzip verifies the CRC and size trailer.
            while compressed.read(BLOCK_SIZE):
                pass
        return bundle
    except ArtifactError:
        raise
    except (EOFError, OSError, ValueError, tarfile.TarError) as exc:
        raise ArtifactError("truncated, corrupt, or unsupported bundle archive") from exc


def pack_bundle(root: Path, output: Path) -> Path:
    """Publish a verified new archive and leave every source file intact."""
    if os.path.lexists(output):
        raise FileExistsError(output)
    bundle = verify_bundle(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.") as temporary:
        with tarfile.open(fileobj=temporary, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
            for name in [MANIFEST, *sorted(bundle["files"])]:
                with (root / member_path(name)).open("rb") as source:
                    member = tarfile.TarInfo(name)
                    member.size = os.fstat(source.fileno()).st_size
                    archive.addfile(member, source)
        temporary.flush()
        temporary.seek(0)
        if _scan_archive(temporary) != bundle:
            raise ArtifactError("archive manifest changed during publication")
        os.fsync(temporary.fileno())
        os.link(temporary.name, output)
        _sync_directory(output.parent)
    return output


def extract_bundle(source: Path, destination: Path) -> Path:
    """Extract into a staged directory and reject links, aliases and corrupt bytes."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with staged_directory(destination) as staging:
        with source.open("rb") as archive:
            _scan_archive(archive, staging)
        verify_bundle(staging)
    return destination
