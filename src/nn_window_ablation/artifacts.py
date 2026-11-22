"""Filesystem primitives for completed, checksummed experiment artifacts."""

from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import BinaryIO


class ArtifactError(ValueError):
    """A persisted artifact is incomplete, incompatible, or corrupt."""


def canonical_json(record: object) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def file_integrity(path: Path) -> dict[str, int | str]:
    with path.open("rb") as file:
        digest = sha256()
        size = 0
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def verify_file(path: Path, expected: object) -> None:
    try:
        actual = file_integrity(path)
    except OSError as exc:
        raise ArtifactError(f"missing or unreadable artifact {path}") from exc
    if actual != expected:
        raise ArtifactError(f"size or checksum mismatch for {path.name}")


def read_json(path: Path) -> dict:
    try:
        record = json.loads(path.read_bytes())
        if not isinstance(record, dict):
            raise ValueError("expected a JSON object")
        return record
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"missing or invalid manifest {path}") from exc


def _sync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def atomic_file(path: Path) -> Iterator[BinaryIO]:
    """Publish a single file only after writing and syncing its complete contents."""
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as file:
            temporary = Path(file.name)
            yield file
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path: Path, record: object) -> None:
    with atomic_file(path) as file:
        file.write(canonical_json(record) + b"\n")


@contextmanager
def staged_directory(path: Path) -> Iterator[Path]:
    """Publish a new nonempty directory on the same filesystem as its staging area."""
    if os.path.lexists(path):
        raise FileExistsError(path)
    with TemporaryDirectory(dir=path.parent, prefix=f".{path.name}.") as directory:
        staging = Path(directory)
        yield staging
        _sync_directory(staging)
        if os.path.lexists(path):
            raise FileExistsError(path)
        os.rename(staging, path)
        _sync_directory(path.parent)
