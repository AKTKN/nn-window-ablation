"""Filesystem primitives for completed, checksummed experiment artifacts."""

from collections.abc import Iterator
from contextlib import contextmanager
import fcntl
from hashlib import sha256
import json
import os
import shutil
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
    return read_json_artifact(path)[0]


def read_json_artifact(path: Path) -> tuple[dict, dict]:
    try:
        payload = path.read_bytes()
        record = json.loads(payload)
        if not isinstance(record, dict):
            raise ValueError("expected a JSON object")
        return record, {"bytes": len(payload), "sha256": sha256(payload).hexdigest()}
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"missing or invalid manifest {path}") from exc


@contextmanager
def exclusive_lock(path: Path, operation: str) -> Iterator[None]:
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ArtifactError(f"another process is updating the selected {operation} run") from exc
        yield


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
        publish_directory(staging, path)


def publish_directory(staging: Path, path: Path) -> None:
    _sync_directory(staging)
    if os.path.lexists(path):
        raise FileExistsError(path)
    os.rename(staging, path)
    _sync_directory(path.parent)


def create_writer_lease(stage: Path, attempt: str) -> None:
    with (stage / "writers.lock").open("x", encoding="ascii") as file:
        file.write(attempt)


def open_writer_lease(path: Path, attempt: str) -> BinaryIO:
    file = path.open("rb")
    try:
        fcntl.flock(file, fcntl.LOCK_SH | fcntl.LOCK_NB)
        # An initializer may have paused between open and flock, or a new
        # generation may now own the same staging path.
        if file.read() != attempt.encode("ascii") or not os.path.samefile(path, file.fileno()):
            raise ArtifactError("worker lease no longer belongs to this attempt")
        return file
    except BaseException:
        file.close()
        raise


def retire_staging(stage: Path, operation: str, retired_lease: Path) -> None:
    """Fence writers and remove the stage under the coordinator's run lock."""
    lease = stage / "writers.lock"
    if lease.is_symlink() or (lease.exists() and not lease.is_file()):
        raise ArtifactError(f"invalid {operation} worker lease")
    if not lease.exists():
        # A coordinator can die before creating the lease or while retiring
        # an already quiescent attempt. No worker opens the lease with O_CREAT.
        shutil.rmtree(stage)
        return
    with lease.open("r+b") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ArtifactError(f"{operation} attempt {stage.name} still has live writers; retry after they exit") from exc
        # Renaming on the same filesystem keeps the locked inode intact. No
        # new worker can open it by its staging name, and NFS has no open
        # lease to silly-rename when rmtree unlinks the staged files.
        # Old attempts may have lost their logs while retaining the stage.
        retired_lease.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(retired_lease):
            raise ArtifactError(f"retired {operation} lease already exists: {retired_lease}")
        lease.rename(retired_lease)
        shutil.rmtree(stage)
