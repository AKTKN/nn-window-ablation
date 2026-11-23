"""Exercise the NFS open-unlink rule against the production staging contexts."""

import fcntl
import os
from pathlib import Path

import pytest

from nn_window_ablation import artifacts, generation, paired_workflows


@pytest.fixture
def nfs_unlink(monkeypatch):
    """Silly-rename an unlinked, locally open inode, including rmtree's dir_fd calls."""
    unlink = os.unlink
    hidden = []

    def unlink_open_file(path, *, dir_fd=None):
        target = Path(path) if dir_fd is None else Path(f"/proc/self/fd/{dir_fd}") / path
        if target.name == "writers.lock":
            for fd in Path("/proc/self/fd").iterdir():
                try:
                    same_inode = target.samefile(fd)
                except FileNotFoundError:
                    continue
                if same_inode:
                    tombstone = target.resolve().with_name(".nfs-test-open-lease")
                    target.rename(tombstone)
                    hidden.append(tombstone)
                    return
        unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "unlink", unlink_open_file)
    yield hidden
    for path in hidden:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("operation", ["generation", "evaluation"])
def test_quiescent_stage_can_be_removed_with_nfs_unlink(tmp_path, nfs_unlink, operation):
    attempt = "a" * 32
    logs = tmp_path / "attempts" / attempt
    logs.mkdir(parents=True)
    if operation == "generation":
        context = generation._owned_staging(tmp_path / ".sample.generation", logs)
    else:
        context = paired_workflows._task_staging(tmp_path / ".evaluation-staging", logs)
    with context as stage:
        (stage / "partial").touch()
    assert not stage.exists()
    assert not nfs_unlink


def test_retiring_legacy_stage_preserves_live_writers_and_can_retry(tmp_path, monkeypatch, nfs_unlink):
    stage = tmp_path / ".sample.generation"
    stage.mkdir()
    lease = stage / "writers.lock"
    lease.touch()  # Pre-fix attempts used an empty lease.
    payload = stage / "dataset"
    payload.mkdir()
    retired = tmp_path / "previous-writers.lock"
    with lease.open("rb") as worker:
        fcntl.flock(worker, fcntl.LOCK_SH)
        with pytest.raises(artifacts.ArtifactError, match="still has live writers"):
            artifacts.retire_staging(stage, "generation", retired)
        assert lease.exists() and payload.exists() and not retired.exists()

    def interrupted(directory):
        assert directory == stage and not lease.exists()
        with retired.open("rb") as competitor, pytest.raises(BlockingIOError):
            fcntl.flock(competitor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        raise OSError("interrupted after lease rename")

    with monkeypatch.context() as patch:
        patch.setattr(artifacts.shutil, "rmtree", interrupted)
        with pytest.raises(OSError, match="interrupted after lease rename"):
            artifacts.retire_staging(stage, "generation", retired)
    assert payload.exists()
    artifacts.retire_staging(stage, "generation", retired)
    assert not stage.exists() and not nfs_unlink


def test_late_worker_cannot_join_reused_generation_directory(tmp_path, monkeypatch):
    stage = tmp_path / ".sample.generation"
    stage.mkdir()
    lease = stage / "writers.lock"
    artifacts.create_writer_lease(stage, "first")
    flock = fcntl.flock

    def retire_between_open_and_lock(file, flags):
        if flags & fcntl.LOCK_SH:
            artifacts.retire_staging(stage, "generation", tmp_path / "first-retired.lock")
            stage.mkdir()
            artifacts.create_writer_lease(stage, "second")
        return flock(file, flags)

    with monkeypatch.context() as patch:
        patch.setattr(fcntl, "flock", retire_between_open_and_lock)
        with pytest.raises(artifacts.ArtifactError, match="no longer belongs"):
            artifacts.open_writer_lease(lease, "first")
    with pytest.raises(artifacts.ArtifactError, match="no longer belongs"):
        artifacts.open_writer_lease(lease, "first")
    with artifacts.open_writer_lease(lease, "second"):
        with pytest.raises(artifacts.ArtifactError, match="still has live writers"):
            artifacts.retire_staging(stage, "generation", tmp_path / "second-retired.lock")


def test_clean_legacy_evaluation_stage_without_attempt_logs(tmp_path, nfs_unlink):
    root = tmp_path / ".evaluation-staging"
    stage = root / ("1" * 32)
    stage.mkdir(parents=True)
    (stage / "writers.lock").touch()
    paired_workflows._clean_staging(root)
    assert not root.exists() and not nfs_unlink


def test_retirement_preserves_existing_archive(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    artifacts.create_writer_lease(stage, "current")
    retired = tmp_path / "archived.lock"
    retired.write_text("prior")
    with pytest.raises(artifacts.ArtifactError, match="already exists"):
        artifacts.retire_staging(stage, "generation", retired)
    assert retired.read_text() == "prior"
    assert (stage / "writers.lock").read_text() == "current"
