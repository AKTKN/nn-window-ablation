"""Atomic training checkpoints with a single integrity-protected envelope."""

from hashlib import sha256
import json
from pathlib import Path
import pickle
from shutil import copyfileobj
from tempfile import SpooledTemporaryFile
from zipfile import BadZipFile, ZIP_STORED, ZipFile

import torch

from .artifacts import ArtifactError, atomic_file, canonical_json
from .models.gnn_rnn import QueryDecoder
from .training import distributed_state, final_update_count
from .timing import span


CHECKPOINT_SCHEMA = 1
STATE_MEMBER = "state.pt"
MANIFEST_MEMBER = "manifest.json"
REQUIRED_STATE = {
    "contract", "model", "optimizer_state", "scheduler_state", "random_states", "counters", "cursor", "last_update",
}


def _transfer(source, destination=None) -> dict:
    digest = sha256()
    size = 0
    for block in iter(lambda: source.read(1 << 20), b""):
        digest.update(block)
        size += len(block)
        if destination is not None:
            destination.write(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def save_checkpoint(path: Path, state: dict) -> None:
    """Keep the previous checkpoint available until its replacement is complete."""
    if not REQUIRED_STATE <= state.keys():
        raise ArtifactError("checkpoint lacks required training state")
    distributed = distributed_state(state)
    with SpooledTemporaryFile(max_size=16 << 20, dir=path.parent) as payload:
        with span("checkpoint_serialize"):
            torch.save(state, payload)
            payload.seek(0)
            integrity = _transfer(payload)
            payload.seek(0)
        with span("checkpoint_publish"), atomic_file(path) as file:
            with ZipFile(file, "w", compression=ZIP_STORED) as archive:
                with archive.open(STATE_MEMBER, "w", force_zip64=True) as member:
                    copyfileobj(payload, member, length=1 << 20)
                archive.writestr(MANIFEST_MEMBER, canonical_json({
                    "schema_version": CHECKPOINT_SCHEMA, "status": "complete", "state": integrity,
                    **({"distributed_state_schema": distributed["schema_version"]} if distributed is not None else {}),
                }))


def load_checkpoint(path: Path) -> dict:
    try:
        with ZipFile(path, "r") as archive:
            if sorted(archive.namelist()) != sorted((STATE_MEMBER, MANIFEST_MEMBER)):
                raise ArtifactError("checkpoint is incomplete or has unexpected members")
            manifest = json.loads(archive.read(MANIFEST_MEMBER))
            if manifest.get("schema_version") != CHECKPOINT_SCHEMA or manifest.get("status") != "complete":
                raise ArtifactError("checkpoint is incomplete or has an incompatible schema")
            with SpooledTemporaryFile(max_size=16 << 20) as payload:
                with archive.open(STATE_MEMBER) as member:
                    integrity = _transfer(member, payload)
                if integrity != manifest.get("state"):
                    raise ArtifactError("checkpoint state checksum or length differs from its manifest")
                payload.seek(0)
                state = torch.load(payload, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not REQUIRED_STATE <= state.keys():
            raise ArtifactError("checkpoint lacks required training state")
        distributed = distributed_state(state)
        if manifest.get("distributed_state_schema") != (None if distributed is None else distributed["schema_version"]):
            raise ArtifactError("checkpoint envelope disagrees with its distributed state schema")
        return state
    except ArtifactError:
        raise
    except (AttributeError, OSError, BadZipFile, TypeError, ValueError, EOFError, RuntimeError, pickle.UnpicklingError) as exc:
        raise ArtifactError(f"incomplete or corrupt checkpoint {path}") from exc


def load_evaluation_model(path: Path) -> QueryDecoder:
    """Load only the terminal model state for evaluation of fixed-budget weights."""
    state = load_checkpoint(path)
    final_update_count(state)
    return QueryDecoder.from_checkpoint(state["model"]).eval()
