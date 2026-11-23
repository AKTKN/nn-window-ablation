"""Immutable packed datasets and deterministic sampling identities."""

from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import mmap
import os

import numpy as np
from numpy.typing import NDArray
import stim

from .timing import span

from .artifacts import (
    ArtifactError, atomic_file, canonical_json, file_integrity, read_json, read_json_artifact, staged_directory, verify_file, write_json,
)
from .continuation import is_digest
from .simulation import (
    Arrival, ArrivalPhase, CircuitArtifact, CircuitBundle, CircuitSpec, CoreInterval, Detector, Measurement,
    Observable, ShotBatch, TemporalCut, sample_shots,
)


DATASET_SCHEMA = 1
SEED_RECIPE = 1
LABEL_SCHEMA = "clean_cut_core_parity_v1"
FEATURE_SCHEMA = "lange_cropped_groups_v1"
BITORDER = "little"
SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}
ROW_SPACE = 1 << 32
ARRAY_FILES = {
    "detectors": "detectors.npy", "global_labels": "global_observables.npy",
    "core_labels": "core_labels.npy", "shot_ids": "shot_ids.npy",
}
PHYSICAL_FILES = ("circuit_global.stim", "circuit_teacher.stim", "global.dem", "structure.json")


@dataclass(frozen=True, slots=True)
class StreamNamespace:
    master_seed: int
    family: str
    split: str
    distance: int
    replicate: int
    phase: str

    def __post_init__(self) -> None:
        if self.split not in SPLIT_CODES:
            raise ValueError("dataset split must be train, validation, or test")
        phases = ("base", "continuation") if self.split == "train" else ("held_out",)
        if self.phase not in phases:
            raise ValueError("dataset phase is incompatible with its split")
        if type(self.master_seed) is not int or not 0 <= self.master_seed < 2**64:
            raise ValueError("master seed must be an unsigned 64-bit integer")
        if type(self.replicate) is not int or self.replicate < 0 or not isinstance(self.family, str) or not self.family.strip():
            raise ValueError("a dataset needs a nonempty family and a nonnegative integer replicate")

    def seed(self, role: str, ordinal: int = 0) -> int:
        record = [SEED_RECIPE, asdict(self), role, ordinal]
        return int.from_bytes(sha256(canonical_json(record)).digest()[:8], "little")


@dataclass(frozen=True, slots=True)
class SamplingPlan:
    namespace: StreamNamespace
    spec: CircuitSpec
    original_sha256: str
    teacher_sha256: str
    stim_version: str
    detector_count: int
    core_count: int
    chunk_rows: int
    total_rows: int | None

    def __post_init__(self) -> None:
        if self.namespace.distance != self.spec.distance:
            raise ValueError("stream distance must match the circuit")
        if type(self.chunk_rows) is not int or not 1 <= self.chunk_rows <= ROW_SPACE:
            raise ValueError("chunk size must be a positive supported row count")
        if self.total_rows is not None and (type(self.total_rows) is not int or not 1 <= self.total_rows <= ROW_SPACE):
            raise ValueError("dataset row count must fit the supported identity range")
        if type(self.detector_count) is not int or self.detector_count < 1:
            raise ValueError("detector count must be positive")
        if self.core_count != self.spec.rounds // self.spec.core_length:
            raise ValueError("core count must match the physical partition")

    @classmethod
    def from_circuit(
        cls, bundle: CircuitBundle, namespace: StreamNamespace, *, chunk_rows: int, total_rows: int | None = None,
    ) -> "SamplingPlan":
        return cls(
            namespace, bundle.spec, bundle.original.sha256, bundle.teacher.sha256, bundle.stim_version,
            len(bundle.detectors), len(bundle.cores), chunk_rows, total_rows,
        )

    def record(self) -> dict:
        return {
            **asdict(self), "seed_recipe": SEED_RECIPE, "label_schema": LABEL_SCHEMA, "feature_schema": FEATURE_SCHEMA,
            "sampler": "stim.compile_detector_sampler/separate_observables",
            "bitorder": BITORDER,
        }

    @classmethod
    def from_record(cls, record: dict) -> "SamplingPlan":
        fields = dict(record)
        for key in ("seed_recipe", "label_schema", "feature_schema", "sampler", "bitorder"):
            fields.pop(key)
        fields["namespace"] = StreamNamespace(**fields["namespace"])
        fields["spec"] = CircuitSpec(**fields["spec"])
        plan = cls(**fields)
        if plan.record() != record:
            raise ArtifactError("incompatible sampling recipe or feature/label schema")
        return plan

    @property
    def dataset_id(self) -> str:
        return "ds-" + sha256(canonical_json(self.record())).hexdigest()

    @property
    def shot_prefix(self) -> int:
        # Int64 IDs are P1/P2 join keys. The complete identity uses the full
        # dataset digest, shard index and row, with no truncated digest.
        split = SPLIT_CODES[self.namespace.split]
        phase = int(self.namespace.phase == "continuation")
        digest = int(self.dataset_id[3:10], 16)
        return (split << 61) | (phase << 60) | (digest << 32)

    def chunk_bounds(self, index: int) -> tuple[int, int]:
        if type(index) is not int or index < 0:
            raise ValueError("shard index must be a nonnegative integer")
        start = index * self.chunk_rows
        limit = ROW_SPACE if self.total_rows is None else self.total_rows
        if start >= limit:
            raise IndexError("trajectory stream is exhausted")
        return start, min(self.chunk_rows, limit - start)

    def assert_circuit(self, bundle: CircuitBundle) -> None:
        if self != self.from_circuit(bundle, self.namespace, chunk_rows=self.chunk_rows, total_rows=self.total_rows):
            raise ArtifactError("circuit, simulator version, or geometry differs from the sampling recipe")


@dataclass(frozen=True, slots=True)
class TrajectoryID:
    dataset_id: str
    shard: int
    row: int


@dataclass(frozen=True, slots=True)
class TrajectoryBatch:
    shots: ShotBatch
    identities: tuple[TrajectoryID, ...]


def sample_chunk(plan: SamplingPlan, bundle: CircuitBundle, index: int) -> ShotBatch:
    plan.assert_circuit(bundle)
    if plan.stim_version != stim.__version__:
        raise ArtifactError(f"sampling recipe requires Stim {plan.stim_version}, found {stim.__version__}")
    start, count = plan.chunk_bounds(index)
    return sample_shots(
        bundle, count, seed=plan.namespace.seed("circuit", index), first_shot_id=plan.shot_prefix + start,
    )


def _shot_ids(plan: SamplingPlan, start: int, count: int) -> NDArray[np.int64]:
    return np.arange(start, start + count, dtype=np.int64) + plan.shot_prefix


def _validate_shots(plan: SamplingPlan, index: int, shots: ShotBatch) -> None:
    start, rows = plan.chunk_bounds(index)
    for role, shape in (
        ("detectors", (rows, plan.detector_count)), ("global_labels", (rows,)), ("core_labels", (rows, plan.core_count)),
    ):
        bits = getattr(shots, role)
        if bits.dtype != np.bool_ or bits.shape != shape:
            raise ArtifactError(f"wrong shape or bit dtype for {role}")
    if shots.shot_ids.dtype != np.int64 or not np.array_equal(shots.shot_ids, _shot_ids(plan, start, rows)):
        raise ArtifactError("wrong, duplicate or reordered shot IDs for shard")
    _check_label_parity(shots)


def _check_label_parity(shots: ShotBatch) -> None:
    if not np.array_equal(np.logical_xor.reduce(shots.core_labels, axis=1), shots.global_labels):
        raise ArtifactError("core label parity differs from global label")


def _bit_widths(plan: SamplingPlan) -> dict[str, int | None]:
    return {"detectors": plan.detector_count, "global_labels": 1, "core_labels": plan.core_count, "shot_ids": None}


def _shard_identity(plan: SamplingPlan, index: int) -> dict:
    start, count = plan.chunk_bounds(index)
    return {
        "schema_version": DATASET_SCHEMA, "status": "complete", "dataset_id": plan.dataset_id,
        "index": index, "start": start, "rows": count, "bitorder": BITORDER,
    }


def _write_shard(root: Path, plan: SamplingPlan, index: int, shots: ShotBatch) -> dict:
    with span("validate_rows"):
        _validate_shots(plan, index, shots)
    with staged_directory(root) as staging:
        records = {}
        for role, width in _bit_widths(plan).items():
            array = getattr(shots, role)
            if width is not None:
                with span("pack"):
                    array = np.packbits(array.reshape(len(shots.shot_ids), width), axis=1, bitorder=BITORDER)
            path = staging / ARRAY_FILES[role]
            with span("write_payload"), atomic_file(path) as file:
                np.save(file, array, allow_pickle=False)
            with span("checksum"):
                receipt = file_integrity(path)
            records[role] = {
                **receipt, "role": role, "filename": path.name, "dtype": array.dtype.name,
                "shape": list(array.shape), "valid_bits": width,
            }
        with span("write_shard_manifest"):
            write_json(staging / "shard.json", {**_shard_identity(plan, index), "arrays": records})
    with span("checksum"):
        return {"index": index, "manifest": file_integrity(root / "shard.json")}


def _write_physical(root: Path, bundle: CircuitBundle) -> dict:
    for name, content in (
        (PHYSICAL_FILES[0], bundle.original.text),
        (PHYSICAL_FILES[1], bundle.teacher.text),
        (PHYSICAL_FILES[2], str(bundle.global_detector_error_model(decompose_errors=True))),
    ):
        with atomic_file(root / name) as file:
            file.write(content.encode("utf-8"))
    record = {
        key: value for key, value in asdict(bundle).items()
        if key not in ("original", "teacher", "spec", "stim_version")
    }
    write_json(root / "structure.json", record)
    return {name: file_integrity(root / name) for name in PHYSICAL_FILES}


def write_dataset(
    path: Path, plan: SamplingPlan, bundle: CircuitBundle, *, provenance: dict,
    chunks: Iterable[ShotBatch] | None = None,
) -> None:
    """Stage a finite immutable dataset and publish its manifest last."""
    if plan.total_rows is None:
        raise ValueError("a persistent dataset requires a finite row count")
    plan.assert_circuit(bundle)
    count = (plan.total_rows + plan.chunk_rows - 1) // plan.chunk_rows
    input_batches = iter(chunks) if chunks is not None else (sample_chunk(plan, bundle, index) for index in range(count))
    with staged_directory(path) as staging:
        physical = _write_physical(staging, bundle)
        (staging / "shards").mkdir()
        inventory = []
        for index in range(count):
            try:
                shots = next(input_batches)
            except StopIteration as exc:
                raise ArtifactError("missing rows for the declared dataset") from exc
            inventory.append(_write_shard(staging / "shards" / str(index), plan, index, shots))
        if next(input_batches, None) is not None:
            raise ArtifactError("extra shards beyond the declared dataset")
        _write_dataset_manifest(staging, plan, physical, inventory, provenance)


def _write_dataset_manifest(staging: Path, plan: SamplingPlan, physical: dict, inventory: list[dict], provenance: dict):
    write_json(staging / "manifest.json", {
        "schema_version": DATASET_SCHEMA, "status": "complete", "dataset_id": plan.dataset_id,
        "recipe": plan.record(), "physical": physical, "shards": inventory, "provenance": provenance,
    })


@dataclass(frozen=True, slots=True)
class ArrayDescriptor:
    role: str
    path: Path
    shape: tuple[int, ...]
    dtype: str
    valid_bits: int | None
    bytes: int
    sha256: str

    @property
    def integrity(self) -> dict:
        return {"bytes": self.bytes, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class ShardDescriptor:
    index: int
    start: int
    rows: int
    directory: Path
    manifest_sha256: str
    manifest_bytes: int
    arrays: tuple[ArrayDescriptor, ...]

    def source_record(self) -> dict:
        return {
            "manifest": {"bytes": self.manifest_bytes, "sha256": self.manifest_sha256},
            "arrays": {array.path.name: array.integrity for array in self.arrays},
        }


@contextmanager
def _map_array(spec: ArrayDescriptor) -> Iterator[NDArray]:
    """Own a read-only mapping. Only the private shard reader gets its ndarray."""
    try:
        with spec.path.open("rb") as file:
            if os.fstat(file.fileno()).st_size != spec.bytes:
                raise ArtifactError(f"size or checksum mismatch for {spec.path.name}")
            version = np.lib.format.read_magic(file)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(file)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(file)
            else:
                raise ArtifactError(f"unsupported NPY header in {spec.path.name}")
            offset = file.tell()
            if (
                shape != spec.shape or dtype != np.dtype(spec.dtype)
                or offset + dtype.itemsize * spec.shape[0] * (spec.shape[1] if len(spec.shape) == 2 else 1) != spec.bytes
            ):
                raise ArtifactError(f"wrong shape, dtype or valid-bit count for {spec.role}")
            mapping = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
    except ArtifactError:
        raise
    except FileNotFoundError as exc:
        raise ArtifactError(f"missing or unreadable artifact {spec.path}") from exc
    except (ValueError, OSError, EOFError) as exc:
        raise ArtifactError(f"invalid NPY payload for {spec.path.name}: {exc}") from exc
    with mapping:
        if sha256(mapping).hexdigest() != spec.sha256:
            raise ArtifactError(f"size or checksum mismatch for {spec.path.name}")
        yield np.ndarray(shape, dtype=dtype, buffer=mapping, offset=offset, order="F" if fortran else "C")


class OpenShard:
    """Read independent arrays from one verified shard until it is closed."""

    def __init__(self, descriptor: ShardDescriptor, plan: SamplingPlan, *, labels_only: bool):
        self.descriptor = descriptor
        self.plan = plan
        self.index = descriptor.index
        self.rows = descriptor.rows
        self._stack = ExitStack()
        self._arrays = {}
        self._closed = False
        try:
            for spec in descriptor.arrays:
                if labels_only and spec.role == "detectors":
                    continue
                self._arrays[spec.role] = self._stack.enter_context(_map_array(spec))
            self._verify_rows()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._closed = True
        self._arrays.clear()
        self._stack.close()

    def _array(self, role: str) -> NDArray:
        if self._closed:
            raise RuntimeError("shard handle is closed")
        if role not in self._arrays:
            raise ValueError("detector payloads are unavailable in a labels-only shard")
        return self._arrays[role]

    def _verify_rows(self) -> None:
        for start in range(0, self.rows, 65536):
            stop = min(start + 65536, self.rows)
            rows = slice(start, stop)
            for spec in self.descriptor.arrays:
                width = spec.valid_bits
                if spec.role in self._arrays and width is not None and width % 8:
                    if np.any(self._array(spec.role)[rows, -1] >> (width % 8)):
                        raise ArtifactError(f"nonzero padding bits for {spec.role}")
            expected = _shot_ids(self.plan, self.descriptor.start + start, stop - start)
            if not np.array_equal(self._array("shot_ids")[rows], expected):
                raise ArtifactError("wrong, duplicate or reordered shot IDs for shard")
            parity = np.bitwise_count(self._array("core_labels")[rows]).sum(axis=1) & 1
            if not np.array_equal(parity, self._array("global_labels")[rows, 0]):
                raise ArtifactError("core label parity differs from global label")

    @property
    def shot_ids(self) -> NDArray[np.int64]:
        return self._array("shot_ids").copy()

    @property
    def mapped_arrays(self) -> int:
        return len(self._arrays)

    def read_labels(self, rows: slice | NDArray[np.int64] = slice(None)) -> tuple[NDArray, NDArray]:
        global_labels = np.unpackbits(
            self._array("global_labels")[rows], axis=1, count=1, bitorder=BITORDER,
        ).reshape(-1)
        core_labels = np.unpackbits(
            self._array("core_labels")[rows], axis=1, count=self.plan.core_count, bitorder=BITORDER,
        )
        return global_labels.astype(bool), core_labels.astype(bool)

    def read(self, rows: slice | NDArray[np.int64] = slice(None)) -> TrajectoryBatch:
        ids = self._array("shot_ids")[rows].copy()
        detectors = np.unpackbits(
            self._array("detectors")[rows], axis=1, count=self.plan.detector_count, bitorder=BITORDER,
        ).astype(bool)
        global_labels, core_labels = self.read_labels(rows)
        shots = ShotBatch(detectors, global_labels, core_labels, ids, None)
        identities = tuple(
            TrajectoryID(self.plan.dataset_id, self.index, int(shot) - self.plan.shot_prefix - self.descriptor.start)
            for shot in ids
        )
        return TrajectoryBatch(shots, identities)


@dataclass(slots=True)
class MappedDataset:
    plan: SamplingPlan
    bundle: CircuitBundle | None
    shards: tuple[ShardDescriptor, ...]
    manifest_sha256: str
    _active: bool = False

    @contextmanager
    def open_shard(self, index: int, *, labels_only: bool = False) -> Iterator[OpenShard]:
        if type(index) is not int or not 0 <= index < len(self.shards):
            raise IndexError("shard index is outside the dataset inventory")
        if self._active:
            raise RuntimeError("close the active dataset shard before opening another")
        self._active = True
        try:
            descriptor = self.shards[index]
            verify_file(
                descriptor.directory / "shard.json",
                {"bytes": descriptor.manifest_bytes, "sha256": descriptor.manifest_sha256},
            )
            handle = OpenShard(descriptor, self.plan, labels_only=labels_only or self.bundle is None)
            try:
                yield handle
            finally:
                handle.close()
        finally:
            self._active = False

    def verify_payloads(self) -> None:
        """Verify every shard sequentially, including labels and exact row identities."""
        for shard in self.shards:
            with self.open_shard(shard.index):
                pass


def _describe_shard(root: Path, plan: SamplingPlan, index: int, inventory: dict) -> ShardDescriptor:
    if type(inventory["index"]) is not int or inventory["index"] != index:
        raise ArtifactError("dataset shard identities are duplicated or out of order")
    path = root / "shard.json"
    record, manifest = read_json_artifact(path)
    if manifest != inventory["manifest"]:
        raise ArtifactError(f"size or checksum mismatch for {path.name}")
    if {key: value for key, value in record.items() if key != "arrays"} != _shard_identity(plan, index):
        raise ArtifactError("shard identity or schema differs from the dataset manifest")
    if set(record["arrays"]) != set(ARRAY_FILES):
        raise ArtifactError("shard must contain exactly the declared array roles")
    arrays = []
    start, count = plan.chunk_bounds(index)
    for role, width in _bit_widths(plan).items():
        entry = record["arrays"][role]
        filename = ARRAY_FILES[role]
        if entry["role"] != role or entry["filename"] != filename:
            raise ArtifactError(f"reassigned artifact identity for {role}")
        dtype = "uint8" if width is not None else "int64"
        shape = (count, (width + 7) // 8) if width is not None else (count,)
        if entry["valid_bits"] != width or entry["dtype"] != dtype or entry["shape"] != list(shape):
            raise ArtifactError(f"wrong shape, dtype or valid-bit count for {role}")
        length, digest = entry["bytes"], entry["sha256"]
        if type(length) is not int or length <= 0 or not is_digest(digest):
            raise ArtifactError(f"invalid payload integrity record for {role}")
        arrays.append(ArrayDescriptor(role, root / filename, shape, dtype, width, length, digest))
    return ShardDescriptor(index, start, count, root, manifest["sha256"], manifest["bytes"], tuple(arrays))


def _arrival(record: dict) -> Arrival:
    return Arrival(record["round"], ArrivalPhase(record["phase"]))


def _read_physical(path: Path, plan: SamplingPlan, records: dict) -> CircuitBundle:
    if set(records) != set(PHYSICAL_FILES):
        raise ArtifactError("incomplete circuit artifact inventory")
    for name in PHYSICAL_FILES:
        verify_file(path / name, records[name])
    structure = read_json(path / "structure.json")
    original = CircuitArtifact((path / PHYSICAL_FILES[0]).read_text())
    teacher = CircuitArtifact((path / PHYSICAL_FILES[1]).read_text())
    measurements = tuple(Measurement(**{**m, "arrival": _arrival(m["arrival"])}) for m in structure["measurements"])
    detectors = tuple(
        Detector(**{**d, "coordinates": tuple(d["coordinates"]), "measurement_indices": tuple(d["measurement_indices"]),
                    "arrival": _arrival(d["arrival"])})
        for d in structure["detectors"]
    )
    observables = tuple(Observable(**{
        **o, "logical_support": tuple(o["logical_support"]), "measurement_indices": tuple(o["measurement_indices"]),
    }) for o in structure["observables"])
    bundle = CircuitBundle(
        plan.spec, plan.stim_version, original, teacher, measurements, detectors, observables,
        tuple(CoreInterval(**c) for c in structure["cores"]), tuple(TemporalCut(**c) for c in structure["cuts"]),
        structure["preparation_stop"], tuple(structure["round_measurement_instructions"]),
        structure["destructive_readout_instruction"],
    )
    plan.assert_circuit(bundle)
    return bundle


def open_dataset(path: Path, *, expected_split: str | None = None) -> MappedDataset:
    """Open dataset metadata. Payloads are verified when their shard is opened."""
    return _open_dataset(path, expected_split, labels_only=False)


def open_labels(path: Path, *, expected_split: str | None = None) -> MappedDataset:
    """Open metadata for labels and IDs without requiring detector or circuit files."""
    return _open_dataset(path, expected_split, labels_only=True)


def _open_dataset(path: Path, expected_split: str | None, *, labels_only: bool):
    manifest, receipt = read_json_artifact(path / "manifest.json")
    if (type(manifest.get("schema_version")) is not int or manifest["schema_version"] != DATASET_SCHEMA
            or manifest.get("status") != "complete"):
        raise ArtifactError("incomplete dataset or incompatible dataset schema")
    if not isinstance(manifest.get("provenance"), dict):
        raise ArtifactError("dataset manifest lacks a provenance record")
    try:
        plan = SamplingPlan.from_record(manifest["recipe"])
        if manifest["dataset_id"] != plan.dataset_id or plan.total_rows is None:
            raise ArtifactError("dataset identity or row count differs from its recipe")
        if expected_split is not None and plan.namespace.split != expected_split:
            raise ArtifactError(f"expected {expected_split} dataset, found {plan.namespace.split}")
        count = (plan.total_rows + plan.chunk_rows - 1) // plan.chunk_rows
        if len(manifest["shards"]) != count:
            raise ArtifactError("dataset shard inventory does not match its complete row count")
        if not isinstance(manifest["physical"], dict) or set(manifest["physical"]) != set(PHYSICAL_FILES):
            raise ArtifactError("incomplete circuit artifact inventory")
        shards = tuple(_describe_shard(path / "shards" / str(index), plan, index, entry)
                       for index, entry in enumerate(manifest["shards"]))
        bundle = None if labels_only else _read_physical(path, plan, manifest["physical"])
        return MappedDataset(plan, bundle, shards, receipt["sha256"])
    except ArtifactError:
        raise
    except (KeyError, TypeError, ValueError, OSError, EOFError) as exc:
        raise ArtifactError(f"invalid dataset manifest or shard payload at {path}") from exc
