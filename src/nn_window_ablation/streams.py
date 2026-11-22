"""Phase-scoped trajectory order and resumable chunk positions."""

from dataclasses import dataclass

import numpy as np

from .artifacts import ArtifactError
from .data import ROW_SPACE, MappedDataset, SamplingPlan, TrajectoryBatch, TrajectoryID, sample_chunk
from .simulation import CircuitBundle, ShotBatch


@dataclass(frozen=True, slots=True)
class ChunkCursor:
    chunk: int = 0
    row: int = 0


class TrajectoryStream:
    """An ordered training stream with stateless sampling and permutation per chunk."""

    def __init__(self, plan: SamplingPlan, bundle: CircuitBundle, dataset: MappedDataset | None = None):
        if plan.namespace.split != "train":
            raise ValueError("optimization requires a train split")
        if dataset is not None and dataset.plan != plan:
            raise ValueError("mapped training dataset differs from the sampling plan")
        plan.assert_circuit(bundle)
        self.plan = plan
        self.bundle = bundle
        self.dataset = dataset
        self.cursor = ChunkCursor()
        self._cached_index = None
        self._cached_shots = None
        self._permutation = None

    def record(self) -> dict:
        return {
            "plan": self.plan.record(),
            "dataset_manifest_sha256": self.dataset.manifest_sha256 if self.dataset is not None else None,
            "ordering": "pcg64_per_chunk_v1",
        }

    @property
    def position(self) -> int:
        return self.cursor.chunk * self.plan.chunk_rows + self.cursor.row

    @property
    def capacity(self) -> int:
        return ROW_SPACE if self.plan.total_rows is None else self.plan.total_rows

    def validate_cursor(self, cursor: ChunkCursor) -> int:
        if (
            type(cursor.chunk) is not int or type(cursor.row) is not int
            or cursor.chunk < 0 or not 0 <= cursor.row < self.plan.chunk_rows
        ):
            raise ArtifactError("invalid data chunk/row cursor")
        position = cursor.chunk * self.plan.chunk_rows + cursor.row
        if position > self.capacity:
            raise ArtifactError("data cursor exceeds the trajectory stream capacity")
        return position

    def seek(self, cursor: ChunkCursor) -> None:
        self.validate_cursor(cursor)
        self.cursor = cursor

    def _read_part(self, chunk: int, first: int, count: int) -> TrajectoryBatch:
        if chunk != self._cached_index:
            _, rows = self.plan.chunk_bounds(chunk)
            rng = np.random.Generator(np.random.PCG64(self.plan.namespace.seed("trajectory_order", chunk)))
            permutation = rng.permutation(rows)
            shots = sample_chunk(self.plan, self.bundle, chunk) if self.dataset is None else None
            self._permutation = permutation
            self._cached_shots = shots
            self._cached_index = chunk
        indices = self._permutation[first:first + count]
        if self.dataset is not None:
            return self.dataset.shards[chunk].read(indices)
        source = self._cached_shots
        shots = ShotBatch(
            source.detectors[indices], source.global_labels[indices], source.core_labels[indices],
            source.shot_ids[indices], None,
        )
        identities = tuple(TrajectoryID(self.plan.dataset_id, chunk, int(index)) for index in indices)
        return TrajectoryBatch(shots, identities)

    def take(self, count: int) -> TrajectoryBatch:
        if type(count) is not int or count < 1:
            raise ValueError("a minibatch must contain a positive number of trajectories")
        stop = self.position + count
        if stop > self.capacity:
            raise ValueError("training dataset cannot supply the complete trajectory batch")
        position = self.position
        pieces = []
        while position < stop:
            chunk, row = divmod(position, self.plan.chunk_rows)
            _, rows = self.plan.chunk_bounds(chunk)
            take = min(rows - row, stop - position)
            pieces.append(self._read_part(chunk, row, take))
            position += take
        if len(pieces) == 1:
            batch = pieces[0]
        else:
            arrays = {
                role: np.concatenate([getattr(piece.shots, role) for piece in pieces])
                for role in ("detectors", "global_labels", "core_labels", "shot_ids")
            }
            batch = TrajectoryBatch(
                ShotBatch(**arrays, snapshots=None), tuple(identity for piece in pieces for identity in piece.identities),
            )
        self.cursor = ChunkCursor(*divmod(stop, self.plan.chunk_rows))
        return batch
