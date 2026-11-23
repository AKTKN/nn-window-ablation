"""Phase-scoped trajectory order and resumable chunk positions."""

from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from contextlib import ExitStack

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
        self._orders = OrderedDict()
        self._shard = None
        self._owner = ExitStack()
        self._closed = False

    def __enter__(self) -> "TrajectoryStream":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def _release_chunk(self) -> None:
        self._owner.close()
        self._shard = self._cached_shots = self._cached_index = None

    def close(self) -> None:
        self._closed = True
        self._release_chunk()
        self._orders.clear()

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
        if cursor.chunk != self._cached_index:
            self._release_chunk()
        self.cursor = cursor

    def _order(self, chunk: int):
        if chunk not in self._orders:
            _, rows = self.plan.chunk_bounds(chunk)
            rng = np.random.Generator(np.random.PCG64(self.plan.namespace.seed("trajectory_order", chunk)))
            self._orders[chunk] = rng.permutation(rows)
            if len(self._orders) > 2:
                self._orders.popitem(last=False)
        self._orders.move_to_end(chunk)
        return self._orders[chunk]

    def _batch_end(self, count: int) -> int:
        if self._closed:
            raise RuntimeError("trajectory stream is closed")
        if type(count) is not int or count < 1:
            raise ValueError("a minibatch must contain a positive number of trajectories")
        stop = self.position + count
        if stop > self.capacity:
            raise ValueError("training dataset cannot supply the complete trajectory batch")
        return stop

    def _segments(self, stop: int) -> Iterator[tuple[int, int, int]]:
        position = self.position
        while position < stop:
            chunk, row = divmod(position, self.plan.chunk_rows)
            _, rows = self.plan.chunk_bounds(chunk)
            count = min(rows - row, stop - position)
            yield chunk, row, count
            position += count

    def planned_identities(self, count: int) -> tuple[TrajectoryID, ...]:
        """Preview the physical row identities without opening or sampling a shard."""
        stop = self._batch_end(count)
        return tuple(
            TrajectoryID(self.plan.dataset_id, chunk, int(index))
            for chunk, row, size in self._segments(stop) for index in self._order(chunk)[row:row + size]
        )

    def _read_part(self, chunk: int, first: int, count: int) -> TrajectoryBatch:
        if chunk != self._cached_index:
            self._release_chunk()
            if self.dataset is None:
                self._cached_shots = sample_chunk(self.plan, self.bundle, chunk)
            else:
                self._shard = self._owner.enter_context(self.dataset.open_shard(chunk))
            self._cached_index = chunk
        indices = self._order(chunk)[first:first + count]
        if self.dataset is not None:
            return self._shard.read(indices)
        source = self._cached_shots
        shots = ShotBatch(
            source.detectors[indices], source.global_labels[indices], source.core_labels[indices],
            source.shot_ids[indices], None,
        )
        identities = tuple(TrajectoryID(self.plan.dataset_id, chunk, int(index)) for index in indices)
        return TrajectoryBatch(shots, identities)

    def take(self, count: int) -> TrajectoryBatch:
        if self._closed:
            raise RuntimeError("trajectory stream is closed")
        try:
            return self._take(count)
        except BaseException:
            self._release_chunk()
            raise

    def _take(self, count: int) -> TrajectoryBatch:
        stop = self._batch_end(count)
        pieces = [self._read_part(chunk, row, size) for chunk, row, size in self._segments(stop)]
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
        if self.dataset is not None and (self.cursor.chunk != self._cached_index or stop == self.capacity):
            self._release_chunk()
        return batch
