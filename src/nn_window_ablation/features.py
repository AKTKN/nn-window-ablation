"""Lange event graphs built exclusively from cropped physical observations.

The coordinate, edge and tiling conventions are adapted from Lange et al.,
gnnrnn/graphs.py and gnnrnn/data.py at 458210ac74ed0047760b8789c169e6b9eff732b8.
See the packaged LANGE_LICENSE for the upstream copyright and MIT license.
"""

from dataclasses import dataclass
from functools import cache
from itertools import accumulate

import numpy as np
from numpy.typing import NDArray
import stim

from .simulation import ShotBatch
from .windows import CroppedObservations, ObservationLayout, QueryMetadata, SlotState


@dataclass(frozen=True, slots=True)
class SpatialTile:
    path: tuple[tuple[int, int], ...]
    detector_indices: NDArray[np.int64]
    xy: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class FeatureGeometry:
    distance: int
    detector_coordinates: tuple[tuple[float, ...], ...]
    tiles: tuple[SpatialTile, ...]


@dataclass(frozen=True, slots=True)
class TemporalGroup:
    first_layer: int
    second_layer: int


@dataclass(frozen=True, slots=True)
class GraphRecord:
    group_index: int
    tile_index: int
    detector_indices: NDArray[np.int64]
    node_features: NDArray[np.float32]
    edge_index: NDArray[np.int64]
    edge_weight: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class ExampleMetadata:
    shot_id: int
    query: QueryMetadata
    groups: tuple[TemporalGroup, ...]
    tile_paths: tuple[tuple[tuple[int, int], ...], ...]


@dataclass(frozen=True, slots=True)
class FeatureExample:
    metadata: ExampleMetadata
    graphs: tuple[GraphRecord, ...]


@dataclass(frozen=True, slots=True)
class FeatureBatch:
    examples: tuple[ExampleMetadata, ...]
    example_graph_offsets: NDArray[np.int64]
    graph_node_offsets: NDArray[np.int64]
    graph_edge_offsets: NDArray[np.int64]
    graph_group_indices: NDArray[np.int64]
    graph_tile_indices: NDArray[np.int64]
    node_detector_indices: NDArray[np.int64]
    node_features: NDArray[np.float32]
    edge_index: NDArray[np.int64]
    edge_weight: NDArray[np.float32]

    def to_arrays(self) -> dict[str, NDArray]:
        """Export the model-array schema explicitly, separate from supervision."""
        return {
            "shot_ids": np.array([example.shot_id for example in self.examples], dtype=np.int64),
            "core_indices": np.array([
                -1 if example.query.target.core_index is None else example.query.target.core_index
                for example in self.examples
            ], dtype=np.int64),
            "example_graph_offsets": self.example_graph_offsets,
            "graph_node_offsets": self.graph_node_offsets,
            "graph_edge_offsets": self.graph_edge_offsets,
            "graph_group_indices": self.graph_group_indices,
            "graph_tile_indices": self.graph_tile_indices,
            "node_detector_indices": self.node_detector_indices,
            "node_features": self.node_features,
            "edge_index": self.edge_index,
            "edge_weight": self.edge_weight,
        }


@dataclass(frozen=True, slots=True)
class SupervisedBatch:
    inputs: FeatureBatch
    labels: NDArray[np.bool_]


@cache
def _standalone_positions(distance: int) -> frozenset[tuple[int, int]]:
    # Static geometry of the child patch, with all bulk stabilizer positions.
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=distance, rounds=3)
    xy = np.array([point[:2] for point in circuit.get_detector_coordinates().values()])
    return frozenset(map(tuple, (xy - xy.min(axis=0)).astype(np.int64).tolist()))


def _grid_ranges(values: NDArray[np.int64], count: int) -> tuple[tuple[float, float], ...]:
    low, high = int(values.min()), int(values.max())
    if count == 2:
        center = (low + high) / 2
        return ((-np.inf, center + 1), (center - 1, np.inf))
    step = (high - low - 6) // 2
    first, second = low + step + 1, low + 2 * step + 1
    return ((-np.inf, first + 1), (first - 1, second + 1), (second - 1, np.inf))


def _grid_tiles(
    xy: NDArray[np.int64], indices: NDArray[np.int64], distance: int, grids: tuple[int, ...],
    path: tuple[tuple[int, int], ...] = (),
) -> tuple[SpatialTile, ...]:
    if not grids:
        return (SpatialTile(path, indices, xy),)
    grid, *inner = grids
    child_distance = (distance + 1) // 2 if grid == 2 else 3
    valid = _standalone_positions(child_distance)
    tiles = []
    for row, (y_min, y_max) in enumerate(_grid_ranges(xy[:, 1], grid)):
        for column, (x_min, x_max) in enumerate(_grid_ranges(xy[:, 0], grid)):
            inside = (
                (xy[:, 0] >= x_min) & (xy[:, 0] <= x_max)
                & (xy[:, 1] >= y_min) & (xy[:, 1] <= y_max)
            )
            cropped = xy[inside]
            cropped -= cropped.min(axis=0)
            keep = np.array([tuple(position) in valid for position in cropped], dtype=bool)
            tiles.extend(_grid_tiles(
                cropped[keep], indices[inside][keep], child_distance, tuple(inner), (*path, (row, column)),
            ))
    return tuple(tiles)


def memory_grids(distance: int) -> tuple[int, ...]:
    """Square memory grid widths, from the outermost split to the innermost."""
    grids = {3: (), 5: (2,), 7: (3,), 9: (2, 2)}
    if distance not in grids:
        raise ValueError("Lange square memory tiling is defined for distances 3, 5, 7 and 9")
    return grids[distance]


def build_feature_geometry(layout: ObservationLayout) -> FeatureGeometry:
    """Prepare fixed spatial tiles from the public detector lattice."""
    grids = memory_grids(layout.distance)
    coordinates = tuple(site.coordinates for site in layout.detectors)
    xy = np.array([point[:2] for point in coordinates])
    if xy.shape != (len(coordinates), 2) or not len(xy) or not np.isfinite(xy).all():
        raise ValueError("expected a nonempty detector lattice with finite spatial coordinates")
    xy = (xy - xy.min(axis=0)).astype(np.int64)
    indices = np.arange(len(xy), dtype=np.int64)
    return FeatureGeometry(layout.distance, coordinates, _grid_tiles(xy, indices, layout.distance, grids))


def _temporal_groups(metadata: QueryMetadata) -> tuple[TemporalGroup, ...]:
    observed = {slot.layer_index for slot in metadata.slots if slot.state == SlotState.OBSERVED}
    # Complete adjacent pairs match Lange's full-context layout. An isolated
    # observed layer still gets a slot, with the neighboring context absent.
    return tuple(
        TemporalGroup(slot.layer_index, slot.layer_index + 1)
        for slot in metadata.slots
        if slot.layer_index in observed
        and (slot.layer_index + 1 in observed or slot.layer_index - 1 not in observed)
    )


def _event_graph(group: int, tile: int, ids: NDArray[np.int64], nodes: NDArray[np.float32]) -> GraphRecord:
    source, destination = np.nonzero(~np.eye(len(nodes), dtype=bool))
    distance = np.abs(nodes[source] - nodes[destination]).max(axis=1)
    weights = np.divide(1.0, distance**2, out=np.zeros_like(distance), where=distance > 0)
    return GraphRecord(group, tile, ids, nodes, np.stack([source, destination]), weights)


def build_features(cropped: CroppedObservations, geometry: FeatureGeometry) -> tuple[FeatureExample, ...]:
    """Build group-major, tile-minor graphs from an already cropped bit array."""
    view = cropped.view
    if (
        view.distance != geometry.distance or view.detector_count != len(geometry.detector_coordinates)
        or any(geometry.detector_coordinates[site.detector_index] != site.coordinates for site in view.detectors)
    ):
        raise ValueError("feature geometry does not match the physical window")
    if cropped.bits.dtype != np.bool_ or cropped.bits.shape != (len(cropped.shot_ids), len(view.detectors)):
        raise ValueError("cropped bits must align with selected detector columns and shot IDs")

    groups = _temporal_groups(view.metadata)
    tile_paths = tuple(tile.path for tile in geometry.tiles)
    column_by_detector = np.full(view.detector_count, -1, dtype=np.int64)
    column_by_detector[list(view.detector_indices)] = np.arange(len(view.detectors))
    layer_by_column = np.array([site.layer_index for site in view.detectors], dtype=np.int64)
    projections = []
    for tile in geometry.tiles:
        columns = column_by_detector[tile.detector_indices]
        present = columns >= 0
        projections.append((columns[present], tile.detector_indices[present], tile.xy[present]))

    graphs_by_shot: list[list[GraphRecord]] = [[] for _ in cropped.shot_ids]
    for group_index, group in enumerate(groups):
        for tile_index, (columns, detector_ids, xy) in enumerate(projections):
            local_time = layer_by_column[columns] - group.first_layer
            in_group = (local_time >= 0) & (local_time <= 1)
            group_ids = detector_ids[in_group]
            node_coordinates = np.column_stack((xy[in_group], local_time[in_group])).astype(np.float32)
            for row, events in enumerate(cropped.bits[:, columns[in_group]]):
                graphs_by_shot[row].append(
                    _event_graph(group_index, tile_index, group_ids[events], node_coordinates[events])
                )
    return tuple(
        FeatureExample(ExampleMetadata(int(shot_id), view.metadata, groups, tile_paths), tuple(graphs))
        for shot_id, graphs in zip(cropped.shot_ids, graphs_by_shot, strict=True)
    )


def _offsets(counts) -> NDArray[np.int64]:
    return np.fromiter(accumulate(counts, initial=0), dtype=np.int64)


def batch_features(examples: tuple[FeatureExample, ...]) -> FeatureBatch:
    all_graphs = []
    for example in examples:
        expected = [
            (group, tile) for group in range(len(example.metadata.groups))
            for tile in range(len(example.metadata.tile_paths))
        ]
        if [(graph.group_index, graph.tile_index) for graph in example.graphs] != expected:
            raise ValueError("graph slots must retain every temporal group and tile, including empty graphs")
        all_graphs.extend(example.graphs)
    node_offsets = _offsets(len(graph.node_features) for graph in all_graphs)
    edge_offsets = _offsets(len(graph.edge_weight) for graph in all_graphs)
    nodes = np.empty((node_offsets[-1], 3), dtype=np.float32)
    detector_ids = np.empty(node_offsets[-1], dtype=np.int64)
    edges = np.empty((2, edge_offsets[-1]), dtype=np.int64)
    weights = np.empty(edge_offsets[-1], dtype=np.float32)
    for index, graph in enumerate(all_graphs):
        node_slice = slice(node_offsets[index], node_offsets[index + 1])
        edge_slice = slice(edge_offsets[index], edge_offsets[index + 1])
        nodes[node_slice] = graph.node_features
        detector_ids[node_slice] = graph.detector_indices
        edges[:, edge_slice] = graph.edge_index + node_offsets[index]
        weights[edge_slice] = graph.edge_weight
    return FeatureBatch(
        tuple(example.metadata for example in examples),
        _offsets(len(example.graphs) for example in examples), node_offsets, edge_offsets,
        np.array([graph.group_index for graph in all_graphs], dtype=np.int64),
        np.array([graph.tile_index for graph in all_graphs], dtype=np.int64),
        detector_ids, nodes, edges, weights,
    )


def unbatch_features(batch: FeatureBatch) -> tuple[FeatureExample, ...]:
    examples = []
    for example_index, metadata in enumerate(batch.examples):
        graphs = []
        for index in range(batch.example_graph_offsets[example_index], batch.example_graph_offsets[example_index + 1]):
            start = batch.graph_node_offsets[index]
            nodes = slice(start, batch.graph_node_offsets[index + 1])
            edges = slice(batch.graph_edge_offsets[index], batch.graph_edge_offsets[index + 1])
            graphs.append(GraphRecord(
                int(batch.graph_group_indices[index]), int(batch.graph_tile_indices[index]),
                batch.node_detector_indices[nodes], batch.node_features[nodes],
                batch.edge_index[:, edges] - start, batch.edge_weight[edges],
            ))
        examples.append(FeatureExample(metadata, tuple(graphs)))
    return tuple(examples)


def attach_supervision(inputs: FeatureBatch, shots: ShotBatch) -> SupervisedBatch:
    """Join target labels by shot ID and query, outside the model-input schema."""
    if shots.shot_ids.dtype != np.int64 or shots.shot_ids.ndim != 1:
        raise ValueError("supervision shot IDs must be an int64 vector")
    rows = {int(shot_id): row for row, shot_id in enumerate(shots.shot_ids)}
    if len(rows) != len(shots.shot_ids):
        raise ValueError("supervision requires unique physical shot IDs")
    if (
        shots.global_labels.shape != (len(rows),) or shots.core_labels.ndim != 2
        or shots.core_labels.shape[0] != len(rows)
        or shots.global_labels.dtype != np.bool_ or shots.core_labels.dtype != np.bool_
    ):
        raise ValueError("labels must align with the physical shot IDs")
    labels = np.empty(len(inputs.examples), dtype=np.bool_)
    for index, example in enumerate(inputs.examples):
        if example.shot_id not in rows:
            raise ValueError(f"missing label row for shot {example.shot_id}")
        row = rows[example.shot_id]
        core = example.query.target.core_index
        if core is not None and not 0 <= core < shots.core_labels.shape[1]:
            raise ValueError("target core is outside the label columns")
        labels[index] = shots.global_labels[row] if core is None else shots.core_labels[row, core]
    return SupervisedBatch(inputs, labels)
