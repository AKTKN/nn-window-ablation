"""Query-conditioned adaptation of Lange et al.'s modular GNN–RNN.

The architecture and initializers follow gnnrnn/model.py at
458210ac74ed0047760b8789c169e6b9eff732b8. See the packaged LANGE_LICENSE
for the upstream copyright and MIT license.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Any

import numpy as np
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_sequence
from torch_geometric.nn import GraphConv, global_mean_pool
import torch

from ..features import FeatureBatch, memory_grids
from ..simulation import ArrivalPhase
from ..windows import SlotState, TargetQuery, TemporalSlot


UPSTREAM_COMMIT = "458210ac74ed0047760b8789c169e6b9eff732b8"
GRAPH_CHANNELS = (3, 64, 256)
HIDDEN_SIZE = 256
GRU_LAYERS = 4
QUERY_SCHEMA_VERSION = 1
SLOT_FIELDS = (
    "relative_to_start", "relative_to_end", "observed", "unavailable", "padding",
    "preparation_observed", "readout_observed",
)
QUERY_FIELDS = (
    "global_query", "preparation_observed", "readout_observed",
    *(f"{slot}_{field}" for slot in ("first", "second") for field in SLOT_FIELDS),
)


@dataclass(frozen=True, slots=True)
class ModelBatch:
    """Graph tensors and query metadata ordered by example, group, then tile."""

    distance: int
    node_features: Tensor
    edge_index: Tensor
    edge_weight: Tensor
    node_graph: Tensor
    group_metadata: Tensor
    sequence_lengths: tuple[int, ...]


def _slot_metadata(slot: TemporalSlot, target: TargetQuery, distance: int) -> tuple[float, ...]:
    if slot.arrival is not None:
        round_number = slot.arrival.round
    else:
        # Padding to the right follows the separate readout slot.
        round_number = slot.layer_index + 1 if slot.layer_index < 0 else slot.layer_index
    observed = slot.state == SlotState.OBSERVED
    return (
        (round_number - target.first_round + 1) / distance,
        (round_number - target.last_round) / distance,
        float(observed),
        float(slot.state == SlotState.UNAVAILABLE),
        float(slot.state == SlotState.PADDING),
        float(observed and slot.layer_index == 0),
        float(observed and slot.arrival is not None and slot.arrival.phase == ArrivalPhase.DESTRUCTIVE_READOUT),
    )


def prepare_model_batch(features: FeatureBatch, *, distance: int) -> ModelBatch:
    """Encode public P2 metadata and keep graph memberships local to each query."""
    grids = memory_grids(distance)
    tile_paths = tuple(product(*(tuple(product(range(grid), repeat=2)) for grid in grids)))
    lengths = tuple(len(example.groups) for example in features.examples)
    if not lengths or min(lengths) < 1:
        raise ValueError("each model query requires at least one temporal group")
    tile_count = len(tile_paths)
    total_groups = sum(lengths)
    graph_count = total_groups * tile_count
    expected_offsets = np.cumsum((0, *(length * tile_count for length in lengths)))
    if not np.array_equal(features.example_graph_offsets, expected_offsets):
        raise ValueError("example graph offsets do not match the query lengths and tiles")
    group_indices = np.concatenate([np.repeat(np.arange(length), tile_count) for length in lengths])
    if (
        not np.array_equal(features.graph_group_indices, group_indices)
        or not np.array_equal(features.graph_tile_indices, np.tile(np.arange(tile_count), total_groups))
    ):
        raise ValueError("graph groups and tiles must match their declared order")

    metadata = []
    for example in features.examples:
        if example.tile_paths != tile_paths:
            raise ValueError("feature tiles must match the model's row-major spatial layout")
        query = example.query
        slots = {slot.layer_index: slot for slot in query.slots}
        common = (float(query.target.core_index is None), float(query.preparation), float(query.destructive_readout))
        for group in example.groups:
            first = slots.get(group.first_layer)
            second = slots.get(group.second_layer)
            if (
                first is None or second is None or first.state != SlotState.OBSERVED
                or group.second_layer != group.first_layer + 1
            ):
                raise ValueError("a graph group must start at an observed slot and retain both slot records")
            metadata.append((
                *common, *_slot_metadata(first, query.target, distance), *_slot_metadata(second, query.target, distance),
            ))

    offsets = features.graph_node_offsets
    if offsets.shape != (graph_count + 1,) or offsets[0] != 0:
        raise ValueError("graph node offsets must include every tile, including empty graphs")
    counts = np.diff(offsets)
    if np.any(counts < 0) or features.node_features.shape != (offsets[-1], GRAPH_CHANNELS[0]):
        raise ValueError("node features do not match the graph offsets")
    node_graph = np.repeat(np.arange(graph_count, dtype=np.int64), counts)
    edges = features.edge_index
    if edges.shape != (2, len(features.edge_weight)):
        raise ValueError("edge indices must align with edge weights")
    if edges.size and (edges.min() < 0 or edges.max() >= len(node_graph)):
        raise ValueError("edge index is outside the batch nodes")
    if edges.size and np.any(node_graph[edges[0]] != node_graph[edges[1]]):
        raise ValueError("edges must stay within their graph")
    return ModelBatch(
        distance, torch.from_numpy(features.node_features),
        torch.from_numpy(edges), torch.from_numpy(features.edge_weight),
        torch.from_numpy(node_graph), torch.tensor(metadata, dtype=torch.float32), lengths,
    )


class EventGNN(nn.Module):
    """Shared weighted GraphConv encoder with a learned empty-graph vector."""

    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.ModuleList(GraphConv(a, b) for a, b in zip(GRAPH_CHANNELS[:-1], GRAPH_CHANNELS[1:]))
        self.empty = nn.Parameter(torch.zeros(HIDDEN_SIZE))

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, node_graph: Tensor, graph_count: int,
    ) -> Tensor:
        if x.shape[0] == 0:
            return self.empty.expand(graph_count, -1)
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index, edge_weight))
        pooled = global_mean_pool(x, node_graph, size=graph_count)
        present = torch.bincount(node_graph, minlength=graph_count) > 0
        return torch.where(present[:, None], pooled, self.empty)


class QueryDecoder(nn.Module):
    """One freshly initialized Lange backbone and scalar head for all queries."""

    def __init__(self, distance: int):
        super().__init__()
        self.distance = distance
        self.grids = tuple(reversed(memory_grids(distance)))
        self.tile_count = prod(grid * grid for grid in self.grids)
        self.gnn = EventGNN()
        self.aggregators = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, grid), nn.ReLU(),
                nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, 1), nn.ReLU(),
            )
            for grid in self.grids
        )
        self.query_projection = nn.Linear(len(QUERY_FIELDS), HIDDEN_SIZE, bias=False)
        self.rnn = nn.GRU(HIDDEN_SIZE, HIDDEN_SIZE, GRU_LAYERS, batch_first=True)
        self.head = nn.Linear(HIDDEN_SIZE, 1)

    def aggregate_tiles(self, tile_embeddings: Tensor) -> Tensor:
        """Reduce innermost row-major tile grids first for each time group."""
        groups = tile_embeddings.shape[0]
        h = tile_embeddings
        for grid, layer in zip(self.grids, self.aggregators, strict=True):
            h = h.reshape(groups, -1, grid, grid, HIDDEN_SIZE)
            h = h.permute(0, 1, 4, 2, 3).reshape(-1, HIDDEN_SIZE, grid, grid)
            h = layer(h).reshape(groups, -1, HIDDEN_SIZE)
        return h.squeeze(1)

    def readout_groups(self, embeddings: Tensor, lengths: tuple[int, ...]) -> Tensor:
        """Run an independent zero-state GRU on each valid-length sequence."""
        packed = pack_sequence(embeddings.split(lengths), enforce_sorted=False)
        _, state = self.rnn(packed)
        return self.head(state[-1]).squeeze(-1)

    def forward(self, batch: ModelBatch) -> Tensor:
        if batch.distance != self.distance:
            raise ValueError("model distance does not match the prepared graph batch")
        groups = sum(batch.sequence_lengths)
        pooled = self.gnn(
            batch.node_features, batch.edge_index, batch.edge_weight, batch.node_graph, groups * self.tile_count,
        )
        spatial = self.aggregate_tiles(pooled.reshape(groups, self.tile_count, HIDDEN_SIZE))
        embedded = spatial + self.query_projection(batch.group_metadata)
        return self.readout_groups(embedded, batch.sequence_lengths)

    def architecture(self) -> dict[str, Any]:
        """Describe the fixed architecture and exact query-vector ordering."""
        return {
            "model": "query_gnn_rnn",
            "upstream_commit": UPSTREAM_COMMIT,
            "distance": self.distance,
            "graph_channels": list(GRAPH_CHANNELS),
            "spatial_grids_inner_first": list(self.grids),
            "feature_group_rounds": 2,
            "hidden_size": HIDDEN_SIZE,
            "gru_layers": GRU_LAYERS,
            "gru_dropout": 0.0,
            "query_schema_version": QUERY_SCHEMA_VERSION,
            "query_fields": list(QUERY_FIELDS),
            "query_projection": "additive_linear_no_bias",
            "head": "shared_scalar",
        }

    def checkpoint_payload(self) -> dict[str, Any]:
        """Create the model payload for caller-managed serialization."""
        return {"architecture": self.architecture(), "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> "QueryDecoder":
        architecture = payload["architecture"]
        with torch.random.fork_rng(devices=[]):
            model = cls(architecture["distance"])
        if architecture != model.architecture():
            raise ValueError("incompatible decoder architecture or query metadata schema")
        model.load_state_dict(payload["state_dict"], strict=True)
        return model


def probabilities(logits: Tensor) -> Tensor:
    return logits.sigmoid()


def hard_decisions(logits: Tensor) -> Tensor:
    return logits >= 0
