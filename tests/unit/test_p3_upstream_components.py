from dataclasses import replace

import numpy as np
import pytest
import torch

from nn_window_ablation.features import (
    ExampleMetadata, FeatureExample, GraphRecord, TemporalGroup, batch_features,
)
from nn_window_ablation.models.gnn_rnn import EventGNN, QueryDecoder, prepare_model_batch
from nn_window_ablation.simulation import Arrival, ArrivalPhase
from nn_window_ablation.windows import QueryMetadata, SlotState, TargetQuery, TemporalSlot
from third_party.lange_reference.graphs import Graphs
from third_party.lange_reference.model import Decoder, GNN


def _fixed_weights(module):
    generator = torch.Generator().manual_seed(84)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)


def test_p3_upstream_gnn_and_empty_slots_match_with_known_weights():
    reference = GNN()
    adapter = EventGNN()
    torch.testing.assert_close(adapter.empty, torch.zeros(256), atol=0, rtol=0)
    _fixed_weights(reference)
    adapter.load_state_dict(reference.state_dict())
    x = torch.tensor([[2., 1., 0.], [0., 3., 1.], [1., 1., 0.]])
    edges = torch.tensor([[0, 1], [1, 0]])
    weight = torch.tensor([0.25, 0.25])
    expected = reference(Graphs(x, edges, weight, torch.tensor([0, 0, 1]), torch.tensor([0, 2])), 4)
    actual = adapter(x, edges, weight, torch.tensor([0, 0, 2]), 4)

    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(actual[1], reference.empty, atol=0, rtol=0)
    torch.testing.assert_close(actual[3], reference.empty, atol=0, rtol=0)


def test_p3_weighted_graph_and_mean_pool_match_hand_calculation():
    model = EventGNN()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        first, second = model.convs
        first.lin_rel.weight[0, 0] = 2
        first.lin_root.weight[0, 0] = 3
        first.lin_rel.bias[0] = 1
        second.lin_rel.weight[0, 0] = 4
        second.lin_root.weight[0, 0] = 5
        second.lin_rel.bias[0] = 0.5
        model.empty.fill_(7)

    observed = model(
        torch.tensor([[2., 0., 0.], [6., 0., 0.]]), torch.tensor([[0, 1], [1, 0]]),
        torch.tensor([0.25, 0.25]), torch.tensor([1, 1]), 3,
    )
    # First channel after conv 1: [10, 20]. After conv 2: [70.5, 110.5].
    expected = torch.zeros(3, 256)
    expected[[0, 2]] = 7
    expected[1, 0] = 90.5
    torch.testing.assert_close(observed, expected, atol=0, rtol=0)


def test_p3_upstream_spatial_gru_and_readout_match_with_known_weights():
    reference = Decoder(grids=[2])
    adapter = QueryDecoder(5)
    _fixed_weights(reference)
    adapter.aggregators.load_state_dict(reference.aggregators.state_dict())
    adapter.rnn.load_state_dict(reference.rnn.state_dict())
    adapter.head.load_state_dict(reference.head.state_dict())

    tiles = torch.arange(3 * 4 * 256, dtype=torch.float32).reshape(3, 4, 256) / 3072
    reference_grid = torch.stack([
        torch.stack((torch.stack((row[0], row[1]), dim=-1), torch.stack((row[2], row[3]), dim=-1)), dim=1)
        for row in tiles
    ]).contiguous(memory_format=torch.channels_last)
    expected = reference.aggregators[0](reference_grid)[:, :, 0, 0]
    actual = adapter.aggregate_tiles(tiles)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    # Common GRU input contract after spatial aggregation, with additive metadata neutral.
    neutral = torch.zeros((3, adapter.query_projection.in_features))
    sequences = (expected[:2], expected[2:])
    reference_logits = torch.stack([reference.head(reference.rnn(sequence[None])[0][0, -1, :])[0]
                                    for sequence in sequences])
    adapter_logits = adapter.readout_groups(actual + adapter.query_projection(neutral), (2, 1))
    torch.testing.assert_close(adapter_logits, reference_logits, atol=1e-7, rtol=1e-6)


def test_p3_tile_order_is_row_major_and_swapping_tiles_is_detected():
    model = QueryDecoder(5)
    kernel, project = model.aggregators[0][0], model.aggregators[0][2]
    with torch.no_grad():
        for parameter in model.aggregators.parameters():
            parameter.zero_()
        kernel.weight[0, 0] = torch.tensor([[1., 2.], [3., 4.]])
        project.weight[0, 0, 0, 0] = 2
        project.bias[0] = 1
    tiles = torch.zeros(1, 4, 256)
    tiles[0, :, 0] = torch.tensor([2., 3., 5., 7.])
    expected = torch.zeros(1, 256)
    expected[0, 0] = 103
    torch.testing.assert_close(model.aggregate_tiles(tiles), expected, atol=0, rtol=0)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(model.aggregate_tiles(tiles[:, [1, 0, 2, 3]]), expected, atol=0, rtol=0)


def _single_group(slot_state):
    slots = (
        TemporalSlot(0, Arrival(1, ArrivalPhase.ANCILLA), SlotState.OBSERVED, (0,)),
        TemporalSlot(1, Arrival(2, ArrivalPhase.ANCILLA), slot_state, ()),
    )
    metadata = ExampleMetadata(
        99, QueryMetadata(TargetQuery(0, 1, 3), None, 1, 1, Arrival(1, ArrivalPhase.ANCILLA), True, False, slots),
        (TemporalGroup(0, 1),), ((),),
    )
    graph = GraphRecord(
        0, 0, np.empty(0, np.int64), np.empty((0, 3), np.float32),
        np.empty((2, 0), np.int64), np.empty(0, np.float32),
    )
    return FeatureExample(metadata, (graph,))


def test_p3_missing_slot_is_distinct_from_an_observed_empty_slot():
    missing = _single_group(SlotState.UNAVAILABLE)
    empty = _single_group(SlotState.OBSERVED)
    expected = torch.tensor([[0, 1, 0, 1/3, -2/3, 1, 0, 0, 1, 0, 2/3, -1/3, 0, 1, 0, 0, 0]])
    missing_inputs = prepare_model_batch(batch_features((missing,)), distance=3)
    empty_inputs = prepare_model_batch(batch_features((empty,)), distance=3)
    torch.testing.assert_close(missing_inputs.group_metadata, expected, atol=0, rtol=0)
    assert missing_inputs.node_features.numel() == empty_inputs.node_features.numel() == 0
    with pytest.raises(AssertionError):
        torch.testing.assert_close(empty_inputs.group_metadata, expected, atol=0, rtol=0)


def test_p3_batch_rejects_edges_crossing_query_graphs():
    example = _single_group(SlotState.UNAVAILABLE)
    graph = replace(example.graphs[0], detector_indices=np.array([0]), node_features=np.zeros((1, 3), np.float32))
    example = replace(example, graphs=(graph,))
    valid = batch_features((example, example))
    with pytest.raises(ValueError, match="declared order"):
        prepare_model_batch(replace(valid, graph_tile_indices=np.array([0, 1])), distance=3)
    crossed = replace(valid, edge_index=np.array([[0], [1]]), edge_weight=np.array([1.], np.float32))
    with pytest.raises(ValueError, match="stay within"):
        prepare_model_batch(crossed, distance=3)
