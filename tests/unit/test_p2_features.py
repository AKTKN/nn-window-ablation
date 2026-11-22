from dataclasses import replace
from io import BytesIO

import numpy as np
import pytest

from nn_window_ablation.features import (
    attach_supervision, batch_features, build_feature_geometry, build_features, unbatch_features,
)
from nn_window_ablation.simulation import CircuitSpec, ShotBatch, build_circuit_bundle, sample_shots
from nn_window_ablation.windows import (
    Buffers, SlotState, build_observation_layout, crop_observations, pack_observations,
    select_window, unpack_observations,
)
from tests.p2_checks import (
    assert_examples_equal, assert_graph, assert_metadata_schema, assert_model_arrays, toy_bundle,
)
from tests.reference.p2_observations import (
    D5_TILE_ORIGINS, PILOT_SCHEDULE, TOY_COORDS, TOY_SCHEDULE, expected_graph,
)


@pytest.fixture(scope="module")
def toy():
    layout = build_observation_layout(toy_bundle())
    return layout, build_feature_geometry(layout)


@pytest.fixture(scope="module")
def pilot():
    bundle = build_circuit_bundle(CircuitSpec(5, 60, 5, 0.003))
    layout = build_observation_layout(bundle)
    return bundle, layout, build_feature_geometry(layout)


@pytest.mark.parametrize(
    "events, nodes, pair, weight",
    [
        ((), (), (3, 4), ()),
        ((3,), ((0, 0, 0),), (3, 4), ()),
        ((3, 4), ((0, 0, 0), (1, 0, 1)), (3, 4), (1, 1)),
        ((5, 6), ((3, 0, 1), (1, 0, 1)), (4, 5), (0.25, 0.25)),
    ],
)
def test_p2_hand_graphs_preserve_local_coordinates_and_edges(toy, events, nodes, pair, weight):
    layout, geometry = toy
    view = select_window(layout, layout.core_query(1), Buffers(0, 0))
    bits = np.zeros((1, 11), dtype=np.bool_)
    bits[0, list(events)] = True
    example, = build_features(crop_observations(view, bits, np.array([7], dtype=np.int64)), geometry)
    groups = [(group.first_layer, group.second_layer) for group in example.metadata.groups]
    assert groups == [(3, 4), (4, 5)]
    assert example.metadata.tile_paths == ((),)
    graph = example.graphs[groups.index(pair)]
    edges = ((0, 1), (1, 0)) if len(nodes) == 2 else ()
    assert_graph(graph, (events, nodes, edges, weight))


def test_p2_raw_crop_precedes_two_layer_features(pilot):
    bundle, layout, geometry = pilot
    target = layout.core_query(5)
    view = select_window(layout, target, Buffers(3, 3))
    allowed = PILOT_SCHEDULE.select(5, (3, 3))
    pairs = PILOT_SCHEDULE.pairs(allowed)
    assert pairs[0] == (22, 23) and pairs[-1] == (31, 32)
    for_layers = (21, 22, 23, 31, 32, 33)
    events = {
        12 + (layer - 1) * 24 + offset
        for layer in for_layers for offset in (0, 1, 2)
    }
    bits = np.zeros((1, 1440), dtype=np.bool_)
    bits[0, sorted(events)] = True
    shot_ids = np.array([19], dtype=np.int64)
    example, = build_features(crop_observations(view, bits, shot_ids), geometry)
    assert [(group.first_layer, group.second_layer) for group in example.metadata.groups] == list(pairs)
    assert example.metadata.tile_paths == (((0, 0),), ((0, 1),), ((1, 0),), ((1, 1),))
    coords = tuple(detector.coordinates for detector in bundle.detectors)
    assert len(example.graphs) == len(pairs) * 4
    for group_index, pair in enumerate(pairs):
        for tile_index, origin in enumerate(D5_TILE_ORIGINS):
            graph = example.graphs[group_index * 4 + tile_index]
            assert (graph.group_index, graph.tile_index) == (group_index, tile_index)
            assert_graph(graph, expected_graph(PILOT_SCHEDULE, allowed, events, coords, pair, origin, tiled=True))

    # The forbidden path keeps precomputed edge groups whose second layer or
    # first layer touches the retained span. Those groups already have outside events.
    full = select_window(layout, target)
    prebuilt, = build_features(crop_observations(full, bits, shot_ids), geometry)
    for leaking_pair, tile in [((21, 22), 0), ((32, 33), 0)]:
        leaking_group = next(index for index, group in enumerate(prebuilt.metadata.groups)
                             if (group.first_layer, group.second_layer) == leaking_pair)
        leaking = prebuilt.graphs[leaking_group * 4 + tile]
        assert set(leaking.detector_indices) - set(allowed)
        with pytest.raises(AssertionError):
            assert_graph(leaking, expected_graph(
                PILOT_SCHEDULE, allowed, events, coords, leaking_pair, D5_TILE_ORIGINS[tile], tiled=True,
            ))


def test_p2_unavailable_perturbations_do_not_change_features(pilot):
    bundle, layout, geometry = pilot
    shots = sample_shots(bundle, 3, seed=617, first_shot_id=10)
    view = select_window(layout, layout.core_query(10), Buffers(3, 5))
    allowed = PILOT_SCHEDULE.select(10, (3, 5))
    excluded = sorted(set(range(1440)) - set(allowed))
    assert set(range(1428, 1440)) <= set(excluded)
    perturbed = shots.detectors.copy()
    perturbed[:, excluded] ^= True

    def pipeline(bits):
        restored, ids = unpack_observations(pack_observations(bits, shots.shot_ids))
        return build_features(crop_observations(view, restored, ids), geometry)

    original, altered = pipeline(shots.detectors), pipeline(perturbed)
    for before, after in zip(original, altered, strict=True):
        assert_examples_equal(before, after)
    before_arrays = batch_features(original).to_arrays()
    after_arrays = batch_features(altered).to_arrays()
    for name in before_arrays:
        np.testing.assert_array_equal(before_arrays[name], after_arrays[name], err_msg=name)


def test_p2_empty_missing_padding_are_distinct(toy):
    layout, geometry = toy
    initial = select_window(layout, layout.core_query(0), Buffers(3, 0))
    final = select_window(layout, layout.core_query(2), Buffers(0, 3))
    assert [(slot.layer_index, slot.state) for slot in initial.metadata.slots] == [
        (-3, SlotState.PADDING), (-2, SlotState.PADDING), (-1, SlotState.PADDING),
        (0, SlotState.OBSERVED), (1, SlotState.OBSERVED), (2, SlotState.OBSERVED), (3, SlotState.UNAVAILABLE),
    ]
    assert [(slot.layer_index, slot.state) for slot in final.metadata.slots] == [
        (5, SlotState.UNAVAILABLE), (6, SlotState.OBSERVED), (7, SlotState.OBSERVED),
        (8, SlotState.OBSERVED), (9, SlotState.OBSERVED),
        (10, SlotState.PADDING), (11, SlotState.PADDING), (12, SlotState.PADDING),
    ]
    zero = np.zeros((1, 11), dtype=np.bool_)
    event = zero.copy()
    event[0, 10] = True
    early, = build_features(crop_observations(initial, zero, np.array([33], dtype=np.int64)), geometry)
    late, = build_features(crop_observations(final, event, np.array([12], dtype=np.int64)), geometry)
    assert len(early.graphs) == 2
    assert len(late.graphs) == 3
    assert all(graph.node_features.shape == (0, 3) for graph in (*early.graphs, *late.graphs[:-1]))
    assert_graph(late.graphs[-1], ([10], [(4, 0, 1)], [], []))
    batch = batch_features((early, late))
    np.testing.assert_array_equal(batch.example_graph_offsets, [0, 2, 5])
    np.testing.assert_array_equal(batch.graph_node_offsets, [0, 0, 0, 0, 0, 1])
    for recovered, original in zip(unbatch_features(batch), (early, late), strict=True):
        assert_examples_equal(recovered, original)
    with pytest.raises(ValueError, match="including empty graphs"):
        batch_features((replace(early, graphs=early.graphs[1:]), late))


def test_p2_labels_never_enter_features(toy):
    layout, geometry = toy
    bits = np.zeros((2, 11), dtype=np.bool_)
    bits[1, (3, 4)] = True
    shots = ShotBatch(
        bits, np.array([True, False]), np.array([[False, True, False], [True, False, True]]),
        np.array([8, 2], dtype=np.int64), np.array([[True, False], [False, True]]),
    )
    changed = replace(
        shots, global_labels=~shots.global_labels, core_labels=~shots.core_labels, snapshots=~shots.snapshots,
    )
    view = select_window(layout, layout.core_query(1), Buffers(0, 0))

    def inputs_for(physical):
        observations = crop_observations(view, physical.detectors, physical.shot_ids)
        return batch_features(build_features(observations, geometry))

    original, altered = inputs_for(shots), inputs_for(changed)
    fields = original.to_arrays()
    assert_model_arrays(fields)
    assert original.examples == altered.examples
    for example in original.examples:
        assert_metadata_schema(example)
    for name, array in altered.to_arrays().items():
        np.testing.assert_array_equal(array, fields[name], err_msg=name)
    np.testing.assert_array_equal(attach_supervision(original, shots).labels, [True, False])
    np.testing.assert_array_equal(attach_supervision(altered, changed).labels, [False, True])

    artifact = BytesIO()
    np.savez(artifact, **fields)
    artifact.seek(0)
    with np.load(artifact, allow_pickle=False) as loaded:
        assert_model_arrays(loaded)
        for name in loaded:
            np.testing.assert_array_equal(loaded[name], fields[name])
    with pytest.raises(AssertionError, match="model array whitelist"):
        assert_model_arrays({**fields, "core_labels": shots.core_labels})
    with pytest.raises(ValueError, match="int64"):
        attach_supervision(original, replace(shots, shot_ids=shots.shot_ids.astype(np.float64)))
