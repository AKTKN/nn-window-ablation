"""Fixtures and assertions that connect the independent P2 expectations to tests."""

from dataclasses import fields

import numpy as np

from nn_window_ablation.simulation import (
    Arrival, ArrivalPhase, CircuitArtifact, CircuitBundle, CircuitSpec, CoreInterval, Detector, Measurement,
)
from tests.reference.p2_observations import TOY_COORDS, TOY_DECLARATIONS, TOY_SCHEDULE


MODEL_ARRAY_FIELDS = {
    "shot_ids", "core_indices", "example_graph_offsets", "graph_node_offsets", "graph_edge_offsets",
    "graph_group_indices", "graph_tile_indices", "node_detector_indices", "node_features", "edge_index", "edge_weight",
}


def toy_bundle():
    mr = tuple(range(10, 100, 10))
    measurements = tuple(
        Measurement(index, index, instruction, Arrival(index + 1, ArrivalPhase.ANCILLA))
        for index, instruction in enumerate(mr)
    ) + (Measurement(9, 9, 100, Arrival(9, ArrivalPhase.DESTRUCTIVE_READOUT)),)
    # Intentionally stale cached arrival for the terminal detector: the
    # selection implementation must honor its actual measurement dependencies.
    detectors = tuple(
        Detector(index, instruction, coordinates, dependencies, Arrival(layer + 1, ArrivalPhase.ANCILLA), instruction)
        for index, (instruction, coordinates, dependencies, layer) in enumerate(zip(
            TOY_DECLARATIONS, TOY_COORDS, TOY_SCHEDULE.detector_dependencies, TOY_SCHEDULE.detector_layers,
        ))
    )
    return CircuitBundle(
        spec=CircuitSpec(3, 9, 3, 0), stim_version="literal fixture",
        original=CircuitArtifact(""), teacher=CircuitArtifact(""), measurements=measurements,
        detectors=detectors, observables=(), cuts=(), preparation_stop=0,
        cores=(CoreInterval(0, 1, 3, 0, 31), CoreInterval(1, 4, 6, 31, 61), CoreInterval(2, 7, 9, 61, 102)),
        round_measurement_instructions=mr, destructive_readout_instruction=100,
    )


def assert_indices(actual, expected):
    assert tuple(actual) == tuple(expected), "permitted detector indices"


def assert_graph(actual, expected):
    detector_ids, nodes, edges, weights = expected
    np.testing.assert_array_equal(actual.detector_indices, np.array(detector_ids, dtype=np.int64))
    np.testing.assert_array_equal(actual.node_features, np.array(nodes, dtype=np.float32).reshape(-1, 3))
    np.testing.assert_array_equal(actual.edge_index, np.array(edges, dtype=np.int64).reshape(-1, 2).T)
    np.testing.assert_array_equal(actual.edge_weight, np.array(weights, dtype=np.float32))


def assert_examples_equal(left, right):
    assert left.metadata == right.metadata
    assert len(left.graphs) == len(right.graphs), "graph slots"
    for actual, expected in zip(left.graphs, right.graphs):
        assert (actual.group_index, actual.tile_index) == (expected.group_index, expected.tile_index)
        assert_graph(actual, (
            expected.detector_indices, expected.node_features, expected.edge_index.T, expected.edge_weight,
        ))


def assert_model_arrays(fields_by_name):
    assert set(fields_by_name) == MODEL_ARRAY_FIELDS, "model array whitelist"
    assert all(isinstance(value, np.ndarray) and value.dtype != object for value in fields_by_name.values())


def assert_metadata_schema(example):
    assert {field.name for field in fields(example)} == {"shot_id", "query", "groups", "tile_paths"}
    assert {field.name for field in fields(example.query)} == {
        "target", "buffers", "first_observed_round", "last_observed_round", "release",
        "preparation", "destructive_readout", "slots",
    }
    assert {field.name for field in fields(example.query.target)} == {"core_index", "first_round", "last_round"}
    for slot in example.query.slots:
        assert {field.name for field in fields(slot)} == {"layer_index", "arrival", "state", "detector_indices"}
