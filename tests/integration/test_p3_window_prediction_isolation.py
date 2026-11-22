from dataclasses import replace

import numpy as np
import pytest
import torch

from nn_window_ablation.models.gnn_rnn import QueryDecoder
from nn_window_ablation.windows import Buffers, select_window
from tests.p2_checks import assert_examples_equal, assert_graph
from tests.p3_checks import examples_for, model_inputs, pilot
from tests.reference.p2_observations import D5_TILE_ORIGINS, PILOT_SCHEDULE, expected_graph


@pytest.mark.parametrize("core_index", [4, 10])
def test_p3_window_prediction_isolation(pilot, core_index):
    _, layout, geometry, shots = pilot
    torch.manual_seed(422)
    model = QueryDecoder(5).eval()
    view = select_window(layout, layout.core_query(core_index), Buffers(5, 5))
    cutoff = 2 * min(60, 5 * (core_index + 1) + 5)
    future = tuple(
        index for index, dependencies in enumerate(PILOT_SCHEDULE.detector_dependencies)
        if max(PILOT_SCHEDULE.measurement_times[record] for record in dependencies) > cutoff
    )
    assert set(range(1428, 1440)) <= set(future)
    assert not set(view.detector_indices) & set(future)
    if core_index == 10:
        assert future == tuple(range(1428, 1440))

    modified_bits = shots.detectors.copy()
    modified_bits[:, future] ^= True
    with_future = replace(shots, detectors=modified_bits)
    with_other_labels = replace(
        shots, global_labels=~shots.global_labels, core_labels=~shots.core_labels, snapshots=~shots.snapshots,
    )
    baseline_examples = examples_for(view, geometry, shots)
    future_examples = examples_for(view, geometry, with_future)
    other_labels = examples_for(view, geometry, with_other_labels)
    for expected, altered, relabeled in zip(baseline_examples, future_examples, other_labels, strict=True):
        assert_examples_equal(expected, altered)
        assert_examples_equal(expected, relabeled)

    def invariant(before, after):
        torch.testing.assert_close(after, before, atol=0, rtol=0)

    with torch.inference_mode():
        baseline = model(model_inputs(baseline_examples))
        altered = model(model_inputs(future_examples))
        relabeled = model(model_inputs(other_labels))
    invariant(baseline, altered)
    invariant(baseline, relabeled)

    # Deliberately add a raw excluded observation to the readout in this test only.
    def leaky_readout(logits, physical):
        return logits + torch.from_numpy(physical.detectors[:, future[0]])

    with pytest.raises(AssertionError):
        invariant(leaky_readout(baseline, shots), leaky_readout(altered, with_future))


def test_p3_retained_detector_reaches_the_expected_graph(pilot):
    bundle, layout, geometry, shots = pilot
    view = select_window(layout, layout.core_query(10), Buffers(5, 5))
    expected_ids = PILOT_SCHEDULE.select(10, (5, 5))
    detector = 1068  # The first bulk detector in physical layer 45, at tile-local (2, 0).
    assert detector in expected_ids
    changed = shots.detectors.copy()
    changed[0, detector] ^= True
    baseline, = examples_for(view, geometry, shots, slice(0, 1))
    altered, = examples_for(view, geometry, replace(shots, detectors=changed), slice(0, 1))
    assert (baseline.metadata.groups[0].first_layer, baseline.metadata.groups[0].second_layer) == (45, 46)
    coords = tuple(site.coordinates for site in bundle.detectors)
    for example, row in ((baseline, shots.detectors[0]), (altered, changed[0])):
        expected = expected_graph(
            PILOT_SCHEDULE, expected_ids, set(np.flatnonzero(row)), coords, (45, 46), D5_TILE_ORIGINS[0], tiled=True,
        )
        assert_graph(example.graphs[0], expected)
    assert (detector in baseline.graphs[0].detector_indices) != (detector in altered.graphs[0].detector_indices)
