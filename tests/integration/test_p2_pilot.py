from dataclasses import replace
from itertools import product

import numpy as np

from nn_window_ablation.config import resolve_config
from nn_window_ablation.features import (
    attach_supervision, batch_features, build_feature_geometry, build_features, unbatch_features,
)
from nn_window_ablation.simulation import CircuitArtifact, CircuitSpec, build_circuit_bundle, sample_shots
from nn_window_ablation.windows import (
    Buffers, build_observation_layout, crop_observations, pack_observations, select_window, unpack_observations,
)
from tests.p2_checks import assert_examples_equal, assert_indices
from tests.reference.p1_pauli_frame import Fault, inject_faults
from tests.reference.p1_schedule import PILOT
from tests.reference.p2_observations import PILOT_SCHEDULE


def test_p2_full_context_and_batch_roundtrip(compose_settings, tmp_path):
    cfg = resolve_config(compose_settings("runtime=test_cpu"), invocation_dir=tmp_path)
    spec = CircuitSpec(cfg.geometry.distance, cfg.geometry.rounds, cfg.geometry.core_length, cfg.noise.probability)
    assert spec == CircuitSpec(5, 60, 5, 0.003)
    bundle = build_circuit_bundle(spec)
    shots = sample_shots(bundle, 2, seed=cfg.seed, first_shot_id=60, include_snapshots=True)
    layout = build_observation_layout(bundle)
    geometry = build_feature_geometry(layout)
    bits, ids = unpack_observations(pack_observations(shots.detectors, shots.shot_ids))
    np.testing.assert_array_equal(bits, shots.detectors)
    np.testing.assert_array_equal(ids, shots.shot_ids)

    def examples_for(view):
        return build_features(crop_observations(view, bits[::-1], ids[::-1]), geometry)

    def roundtrip(examples, expected):
        inputs = batch_features(examples)
        supervised = attach_supervision(inputs, shots)
        expected = np.asarray(expected, dtype=np.bool_)
        np.testing.assert_array_equal(supervised.labels, expected)
        for (example, label), reference, expected_label in zip(
            zip(unbatch_features(supervised.inputs), supervised.labels, strict=True), examples, expected, strict=True,
        ):
            assert_examples_equal(example, reference)
            assert label == expected_label

    global_view = select_window(layout, layout.global_query)
    global_examples = examples_for(global_view)
    assert_indices(global_view.detector_indices, tuple(range(1440)))
    assert len(global_examples[0].metadata.groups) == 60
    roundtrip(global_examples, shots.global_labels[::-1])

    for index in range(12):
        query = layout.core_query(index)
        view = select_window(layout, query)
        assert view.metadata.target == query
        full = examples_for(view)
        for baseline, example in zip(global_examples, full, strict=True):
            expected = replace(baseline, metadata=replace(baseline.metadata, query=view.metadata))
            assert_examples_equal(example, expected)
        expected_labels = shots.core_labels[::-1, index]
        roundtrip(full, expected_labels)
        # Pair variable-length contexts for the exact same targets and IDs.
        for left, right in sorted(product((0, 3, 5), repeat=2)):
            window = select_window(layout, query, Buffers(left, right))
            assert window.metadata.target == view.metadata.target
            assert_indices(window.detector_indices, PILOT_SCHEDULE.select(index, (left, right)))
            restricted = examples_for(window)
            roundtrip((*restricted, *full), np.tile(expected_labels, 2))

    noiseless = build_circuit_bundle(replace(spec, probability=0))
    zero = sample_shots(noiseless, 2, seed=cfg.seed, first_shot_id=70)
    zero_layout = build_observation_layout(noiseless)
    zero_geometry = build_feature_geometry(zero_layout)
    zero_view = select_window(zero_layout, zero_layout.core_query(11), Buffers(5, 0))
    zero_features = build_features(crop_observations(zero_view, zero.detectors, zero.shot_ids), zero_geometry)
    zero_batch = attach_supervision(batch_features(zero_features), zero)
    np.testing.assert_array_equal(zero_batch.labels, [False, False])
    np.testing.assert_array_equal(zero_batch.inputs.to_arrays()["shot_ids"], [70, 71])
    assert not zero.detectors.any()
    assert len(zero_features) == 2 and len(zero_features[0].graphs) == 40
    assert zero_batch.inputs.node_features.shape == (0, 3)
    assert zero_batch.inputs.graph_node_offsets.tolist() == [0] * 81
    for recovered, original in zip(unbatch_features(zero_batch.inputs), zero_features, strict=True):
        assert_examples_equal(recovered, original)


def test_p2_physical_late_readout_fault_preserves_committed_features():
    bundle = build_circuit_bundle(CircuitSpec(5, 60, 5, 0.003))
    teacher = bundle.teacher.to_stim()
    late = Fault(PILOT.readout_instruction, 0, "M_RESULT", (1,), "FLIP")
    reference = replace(bundle, teacher=CircuitArtifact(str(inject_faults(teacher, ()))))
    changed = replace(bundle, teacher=CircuitArtifact(str(inject_faults(teacher, (late,)))))
    base = sample_shots(reference, 1, seed=18, first_shot_id=41)
    perturbed = sample_shots(changed, 1, seed=18, first_shot_id=41)
    np.testing.assert_array_equal(np.flatnonzero(base.detectors[0] ^ perturbed.detectors[0]), [1430])
    assert base.global_labels[0] != perturbed.global_labels[0]
    layout = build_observation_layout(bundle)
    geometry = build_feature_geometry(layout)
    penultimate = select_window(layout, layout.core_query(10), Buffers(5, 5))
    final = select_window(layout, layout.core_query(11), Buffers(5, 0))
    assert 1430 not in penultimate.detector_indices and 1430 in final.detector_indices
    early_base, = build_features(crop_observations(penultimate, base.detectors, base.shot_ids), geometry)
    early_fault, = build_features(crop_observations(penultimate, perturbed.detectors, perturbed.shot_ids), geometry)
    assert_examples_equal(early_base, early_fault)
