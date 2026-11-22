from dataclasses import replace

import numpy as np

from nn_window_ablation.config import resolve_config
from nn_window_ablation.simulation import Arrival, ArrivalPhase, CircuitSpec, build_circuit_bundle, sample_shots
from tests.p1_checks import assert_fixture_metadata, assert_signatures_match, sample_fault_signature
from tests.reference.p1_pauli_frame import Fault, assert_rounds_preserve_logical_z, oracle_signature
from tests.reference.p1_schedule import PILOT


def test_p1_pilot_boundary_integration(compose_settings, tmp_path):
    config = resolve_config(compose_settings("runtime=test_cpu"), invocation_dir=tmp_path)
    geometry = config.geometry
    spec = CircuitSpec(geometry.distance, geometry.rounds, geometry.core_length, config.noise.probability)
    assert spec == CircuitSpec(5, 60, 5, 0.003)
    bundle = build_circuit_bundle(spec)
    physical = bundle.original.to_stim().flattened()
    assert_fixture_metadata(bundle, PILOT)
    assert_rounds_preserve_logical_z(physical, PILOT)
    assert tuple(cut.instruction_offset for cut in bundle.cuts) == (
        251, 486, 721, 956, 1191, 1426, 1661, 1896, 2131, 2366, 2601,
    )
    assert bundle.global_detector_error_model().num_observables == 1
    assert bundle.teacher.to_stim().num_observables == 12
    assert {detector.arrival for detector in bundle.detectors[1404:1428]} == {Arrival(60, ArrivalPhase.ANCILLA)}
    assert {detector.arrival for detector in bundle.detectors[1428:]} == {Arrival(60, ArrivalPhase.DESTRUCTIVE_READOUT)}
    assert bundle.detectors[1427].arrival < bundle.detectors[1428].arrival

    cases = [
        (Fault(50, 0, "X_ERROR", (1,), "X"), 0),
        (Fault(2862, 0, "M_RESULT", (1,), "FLIP"), 11),
        (Fault(2862, 5, "M_RESULT", (12,), "FLIP"), None),
    ]
    for index, offset in enumerate(PILOT.cut_offsets):
        cases.extend([
            (Fault(offset - 1, 0, "GAP", (1,), "X"), index),
            (Fault(offset, 0, "GAP", (1,), "X"), index + 1),
        ])
    for fault, owner in cases:
        observed = sample_fault_signature(bundle, fault)
        assert_signatures_match(observed, oracle_signature(physical, PILOT, (fault,)))
        assert observed.cores == tuple(int(index == owner) for index in range(12))

    batch = sample_shots(bundle, 256, seed=config.seed, first_shot_id=1024, include_snapshots=True)
    assert batch.detectors.shape == (256, 1440)
    assert batch.core_labels.shape == (256, 12)
    assert batch.snapshots.shape == (256, 11)
    np.testing.assert_array_equal(np.logical_xor.reduce(batch.core_labels, axis=1), batch.global_labels)
    np.testing.assert_array_equal(batch.shot_ids, np.arange(1024, 1280, dtype=np.int64))

    zero_noise = build_circuit_bundle(replace(spec, probability=0.0))
    zero_batch = sample_shots(zero_noise, 32, seed=config.seed, first_shot_id=2048, include_snapshots=True)
    assert zero_noise.spec.probability == 0
    assert zero_batch.detectors.shape == (32, 1440)
    assert zero_batch.core_labels.shape == (32, 12)
    assert not any(bits.any() for bits in (
        zero_batch.detectors, zero_batch.core_labels, zero_batch.global_labels, zero_batch.snapshots,
    ))
    np.testing.assert_array_equal(zero_batch.shot_ids, np.arange(2048, 2080, dtype=np.int64))
