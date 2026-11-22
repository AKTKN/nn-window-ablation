from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest
import stim

from nn_window_ablation.simulation import CircuitSpec, build_circuit_bundle, sample_shots
from tests.p1_checks import assert_fixture_metadata, assert_teacher_annotations, sample_fault_signature
from tests.reference.p1_pauli_frame import Fault, assert_rounds_preserve_logical_z
from tests.reference.p1_schedule import SMALL


@pytest.fixture(scope="module")
def bundle():
    return build_circuit_bundle(CircuitSpec(3, 6, 3, 0.003))


def test_p1_annotations_preserve_physical_circuit(bundle):
    generated = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=3, rounds=6, after_clifford_depolarization=0.003,
        before_round_data_depolarization=0.003, before_measure_flip_probability=0.003,
        after_reset_flip_probability=0.003,
    )
    assert bundle.original.to_stim() == generated
    assert_fixture_metadata(bundle, SMALL)
    assert_rounds_preserve_logical_z(generated.flattened(), SMALL)
    assert bundle.stim_version == stim.__version__ == "1.15.0"
    for artifact in (bundle.original, bundle.teacher):
        assert artifact.sha256 == sha256(artifact.text.encode()).hexdigest()
        assert str(artifact.to_stim()) == artifact.text
    assert bundle.original.sha256 != bundle.teacher.sha256
    assert bundle.original.to_stim().num_observables == 1
    assert bundle.teacher.to_stim().num_observables == 2
    baseline_dem = bundle.global_detector_error_model(decompose_errors=True)
    assert baseline_dem == generated.detector_error_model(decompose_errors=True)
    assert (baseline_dem.num_detectors, baseline_dem.num_observables) == (48, 1)


@pytest.mark.parametrize("rounds, core_count", [(3, 1), (6, 2)])
def test_p1_noiseless_reference_and_single_core(rounds, core_count):
    circuit = build_circuit_bundle(CircuitSpec(3, rounds, 3, 0))
    batch = sample_shots(circuit, 16, seed=17, first_shot_id=80, include_snapshots=True)
    assert batch.detectors.shape == (16, rounds * 8)
    assert batch.global_labels.shape == (16,)
    assert batch.core_labels.shape == (16, core_count)
    assert batch.snapshots.shape == (16, core_count - 1)
    for bits in (batch.detectors, batch.global_labels, batch.core_labels, batch.snapshots):
        assert bits.dtype == np.bool_
        assert not bits.any()
    np.testing.assert_array_equal(batch.shot_ids, np.arange(80, 96, dtype=np.int64))

    if core_count == 1:
        # A flipped record for the first data qubit changes the sole logical parity.
        fault = Fault(circuit.destructive_readout_instruction, 0, "M_RESULT", (1,), "FLIP")
        signature = sample_fault_signature(circuit, fault)
        assert (signature.global_label, signature.snapshots, signature.cores) == (1, (), (1,))


def test_p1_structural_checker_rejects_displaced_cut_or_wrong_support(bundle):
    teacher = bundle.teacher.to_stim()
    physical = bundle.original.to_stim().flattened()
    cut_index = SMALL.cut_offsets[0]
    delayed = teacher[:cut_index] + teacher[cut_index + 1:cut_index + 2] + teacher[cut_index:cut_index + 1]
    delayed += teacher[cut_index + 2:]
    with pytest.raises(AssertionError, match="cut offsets"):
        assert_teacher_annotations(delayed, physical, SMALL)

    wrong_support = teacher[:cut_index]
    wrong_support.append("OBSERVABLE_INCLUDE", [stim.target_z(q) for q in (3, 5, 8)], 1)
    wrong_support += teacher[cut_index + 1:]
    with pytest.raises(AssertionError, match="logical support"):
        assert_teacher_annotations(wrong_support, physical, SMALL)


@pytest.mark.parametrize(
    "spec",
    [(2, 6, 2, 0.003), (3, 5, 3, 0.003), (3, 6, 2, 0.003), (3, 6, 3, -0.01), (3, 6, 3, float("nan"))],
)
def test_p1_invalid_circuit_parameters_fail(spec):
    with pytest.raises(ValueError):
        CircuitSpec(*spec)


def test_p1_sampling_mapping_validates_ids_and_keeps_diagnostics_optional(bundle):
    with pytest.raises(ValueError, match="shot IDs"):
        sample_shots(bundle, 2, seed=2, first_shot_id=2**63 - 1)
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        sample_shots(bundle, 2, seed=-1)
    with pytest.raises(ValueError, match="positive integer"):
        sample_shots(bundle, 0, seed=2)

    batch = sample_shots(bundle, 4, seed=2, first_shot_id=12)
    assert batch.snapshots is None
    assert batch.shot_ids.dtype == np.int64
    np.testing.assert_array_equal(batch.shot_ids, [12, 13, 14, 15])
    with_aux = sample_shots(bundle, 4, seed=2, first_shot_id=12, include_snapshots=True)
    np.testing.assert_array_equal(batch.detectors, with_aux.detectors)
    np.testing.assert_array_equal(batch.global_labels, with_aux.global_labels)
    np.testing.assert_array_equal(batch.core_labels, with_aux.core_labels)

    incompatible = replace(bundle, detectors=bundle.detectors[:-1])
    with pytest.raises(ValueError, match="sampled columns"):
        sample_shots(incompatible, 4, seed=2)
