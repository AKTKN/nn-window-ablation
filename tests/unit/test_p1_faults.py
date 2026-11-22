from collections import Counter
from dataclasses import asdict, replace
import json

import pytest

from nn_window_ablation.simulation import CircuitSpec, build_circuit_bundle
from tests.p1_checks import assert_signatures_match, sample_fault_signature
from tests.reference.p1_pauli_frame import Fault, Signature, elementary_faults, oracle_signature
from tests.reference.p1_schedule import SMALL


BEFORE_CUT = Fault(100, 0, "GAP", (1,), "X")
AFTER_CUT = Fault(101, 0, "GAP", (1,), "X")
PREPARATION = Fault(18, 0, "X_ERROR", (1,), "X")
READOUT = Fault(204, 0, "M_RESULT", (1,), "FLIP")


@pytest.fixture(scope="module")
def fixture_circuits():
    bundle = build_circuit_bundle(CircuitSpec(3, 6, 3, 0.003))
    return bundle, bundle.original.to_stim().flattened()


def _check_faults(fixture_circuits, *faults):
    bundle, physical = fixture_circuits
    expected = oracle_signature(physical, SMALL, faults)
    actual = sample_fault_signature(bundle, *faults)
    assert_signatures_match(actual, expected)
    return actual


def test_p1_single_fault_signatures_against_oracle(fixture_circuits, tmp_path):
    bundle, physical = fixture_circuits
    cases = elementary_faults(physical)
    counts = Counter(case.channel for case in cases)
    # Six rounds with 24 paired locations, 17 single depolarizers and eight MR
    # slots per round, plus initial resets and the nine destructive readouts.
    assert counts == {"DEPOLARIZE2": 2160, "DEPOLARIZE1": 306, "X_ERROR": 122, "MR_RESULT": 48, "M_RESULT": 9}
    assert len(set(cases)) == len(cases) == 2645
    outcomes_by_location = Counter((case.channel, case.instruction_index, case.target_slot) for case in cases)
    assert Counter(outcomes_by_location.values()) == {15: 144, 3: 102, 1: 179}

    for case in cases:
        observed = sample_fault_signature(bundle, case)
        expected = oracle_signature(physical, SMALL, (case,))
        try:
            assert_signatures_match(observed, expected)
        except AssertionError as exc:
            raise AssertionError(f"{case}: {exc}") from exc

    inventory = {
        "stim_version": bundle.stim_version,
        "original_circuit_sha256": bundle.original.sha256,
        "spec": asdict(bundle.spec),
        "counts_by_channel": counts,
        "tested_outcomes": [asdict(case) for case in cases],
    }
    (tmp_path / "fault_coverage.json").write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")


def test_p1_clean_cut_ownership(fixture_circuits):
    before = _check_faults(fixture_circuits, BEFORE_CUT)
    after = _check_faults(fixture_circuits, AFTER_CUT)
    assert (before.global_label, before.snapshots, before.cores) == (1, (1,), (1, 0))
    assert (after.global_label, after.snapshots, after.cores) == (1, (0,), (0, 1))

    # A record flip on Z-check qubit 9 is visible in rounds 3 and 4. Reset
    # clears the qubit. A fresh post-reset X fault moves the syndrome to rounds 4 and 5.
    record_flip = Fault(100, 1, "MR_RESULT", (9,), "FLIP")
    reset_fault = Fault(101, 1, "X_ERROR", (9,), "X")
    recorded = _check_faults(fixture_circuits, record_flip)
    reset = _check_faults(fixture_circuits, reset_fault)
    assert tuple(index for index, bit in enumerate(recorded.detectors) if bit) == (13, 21)
    assert tuple(index for index, bit in enumerate(reset.detectors) if bit) == (21, 29)
    assert recorded.cores == reset.cores == (0, 0)


@pytest.mark.parametrize(
    ("fault", "expected"),
    [
        (PREPARATION, (1, (1,), (1, 0))),
        (Fault(18, 3, "X_ERROR", (8,), "X"), (0, (0,), (0, 0))),
        (READOUT, (1, (0,), (0, 1))),
        (Fault(204, 3, "M_RESULT", (8,), "FLIP"), (0, (0,), (0, 0))),
    ],
)
def test_p1_preparation_and_readout_ownership(fixture_circuits, fault, expected):
    signature = _check_faults(fixture_circuits, fault)
    assert (signature.global_label, signature.snapshots, signature.cores) == expected


@pytest.mark.parametrize(
    "faults",
    [
        (BEFORE_CUT, AFTER_CUT),
        (PREPARATION, Fault(27, 0, "DEPOLARIZE2", (2, 3), "YX"), READOUT),
        (Fault(100, 1, "MR_RESULT", (9,), "FLIP"), Fault(101, 1, "X_ERROR", (9,), "X"), READOUT),
    ],
)
def test_p1_composed_fault_signatures(fixture_circuits, faults):
    _, physical = fixture_circuits
    elementary = [oracle_signature(physical, SMALL, (fault,)) for fault in faults]

    def parity_of_fields(field):
        return tuple(sum(column) % 2 for column in zip(*(getattr(signature, field) for signature in elementary)))

    xor_of_elementary = Signature(
        parity_of_fields("detectors"),
        sum(signature.global_label for signature in elementary) % 2,
        parity_of_fields("snapshots"),
        parity_of_fields("cores"),
    )
    assert_signatures_match(_check_faults(fixture_circuits, *faults), xor_of_elementary)


def test_p1_core_oracle_rejects_bad_preparation_bad_residual_and_xor_preserving_labels(fixture_circuits):
    bundle, physical = fixture_circuits
    expected = oracle_signature(physical, SMALL, (PREPARATION,))
    observed = sample_fault_signature(bundle, PREPARATION)
    assert_signatures_match(observed, expected)
    assert observed.global_label == 1 and observed.cores == (1, 0)

    omitted_preparation = replace(observed, cores=(0, observed.cores[-1]))
    incorrect_residual = replace(observed, cores=(observed.cores[0], observed.global_label))
    equal_global_parity = replace(observed, cores=tuple(bit ^ 1 for bit in observed.cores))
    assert sum(equal_global_parity.cores) % 2 == observed.global_label
    for corrupt in (omitted_preparation, incorrect_residual, equal_global_parity):
        with pytest.raises(AssertionError, match="interval core labels"):
            assert_signatures_match(corrupt, expected)
