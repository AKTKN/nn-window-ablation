from itertools import product

import numpy as np
import pytest

from nn_window_ablation.simulation import Arrival, ArrivalPhase, CircuitSpec, build_circuit_bundle
from nn_window_ablation.windows import (
    Buffers, SlotState, TargetQuery, buffer_lengths, build_observation_layout, pack_observations,
    select_window, unpack_observations,
)
from tests.p2_checks import assert_indices, toy_bundle
from tests.reference.p2_observations import PILOT_SCHEDULE, TOY_SCHEDULE


@pytest.fixture(scope="module")
def pilot():
    return build_observation_layout(build_circuit_bundle(CircuitSpec(5, 60, 5, 0.003)))


@pytest.mark.parametrize("left,right", sorted(product((0, 3, 5), repeat=2)))
def test_p2_window_sets_against_reference(pilot, left, right):
    assert buffer_lengths(5) == (0, 3, 5)
    assert buffer_lengths(3) == (0, 2, 3)
    owned_rounds = []
    views = []
    for index in range(12):
        target = pilot.core_query(index)
        assert target == TargetQuery(index, 5 * index + 1, 5 * index + 5)
        owned_rounds.extend(range(target.first_round, target.last_round + 1))
        window = select_window(pilot, target, Buffers(left, right))
        assert_indices(window.detector_indices, PILOT_SCHEDULE.select(index, (left, right)))
        assert window.metadata.target == target
        assert window.metadata.release == (
            Arrival(60, ArrivalPhase.DESTRUCTIVE_READOUT) if index == 11
            else Arrival((index + 1) * 5 + right, ArrivalPhase.ANCILLA)
        )
        assert all(site.arrival <= window.metadata.release for site in window.detectors)
        views.append(window)
    assert owned_rounds == list(range(1, 61))
    overlap = set(views[4].detector_indices) & set(views[5].detector_indices)
    assert bool(overlap) == bool(left or right)
    if (left, right) == (5, 5):
        interior = views[4].metadata
        assert interior.last_observed_round - interior.first_observed_round + 1 == 15


def test_p2_terminal_phase_equality(pilot):
    layout = build_observation_layout(toy_bundle())
    end_at_ancilla = select_window(layout, layout.core_query(1), Buffers(0, 3))
    terminal = select_window(layout, layout.core_query(2), Buffers(0, 0))
    assert end_at_ancilla.metadata.release == Arrival(9, ArrivalPhase.ANCILLA)
    assert terminal.metadata.release == Arrival(9, ArrivalPhase.DESTRUCTIVE_READOUT)
    assert_indices(end_at_ancilla.detector_indices, (3, 4, 5, 6, 7, 8, 9))
    assert_indices(terminal.detector_indices, (7, 8, 9, 10))
    assert layout.detectors[9].coordinates[2] == layout.detectors[10].coordinates[2] == 8
    assert (
        layout.detectors[10].layer_index, layout.detectors[10].instruction_index,
        layout.detectors[10].availability_instruction, layout.detectors[10].arrival,
    ) == (9, 101, 100, Arrival(9, ArrivalPhase.DESTRUCTIVE_READOUT))
    assert 3 in end_at_ancilla.detector_indices and TOY_SCHEDULE.detector_dependencies[3] == (2, 3)

    leaked = TOY_SCHEDULE.select(1, (0, 3), ignore_readout_phase=True)
    assert leaked[-1] == 10
    with pytest.raises(AssertionError, match="permitted detector indices"):
        assert_indices(leaked, end_at_ancilla.detector_indices)

    penultimate = select_window(pilot, pilot.core_query(10), Buffers(5, 5))
    final = select_window(pilot, pilot.core_query(11), Buffers(5, 0))
    assert penultimate.metadata.release == Arrival(60, ArrivalPhase.ANCILLA)
    assert not set(range(1428, 1440)) & set(penultimate.detector_indices)
    assert set(range(1428, 1440)) <= set(final.detector_indices)
    assert penultimate.metadata.slots[-1].state is SlotState.UNAVAILABLE
    assert penultimate.metadata.slots[-1].arrival == Arrival(60, ArrivalPhase.DESTRUCTIVE_READOUT)


def test_p2_boundary_flags_follow_observations(pilot):
    first = select_window(pilot, pilot.core_query(0), Buffers(5, 5))
    second = select_window(pilot, pilot.core_query(1), Buffers(5, 5))
    interior = select_window(pilot, pilot.core_query(4), Buffers(5, 5))
    last = select_window(pilot, pilot.core_query(11), Buffers(5, 5))
    full_core = select_window(pilot, pilot.core_query(4))
    full_global = select_window(pilot, pilot.global_query)

    def check_flags(view, actual=None):
        expected_sets = set(view.detector_indices)
        expected = (bool(expected_sets & set(range(12))), bool(expected_sets & set(range(1428, 1440))))
        flags = (view.metadata.preparation, view.metadata.destructive_readout) if actual is None else actual
        assert flags == expected, "physical boundary flags"
        return flags

    assert [check_flags(view) for view in (first, second, interior, last, full_core, full_global)] == [
        (True, False), (True, False), (False, False), (False, True), (True, True), (True, True),
    ]
    with pytest.raises(AssertionError, match="physical boundary flags"):
        check_flags(second, (second.metadata.target.core_index == 0, False))
    assert full_core.metadata.slots == full_global.metadata.slots
    assert full_core.metadata.target != full_global.metadata.target


def test_p2_rejects_off_partition_queries_and_unsupported_buffers(pilot):
    with pytest.raises(ValueError, match="bounds"):
        select_window(pilot, TargetQuery(1, 5, 10), Buffers(0, 0))
    with pytest.raises(ValueError, match="global query"):
        select_window(pilot, pilot.global_query, Buffers(0, 0))
    with pytest.raises(ValueError, match="buffer lengths"):
        select_window(pilot, pilot.core_query(1), Buffers(2, 0))


def test_p2_pack_unpack_preserves_non_byte_aligned_detector_order_and_ids():
    bits = np.zeros((2, 11), dtype=np.bool_)
    bits[0, (0, 7, 8, 10)] = True
    bits[1, 3] = True
    ids = np.array([101, 17], dtype=np.int64)
    packed = pack_observations(bits, ids)
    np.testing.assert_array_equal(packed.bits, np.array([[129, 5], [8, 0]], dtype=np.uint8))
    restored, restored_ids = unpack_observations(packed)
    np.testing.assert_array_equal(restored, bits)
    np.testing.assert_array_equal(restored_ids, ids)
