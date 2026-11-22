"""Connect the independent signatures and schedules to the production interface."""

from dataclasses import replace

import stim

from nn_window_ablation.simulation import Arrival, ArrivalPhase, CircuitArtifact, CircuitBundle, sample_shots
from tests.reference.p1_pauli_frame import Fault, Signature, inject_faults
from tests.reference.p1_schedule import MemorySchedule


def sample_fault_signature(bundle: CircuitBundle, *faults: Fault) -> Signature:
    injected = inject_faults(bundle.teacher.to_stim(), faults)
    faulted_bundle = replace(bundle, teacher=CircuitArtifact(str(injected)))
    sample = sample_shots(faulted_bundle, 1, seed=31, include_snapshots=True)
    return Signature(
        tuple(map(int, sample.detectors[0])),
        int(sample.global_labels[0]),
        tuple(map(int, sample.snapshots[0])),
        tuple(map(int, sample.core_labels[0])),
    )


def assert_signatures_match(observed: Signature, expected: Signature) -> None:
    assert observed.detectors == expected.detectors, "detector signature"
    assert observed.global_label == expected.global_label, "global label"
    assert observed.snapshots == expected.snapshots, "cumulative snapshot labels"
    assert observed.cores == expected.cores, "interval core labels"


def assert_teacher_annotations(
    teacher: stim.Circuit, physical: stim.Circuit, schedule: MemorySchedule,
) -> None:
    seen = []
    stripped = stim.Circuit()
    offset = 0
    for instruction in teacher:
        if instruction.name == "OBSERVABLE_INCLUDE" and instruction.gate_args_copy() != [0]:
            targets = instruction.targets_copy()
            assert all(target.is_z_target for target in targets), "logical Z terms"
            assert tuple(target.value for target in targets) == schedule.logical_support, "logical support"
            seen.append((offset, int(instruction.gate_args_copy()[0])))
        else:
            stripped.append(instruction)
            offset += 1

    assert stripped == physical, "physical instructions and measurement order"
    assert seen == list(zip(schedule.cut_offsets, range(1, len(schedule.cut_offsets) + 1))), "cut offsets"


def assert_fixture_metadata(bundle: CircuitBundle, schedule: MemorySchedule) -> None:
    physical = bundle.original.to_stim().flattened()
    teacher = bundle.teacher.to_stim()
    assert_teacher_annotations(teacher, physical, schedule)
    assert bundle.preparation_stop == schedule.preparation_stop
    assert bundle.round_measurement_instructions == schedule.mr_instructions
    assert bundle.destructive_readout_instruction == schedule.readout_instruction
    assert bundle.logical_support == schedule.logical_support
    assert tuple(sorted(bundle.observables[0].measurement_indices)) == schedule.logical_records
    assert physical[bundle.observables[0].original_instruction_index].gate_args_copy() == [0]
    assert teacher[bundle.observables[0].teacher_instruction_index].gate_args_copy() == [0]
    assert len(bundle.observables) == len(schedule.cut_offsets) + 1

    for ordinal, (cut, offset) in enumerate(zip(bundle.cuts, schedule.cut_offsets, strict=True), 1):
        assert (cut.round, cut.instruction_offset, cut.observable_index) == (ordinal * schedule.distance, offset, ordinal)
        assert physical[offset - 1].name == "MR"
        assert physical[offset].name == "X_ERROR"
        assert teacher[cut.teacher_instruction_index - 1].name == "MR"
        assert teacher[cut.teacher_instruction_index].gate_args_copy() == [ordinal]
        assert teacher[cut.teacher_instruction_index + 1].name == "X_ERROR"
        assert bundle.observables[ordinal].teacher_instruction_index == cut.teacher_instruction_index
        assert bundle.observables[ordinal].measurement_indices == ()
        assert bundle.observables[ordinal].original_instruction_index is None

    offsets = (0, *schedule.cut_offsets, len(physical))
    assert [
        (core.core_index, core.first_round, core.last_round, core.start_instruction, core.stop_instruction)
        for core in bundle.cores
    ] == [
        (index, index * schedule.distance + 1, (index + 1) * schedule.distance, start, end)
        for index, (start, end) in enumerate(zip(offsets, offsets[1:]))
    ]

    expected_measurements = [
        (qubit, instruction, Arrival(round_number, ArrivalPhase.ANCILLA))
        for round_number, instruction in enumerate(schedule.mr_instructions, 1)
        for qubit in schedule.ancillas
    ] + [
        (qubit, schedule.readout_instruction, Arrival(schedule.rounds, ArrivalPhase.DESTRUCTIVE_READOUT))
        for qubit in schedule.data
    ]
    assert len(bundle.measurements) == len(expected_measurements) == teacher.num_measurements
    for index, (measurement, expected) in enumerate(zip(bundle.measurements, expected_measurements, strict=True)):
        assert measurement.record_index == index
        assert (measurement.qubit, measurement.instruction_index, measurement.arrival) == expected

    expected_coordinates = physical.get_detector_coordinates()
    for index, (detector, dependencies) in enumerate(zip(bundle.detectors, schedule.detector_records(), strict=True)):
        assert detector.detector_index == index
        assert detector.coordinates == tuple(expected_coordinates[index])
        assert detector.measurement_indices == dependencies
        assert physical[detector.instruction_index].name == "DETECTOR"
        assert detector.available_after_instruction == expected_measurements[max(dependencies)][1]
        assert detector.arrival == expected_measurements[max(dependencies)][2]
