"""Binary relative Pauli frames for the reviewed rotated-Z test schedules."""

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from itertools import product

import stim

from .p1_schedule import MemorySchedule


CLIFFORDS = {"CX", "H", "S", "S_DAG", "I", "X", "Y", "Z"}
PASSIVE = {"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "TICK"}
PAULI_ERRORS = {"X_ERROR": "X", "Y_ERROR": "Y", "Z_ERROR": "Z"}
NOISE = {*PAULI_ERRORS, "DEPOLARIZE1", "DEPOLARIZE2"}


class PauliFrame:
    def __init__(self, qubits: int):
        self.x = [0] * qubits
        self.z = [0] * qubits

    def error(self, qubit: int, pauli: str) -> None:
        self.x[qubit] ^= pauli in "XY"
        self.z[qubit] ^= pauli in "ZY"

    def reset_z(self, qubit: int) -> None:
        self.x[qubit] = self.z[qubit] = 0

    def measure_z(self, qubit: int, *, reset: bool, record_flip: bool = False) -> int:
        result = self.x[qubit] ^ record_flip
        # Z is a phase on the projected eigenstate. Keep X as the relative
        # eigenstate flip unless the simultaneous reset prepares a fresh state.
        self.z[qubit] = 0
        if reset:
            self.x[qubit] = 0
        return result

    def clifford(self, gate: str, targets: tuple[int, ...], *, backwards: bool = False) -> None:
        if gate == "CX":
            pairs = list(zip(targets[::2], targets[1::2], strict=True))
            for control, target in reversed(pairs) if backwards else pairs:
                self.x[target] ^= self.x[control]
                self.z[control] ^= self.z[target]
        elif gate == "H":
            for qubit in targets:
                self.x[qubit], self.z[qubit] = self.z[qubit], self.x[qubit]
        elif gate in ("S", "S_DAG"):
            for qubit in targets:
                self.z[qubit] ^= self.x[qubit]
        elif gate not in ("I", "X", "Y", "Z"):
            raise ValueError(f"unsupported Clifford {gate}")
        # Intended Pauli gates can change the frame's phase but leave x and z unchanged.


@dataclass(frozen=True)
class Fault:
    instruction_index: int
    target_slot: int
    channel: str
    qubits: tuple[int, ...]
    outcome: str


@dataclass(frozen=True)
class Signature:
    detectors: tuple[int, ...]
    global_label: int
    snapshots: tuple[int, ...]
    cores: tuple[int, ...]


def elementary_faults(physical: stim.Circuit) -> tuple[Fault, ...]:
    """List every nonidentity outcome for each generated channel and record slot."""
    faults = []
    for index, instruction in enumerate(physical):
        name = instruction.name
        qubits = tuple(target.value for target in instruction.targets_copy())
        if name == "DEPOLARIZE2":
            for slot, pair in enumerate(zip(qubits[::2], qubits[1::2], strict=True)):
                faults.extend(
                    Fault(index, slot, name, pair, first + second)
                    for first, second in product("IXYZ", repeat=2)
                    if (first, second) != ("I", "I")
                )
        elif name == "DEPOLARIZE1":
            for slot, qubit in enumerate(qubits):
                faults.extend(Fault(index, slot, name, (qubit,), pauli) for pauli in "XYZ")
        elif name in PAULI_ERRORS:
            faults.extend(
                Fault(index, slot, name, (qubit,), PAULI_ERRORS[name])
                for slot, qubit in enumerate(qubits)
            )
        elif name in ("MR", "M"):
            faults.extend(
                Fault(index, slot, f"{name}_RESULT", (qubit,), "FLIP")
                for slot, qubit in enumerate(qubits)
            )
        elif name not in CLIFFORDS | PASSIVE | {"R"}:
            raise ValueError(f"unreviewed instruction {name}")
    return tuple(faults)


def _by_instruction(faults: tuple[Fault, ...]) -> dict[int, list[Fault]]:
    grouped = defaultdict(list)
    seen = set()
    for fault in faults:
        location = (fault.instruction_index, fault.target_slot)
        if location in seen:
            raise ValueError("a location can have only one fault outcome")
        seen.add(location)
        grouped[fault.instruction_index].append(fault)
    return grouped


def _parity(records: list[int], indices: tuple[int, ...]) -> int:
    return sum(records[index] for index in indices) % 2


def _propagate(
    physical: stim.Circuit, schedule: MemorySchedule, faults: tuple[Fault, ...]
) -> tuple[list[int], tuple[int, ...]]:
    frame = PauliFrame(physical.num_qubits)
    records = []
    snapshots = []
    grouped = _by_instruction(faults)
    cut_offsets = set(schedule.cut_offsets)
    for index, instruction in enumerate(physical):
        name = instruction.name
        active = grouped.get(index, ())
        flipped = set()
        for fault in active:
            if fault.outcome == "FLIP":
                flipped.add(fault.target_slot)
            else:
                for qubit, pauli in zip(fault.qubits, fault.outcome, strict=True):
                    frame.error(qubit, pauli)

        if name in ("R", "MR", "M"):
            for slot, target in enumerate(instruction.targets_copy()):
                if name == "R":
                    frame.reset_z(target.value)
                else:
                    records.append(frame.measure_z(target.value, reset=name == "MR", record_flip=slot in flipped))
        elif name in CLIFFORDS:
            frame.clifford(name, tuple(target.value for target in instruction.targets_copy()))
        elif name not in NOISE | PASSIVE:
            raise ValueError(f"unreviewed instruction {name}")

        if index + 1 in cut_offsets:
            snapshots.append(_parity(frame.x, schedule.logical_support))
    return records, tuple(snapshots)


def oracle_signature(
    physical: stim.Circuit, schedule: MemorySchedule, faults: tuple[Fault, ...]
) -> Signature:
    records, snapshots = _propagate(physical, schedule, faults)
    global_label = _parity(records, schedule.logical_records)

    # Derive core labels by isolating each interval's faults and checking their
    # final readout effect. This deliberately avoids differencing snapshots.
    core_faults = defaultdict(list)
    for fault in faults:
        core_faults[bisect_right(schedule.cut_offsets, fault.instruction_index)].append(fault)
    cores = [0] * (len(schedule.cut_offsets) + 1)
    if len(core_faults) == 1:
        cores[next(iter(core_faults))] = global_label
    elif len(core_faults) > 1:
        for core, isolated in core_faults.items():
            isolated_records, _ = _propagate(physical, schedule, tuple(isolated))
            cores[core] = _parity(isolated_records, schedule.logical_records)
    return Signature(
        tuple(_parity(records, dependencies) for dependencies in schedule.detector_records()),
        global_label, snapshots, tuple(cores),
    )


def assert_rounds_preserve_logical_z(physical: stim.Circuit, schedule: MemorySchedule) -> None:
    """Pull the physical Z representative back through each complete Clifford layer."""
    starts = (schedule.preparation_stop, *(index + 1 for index in schedule.mr_instructions[:-1]))
    for first, last in zip(starts, schedule.mr_instructions, strict=True):
        operator = PauliFrame(physical.num_qubits)
        for qubit in schedule.logical_support:
            operator.error(qubit, "Z")
        for instruction in reversed(list(physical[first:last])):
            name = instruction.name
            if name in CLIFFORDS:
                operator.clifford(name, tuple(target.value for target in instruction.targets_copy()), backwards=True)
            elif name not in PASSIVE | NOISE:
                raise ValueError(f"unexpected instruction inside extraction round: {name}")
        assert not any(operator.x)
        assert tuple(qubit for qubit, bit in enumerate(operator.z) if bit) == schedule.logical_support


def inject_faults(teacher: stim.Circuit, faults: tuple[Fault, ...]) -> stim.Circuit:
    """Preserve teacher cuts while replacing stochastic channels by chosen faults."""
    result = stim.Circuit()
    pauli_target = {"X": stim.target_x, "Y": stim.target_y, "Z": stim.target_z}
    grouped = _by_instruction(faults)
    physical_index = 0
    for instruction in teacher:
        name = instruction.name
        if name == "OBSERVABLE_INCLUDE" and instruction.gate_args_copy() != [0]:
            result.append(instruction)
            continue
        active = grouped.get(physical_index, ())
        physical_index += 1
        for fault in active:
            if fault.outcome != "FLIP":
                targets = [
                    pauli_target[pauli](qubit)
                    for qubit, pauli in zip(fault.qubits, fault.outcome, strict=True)
                    if pauli != "I"
                ]
                result.append("E", targets, 1)
        if name in NOISE:
            continue
        if name in ("MR", "M"):
            flipped = {fault.target_slot for fault in active if fault.outcome == "FLIP"}
            targets = instruction.targets_copy()
            if not flipped:
                result.append(name, targets)
            else:
                for slot, target in enumerate(targets):
                    result.append(name, [target], [1] if slot in flipped else [])
        else:
            result.append(instruction)
    if any(index >= physical_index for index in grouped):
        raise ValueError("fault outside the physical circuit")
    return result
