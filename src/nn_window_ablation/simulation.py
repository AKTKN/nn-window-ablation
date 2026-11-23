"""Rotated surface-code Z memory and jointly sampled circuit-level labels."""

from dataclasses import dataclass, field
from enum import IntEnum
from hashlib import sha256

import numpy as np
from numpy.typing import NDArray
import stim

from .timing import span


@dataclass(frozen=True, slots=True)
class CircuitSpec:
    distance: int
    rounds: int
    core_length: int
    probability: float

    def __post_init__(self) -> None:
        if type(self.distance) is not int or self.distance < 3 or self.distance % 2 == 0:
            raise ValueError("distance must be an odd integer of at least 3")
        if type(self.core_length) is not int or self.core_length != self.distance:
            raise ValueError("core length must equal distance")
        if type(self.rounds) is not int or self.rounds < 1 or self.rounds % self.core_length:
            raise ValueError("rounds must form a positive number of whole cores")
        if isinstance(self.probability, bool) or not 0 <= self.probability <= 1:
            raise ValueError("circuit probability must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class CircuitArtifact:
    text: str
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sha256", sha256(self.text.encode("utf-8")).hexdigest())

    def to_stim(self) -> stim.Circuit:
        """Return an independent copy of the exact serialized circuit."""
        return stim.Circuit(self.text)


class ArrivalPhase(IntEnum):
    ANCILLA = 0
    DESTRUCTIVE_READOUT = 1


@dataclass(frozen=True, order=True, slots=True)
class Arrival:
    """One-based completed-round number and the phase within that boundary."""

    round: int
    phase: ArrivalPhase


@dataclass(frozen=True, slots=True)
class Measurement:
    record_index: int
    qubit: int
    instruction_index: int
    arrival: Arrival


@dataclass(frozen=True, slots=True)
class Detector:
    detector_index: int
    instruction_index: int
    coordinates: tuple[float, ...]
    measurement_indices: tuple[int, ...]
    arrival: Arrival
    available_after_instruction: int


@dataclass(frozen=True, slots=True)
class TemporalCut:
    round: int
    instruction_offset: int
    teacher_instruction_index: int
    observable_index: int


@dataclass(frozen=True, slots=True)
class CoreInterval:
    core_index: int
    first_round: int
    last_round: int
    start_instruction: int
    stop_instruction: int


@dataclass(frozen=True, slots=True)
class Observable:
    observable_index: int
    original_instruction_index: int | None
    teacher_instruction_index: int
    logical_support: tuple[int, ...]
    measurement_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CircuitBundle:
    """Circuit artifacts and metadata in original flattened instruction order.

    Instruction and record indices are zero-based. A cut offset is a gap in
    the physical instruction stream, immediately before the instruction at
    that offset. Core instruction intervals are half-open. A field named
    teacher_instruction_index instead indexes the flattened teacher stream.
    """

    spec: CircuitSpec
    stim_version: str
    original: CircuitArtifact
    teacher: CircuitArtifact
    measurements: tuple[Measurement, ...]
    detectors: tuple[Detector, ...]
    observables: tuple[Observable, ...]
    cores: tuple[CoreInterval, ...]
    cuts: tuple[TemporalCut, ...]
    preparation_stop: int
    round_measurement_instructions: tuple[int, ...]
    destructive_readout_instruction: int

    @property
    def logical_support(self) -> tuple[int, ...]:
        return self.observables[0].logical_support

    def global_detector_error_model(self, *, decompose_errors: bool = False) -> stim.DetectorErrorModel:
        """Build the baseline DEM from the original global-only circuit."""
        return self.original.to_stim().detector_error_model(decompose_errors=decompose_errors)


@dataclass(frozen=True, slots=True)
class ShotBatch:
    detectors: NDArray[np.bool_]
    global_labels: NDArray[np.bool_]
    core_labels: NDArray[np.bool_]
    shot_ids: NDArray[np.int64]
    snapshots: NDArray[np.bool_] | None


@dataclass(slots=True)
class _PhysicalStructure:
    measurements: list[Measurement] = field(default_factory=list)
    detectors: list[Detector] = field(default_factory=list)
    round_ends: list[int] = field(default_factory=list)
    preparation_stop: int | None = None
    destructive_readout: int | None = None
    global_observable: int | None = None
    logical_records: tuple[int, ...] = ()


def _record_indices(instruction: stim.CircuitInstruction, record_count: int) -> tuple[int, ...]:
    targets = instruction.targets_copy()
    if not targets or any(not target.is_measurement_record_target for target in targets):
        raise ValueError(f"expected measurement record targets in {instruction.name}")
    indices = tuple(record_count + target.value for target in targets)
    if any(not 0 <= index < record_count for index in indices):
        raise ValueError(f"invalid measurement dependency in {instruction.name}")
    return indices


def _inspect_physical_circuit(flattened: stim.Circuit, spec: CircuitSpec) -> _PhysicalStructure:
    structure = _PhysicalStructure()
    for index, instruction in enumerate(flattened):
        name = instruction.name
        if name == "TICK" and structure.preparation_stop is None:
            structure.preparation_stop = index
        elif name in ("MR", "M"):
            if structure.destructive_readout is not None:
                raise ValueError("unexpected measurement after the destructive readout")
            if name == "MR":
                structure.round_ends.append(index)
                arrival = Arrival(len(structure.round_ends), ArrivalPhase.ANCILLA)
            else:
                if len(structure.round_ends) != spec.rounds:
                    raise ValueError("destructive readout must follow all ancilla rounds")
                structure.destructive_readout = index
                arrival = Arrival(spec.rounds, ArrivalPhase.DESTRUCTIVE_READOUT)
            for target in instruction.targets_copy():
                record_index = len(structure.measurements)
                structure.measurements.append(Measurement(record_index, target.value, index, arrival))
        elif name == "DETECTOR":
            dependencies = _record_indices(instruction, len(structure.measurements))
            last = structure.measurements[max(dependencies)]
            structure.detectors.append(Detector(
                detector_index=len(structure.detectors),
                instruction_index=index,
                coordinates=tuple(instruction.gate_args_copy()),
                measurement_indices=dependencies,
                arrival=last.arrival,
                available_after_instruction=last.instruction_index,
            ))
        elif name == "OBSERVABLE_INCLUDE":
            if structure.global_observable is not None or instruction.gate_args_copy() != [0]:
                raise ValueError("expected one original global observable at index 0")
            dependencies = _record_indices(instruction, len(structure.measurements))
            if any(
                structure.measurements[record].arrival.phase != ArrivalPhase.DESTRUCTIVE_READOUT
                for record in dependencies
            ):
                raise ValueError("the global observable must use destructive-readout records")
            structure.global_observable = index
            structure.logical_records = dependencies
    if (
        structure.preparation_stop is None
        or structure.destructive_readout is None
        or structure.global_observable is None
        or len(structure.measurements) != flattened.num_measurements
    ):
        raise ValueError("unsupported rotated-memory measurement schedule")
    return structure


def build_circuit_bundle(spec: CircuitSpec) -> CircuitBundle:
    """Generate a physical circuit and insert a Z-parity snapshot at each clean cut."""
    probability = spec.probability
    original = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=spec.distance, rounds=spec.rounds,
        after_clifford_depolarization=probability,
        before_round_data_depolarization=probability,
        before_measure_flip_probability=probability,
        after_reset_flip_probability=probability,
    )
    flattened = original.flattened()
    structure = _inspect_physical_circuit(flattened, spec)
    support = tuple(sorted(structure.measurements[index].qubit for index in structure.logical_records))
    if len(support) != spec.distance or len(set(support)) != len(support):
        raise ValueError("unexpected support for the physical logical Z observable")
    terms = [stim.target_z(qubit) for qubit in support]
    rounds_by_offset = {
        structure.round_ends[round_number - 1] + 1: round_number
        for round_number in range(spec.core_length, spec.rounds, spec.core_length)
    }
    teacher = stim.Circuit()
    cuts = []
    for offset, instruction in enumerate(flattened, 1):
        teacher.append(instruction)
        if (round_number := rounds_by_offset.get(offset)) is not None:
            observable_index = len(cuts) + 1
            cuts.append(TemporalCut(round_number, offset, len(teacher), observable_index))
            teacher.append("OBSERVABLE_INCLUDE", terms, observable_index)

    offsets = [0, *(cut.instruction_offset for cut in cuts), len(flattened)]
    cores = tuple(
        CoreInterval(index, index * spec.core_length + 1, (index + 1) * spec.core_length, start, stop)
        for index, (start, stop) in enumerate(zip(offsets, offsets[1:]))
    )
    # The scanner requires these physical boundaries in every supported circuit.
    assert structure.global_observable is not None
    assert structure.preparation_stop is not None
    assert structure.destructive_readout is not None
    observables = (
        Observable(
            0, structure.global_observable, structure.global_observable + len(cuts), support, structure.logical_records,
        ),
        *(Observable(cut.observable_index, None, cut.teacher_instruction_index, support, ()) for cut in cuts),
    )
    return CircuitBundle(
        spec, stim.__version__, CircuitArtifact(str(original)), CircuitArtifact(str(teacher)),
        tuple(structure.measurements), tuple(structure.detectors), observables, cores, tuple(cuts),
        structure.preparation_stop, tuple(structure.round_ends), structure.destructive_readout,
    )


def sample_shots(
    bundle: CircuitBundle,
    shots: int,
    *,
    seed: int,
    first_shot_id: int = 0,
    include_snapshots: bool = False,
) -> ShotBatch:
    """Draw aligned detector and label rows with caller-assigned consecutive shot IDs."""
    if type(shots) is not int or shots < 1:
        raise ValueError("shots must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if type(first_shot_id) is not int or not 0 <= first_shot_id <= 2**63 - shots:
        raise ValueError("shot IDs must fit in the nonnegative int64 range")

    with span("sampler_compile"):
        sampler = bundle.teacher.to_stim().compile_detector_sampler(seed=seed)
    with span("stim_sample"):
        detectors, observables = sampler.sample(shots, separate_observables=True)
    if detectors.shape != (shots, len(bundle.detectors)) or observables.shape != (shots, len(bundle.cores)):
        raise ValueError("sampled columns do not match the circuit bundle")

    with span("labels_and_ids"):
        global_labels = observables[:, 0].copy()
        snapshots = observables[:, 1:]
        core_labels = np.empty_like(observables)
        core_labels[:, :-1] = snapshots
        core_labels[:, -1] = global_labels
        core_labels[:, 1:] ^= snapshots
        shot_ids = np.arange(shots, dtype=np.int64) + first_shot_id
        return ShotBatch(detectors, global_labels, core_labels, shot_ids, snapshots.copy() if include_snapshots else None)
