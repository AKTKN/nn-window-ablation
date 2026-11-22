"""Reviewed schedules from flattened, noisy Stim 1.15 rotated-Z fixtures.

Qubit order and the joint parity of each closing Z check are specified
explicitly. Round-end indices count flattened physical instructions, including
noise and annotations. These constants are independent of production metadata.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MemorySchedule:
    distance: int
    data: tuple[int, ...]
    ancillas: tuple[int, ...]
    logical_support: tuple[int, ...]
    # In detector order, each initial/closing Z check names its ancilla slot
    # and the data qubits forming the closing parity.
    z_checks: tuple[tuple[int, tuple[int, ...]], ...]
    preparation_stop: int
    mr_instructions: tuple[int, ...]
    readout_instruction: int

    @property
    def rounds(self) -> int:
        return len(self.mr_instructions)

    @property
    def cut_offsets(self) -> tuple[int, ...]:
        return tuple(index + 1 for index in self.mr_instructions[self.distance - 1:-1:self.distance])

    @property
    def readout_record_start(self) -> int:
        return len(self.ancillas) * self.rounds

    @property
    def logical_records(self) -> tuple[int, ...]:
        return tuple(self.readout_record_start + self.data.index(qubit) for qubit in self.logical_support)

    def detector_records(self) -> tuple[tuple[int, ...], ...]:
        width = len(self.ancillas)
        opening = [(check,) for check, _ in self.z_checks]
        bulk = [
            (current * width + check, (current - 1) * width + check)
            for current in range(1, self.rounds)
            for check in range(width)
        ]
        closing = [
            tuple(self.readout_record_start + self.data.index(qubit) for qubit in qubits)
            + ((self.rounds - 1) * width + check,)
            for check, qubits in self.z_checks
        ]
        return tuple(opening + bulk + closing)


SMALL = MemorySchedule(
    distance=3,
    data=(1, 3, 5, 8, 10, 12, 15, 17, 19),
    ancillas=(2, 9, 11, 13, 14, 16, 18, 25),
    logical_support=(1, 3, 5),
    z_checks=((4, (15, 8)), (1, (10, 8, 3, 1)), (6, (19, 17, 12, 10)), (3, (12, 5))),
    preparation_stop=21,
    mr_instructions=(42, 69, 100, 131, 162, 193),
    readout_instruction=204,
)


PILOT = MemorySchedule(
    distance=5,
    data=(1, 3, 5, 7, 9, 12, 14, 16, 18, 20, 23, 25, 27, 29, 31, 34, 36, 38, 40, 42, 45, 47, 49, 51, 53),
    ancillas=(2, 6, 13, 15, 17, 19, 21, 22, 24, 26, 28, 30, 35, 37, 39, 41, 43, 44, 46, 48, 50, 52, 59, 63),
    logical_support=(1, 3, 5, 7, 9),
    z_checks=(
        (7, (23, 12)), (17, (45, 34)), (2, (14, 12, 3, 1)), (12, (36, 34, 25, 23)),
        (9, (27, 25, 16, 14)), (19, (49, 47, 38, 36)), (4, (18, 16, 7, 5)),
        (14, (40, 38, 29, 27)), (11, (31, 29, 20, 18)), (21, (53, 51, 42, 40)),
        (6, (20, 9)), (16, (42, 31)),
    ),
    preparation_stop=53,
    mr_instructions=(74, *range(109, 2836, 47)),
    readout_instruction=2862,
)
