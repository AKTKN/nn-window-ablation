"""Reference observation schedules independent of P2 masks and graph builders."""

from dataclasses import dataclass
from itertools import pairwise

from .p1_schedule import PILOT


@dataclass(frozen=True)
class Schedule:
    rounds: int
    distance: int
    # Arrival times use 2*r for the ancilla round and 2*T+1 for readout.
    measurement_times: tuple[int, ...]
    detector_dependencies: tuple[tuple[int, ...], ...]
    detector_layers: tuple[int, ...]

    def select(
        self, core: int | None = None, buffers: tuple[int, int] | None = None,
        *, ignore_readout_phase: bool = False,
    ) -> tuple[int, ...]:
        first, end = 1, self.rounds
        final = True
        if buffers is not None:
            first = max(1, 1 + core * self.distance - buffers[0])
            end = min(self.rounds, (core + 1) * self.distance + buffers[1])
            final = (core + 1) * self.distance == self.rounds
        phase = 2 * end + final
        permitted = []
        for index, (layer, dependencies) in enumerate(zip(self.detector_layers, self.detector_dependencies)):
            latest = max(self.measurement_times[record] for record in dependencies)
            if ignore_readout_phase:
                latest -= latest % 2
            # The readout layer belongs to physical round T for this test.
            if first <= min(layer + 1, self.rounds) <= end and latest <= phase:
                permitted.append(index)
        return tuple(permitted)

    def pairs(self, allowed: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
        layers = sorted({self.detector_layers[index] for index in allowed})
        return tuple((left, right) for left, right in pairwise(layers) if right - left == 1)


PILOT_SCHEDULE = Schedule(
    60, 5, tuple(2 * round_number for round_number in range(1, 61) for _ in range(24)) + (121,) * 25,
    PILOT.detector_records(),
    (0,) * 12 + tuple(layer for layer in range(1, 60) for _ in range(24)) + (60,) * 12,
)

# The last two detectors have the same geometric time but different physical
# phases. Detector 3 depends in part on a measurement outside the zero-buffer
# middle window, so the oracle still selects that already-computed bit.
TOY_SCHEDULE = Schedule(
    9, 3, (2, 4, 6, 8, 10, 12, 14, 16, 18, 19),
    ((0,), (0, 1), (2,), (2, 3), (4,), (5,), (4, 5), (6,), (7,), (8,), (8, 9)),
    (0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9),
)
TOY_COORDS = (
    (6, 5, 0), (5, 5, 1), (9, 5, 2), (5, 5, 3), (6, 5, 4), (8, 5, 5),
    (6, 5, 5), (9, 5, 6), (5, 5, 7), (7, 5, 8), (9, 5, 8),
)
TOY_DECLARATIONS = (11, 21, 31, 41, 51, 61, 62, 71, 81, 91, 101)

# Local stabilizer positions and physical origins of the four rotated d=3
# tiles, in y-then-x order for the distance-5 patch.
D3_STABILIZERS = {(2, 0), (2, 2), (4, 2), (6, 2), (0, 4), (2, 4), (4, 4), (4, 6)}
D5_TILE_ORIGINS = ((0, 0), (4, 0), (0, 4), (4, 4))


def expected_graph(schedule, allowed, event_ids, coordinates, pair, origin, *, tiled=False):
    """Calculate a small directed graph using explicit event membership."""
    nodes = []
    detector_ids = []
    for index in allowed:
        layer = schedule.detector_layers[index]
        if index in event_ids and layer in pair:
            x, y, _ = coordinates[index]
            local = (x - origin[0], y - origin[1])
            if not tiled or local in D3_STABILIZERS:
                detector_ids.append(index)
                nodes.append((*local, layer - pair[0]))
    edges = []
    weights = []
    for i, first in enumerate(nodes):
        for j, second in enumerate(nodes):
            if i != j:
                edges.append((i, j))
                delta = max(abs(a - b) for a, b in zip(first, second))
                weights.append(1 / delta**2 if delta else 0)
    return detector_ids, nodes, edges, weights
