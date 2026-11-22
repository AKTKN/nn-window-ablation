"""Select physical detector observations for label-independent temporal queries."""

from bisect import bisect_left
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from .simulation import Arrival, ArrivalPhase, CircuitBundle


@dataclass(frozen=True, slots=True)
class TargetQuery:
    """Zero-based core index, or None for a global target, and inclusive round bounds."""

    core_index: int | None
    first_round: int
    last_round: int


@dataclass(frozen=True, slots=True)
class Buffers:
    left: int
    right: int


def buffer_lengths(distance: int) -> tuple[int, int, int]:
    if type(distance) is not int or distance < 3 or distance % 2 == 0:
        raise ValueError("distance must be an odd integer of at least 3")
    return (0, (distance + 1) // 2, distance)


class SlotState(StrEnum):
    OBSERVED = "observed"
    UNAVAILABLE = "unavailable"
    PADDING = "padding"


@dataclass(frozen=True, slots=True)
class DetectorSite:
    detector_index: int
    coordinates: tuple[float, ...]
    layer_index: int
    instruction_index: int
    availability_instruction: int
    arrival: Arrival


@dataclass(frozen=True, slots=True)
class ObservationLayout:
    distance: int
    rounds: int
    queries: tuple[TargetQuery, ...]
    detectors: tuple[DetectorSite, ...]
    layers: tuple[tuple[DetectorSite, ...], ...]

    @property
    def global_query(self) -> TargetQuery:
        return TargetQuery(None, 1, self.rounds)

    def core_query(self, index: int) -> TargetQuery:
        if type(index) is not int or not 0 <= index < len(self.queries):
            raise ValueError("core index is outside the target partition")
        return self.queries[index]


@dataclass(frozen=True, slots=True)
class TemporalSlot:
    """Physical layer index, or a nominal index outside the circuit for padding."""

    layer_index: int
    arrival: Arrival | None
    state: SlotState
    detector_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class QueryMetadata:
    target: TargetQuery
    buffers: Buffers | None
    first_observed_round: int
    last_observed_round: int
    release: Arrival
    preparation: bool
    destructive_readout: bool
    slots: tuple[TemporalSlot, ...]


@dataclass(frozen=True, slots=True)
class WindowView:
    metadata: QueryMetadata
    detector_count: int
    distance: int
    detectors: tuple[DetectorSite, ...]

    @property
    def detector_indices(self) -> tuple[int, ...]:
        return tuple(site.detector_index for site in self.detectors)


def build_observation_layout(bundle: CircuitBundle) -> ObservationLayout:
    """Extract public detector metadata from the P1 measurement and circuit schedule."""
    rounds = bundle.spec.rounds
    measurements = {measurement.record_index: measurement for measurement in bundle.measurements}
    layers: list[list[DetectorSite]] = [[] for _ in range(rounds + 1)]
    detectors = []
    for index, detector in enumerate(bundle.detectors):
        if detector.detector_index != index or not detector.measurement_indices:
            raise ValueError("expected contiguous detector columns with measurement dependencies")
        try:
            latest = max(
                (measurements[key] for key in detector.measurement_indices),
                key=lambda measurement: (measurement.arrival, measurement.instruction_index),
            )
        except KeyError as exc:
            raise ValueError("detector refers to a missing measurement record") from exc

        # The declaration's physical layer and its latest required measurement
        # are separate from the time coordinate used by upstream for full shots.
        if detector.instruction_index > bundle.destructive_readout_instruction:
            layer_index = rounds
        else:
            layer_index = bisect_left(bundle.round_measurement_instructions, detector.instruction_index) - 1
            if layer_index < 0:
                raise ValueError("detector is declared before the first ancilla measurement")
        site = DetectorSite(
            index, detector.coordinates, layer_index, detector.instruction_index,
            latest.instruction_index, latest.arrival,
        )
        detectors.append(site)
        layers[layer_index].append(site)

    queries = tuple(TargetQuery(core.core_index, core.first_round, core.last_round) for core in bundle.cores)
    return ObservationLayout(bundle.spec.distance, rounds, queries, tuple(detectors), tuple(map(tuple, layers)))


def select_window(layout: ObservationLayout, target: TargetQuery, buffers: Buffers | None = None) -> WindowView:
    """Plan a buffered core query, or a full-context core or global query."""
    if target.core_index is None:
        if target != layout.global_query or buffers is not None:
            raise ValueError("a global query must cover the entire circuit with full context")
    elif target != layout.core_query(target.core_index):
        raise ValueError("target bounds differ from the physical core partition")

    terminal = buffers is None or target.last_round == layout.rounds
    if buffers is None:
        first, last = 1, layout.rounds
    else:
        allowed = buffer_lengths(layout.distance)
        if any(type(value) is not int or value not in allowed for value in (buffers.left, buffers.right)):
            raise ValueError(f"buffer lengths must be in {allowed}")
        first, last = target.first_round - buffers.left, target.last_round + buffers.right
    lower, upper = max(1, first), min(layout.rounds, last)
    release = Arrival(upper, ArrivalPhase.DESTRUCTIVE_READOUT if terminal else ArrivalPhase.ANCILLA)
    last_layer = layout.rounds if terminal else upper - 1

    # Include at least one guard on each edge, plus any requested padding
    # beyond the circuit. The terminal layer occupies its own slot.
    left_guard = min(first - 1, lower - 2)
    right_guard = max(last - 1 + int(terminal), last_layer + 1)
    slots = []
    selected = []
    for layer_index in range(left_guard, right_guard + 1):
        if not 0 <= layer_index <= layout.rounds:
            slots.append(TemporalSlot(layer_index, None, SlotState.PADDING, ()))
            continue
        if layer_index == layout.rounds:
            arrival = Arrival(layout.rounds, ArrivalPhase.DESTRUCTIVE_READOUT)
        else:
            arrival = Arrival(layer_index + 1, ArrivalPhase.ANCILLA)
        sites = layout.layers[layer_index]
        in_context = lower - 1 <= layer_index <= last_layer and arrival <= release
        permitted = tuple(site for site in sites if site.arrival <= release) if in_context else ()
        observed = in_context and (bool(permitted) or not sites)
        slots.append(TemporalSlot(
            layer_index, arrival, SlotState.OBSERVED if observed else SlotState.UNAVAILABLE,
            tuple(site.detector_index for site in permitted),
        ))
        selected.extend(permitted)

    metadata = QueryMetadata(
        target, buffers, lower, upper, release,
        any(slot.layer_index == 0 and slot.state == SlotState.OBSERVED for slot in slots),
        any(slot.layer_index == layout.rounds and slot.state == SlotState.OBSERVED for slot in slots),
        tuple(slots),
    )
    selected.sort(key=lambda site: site.detector_index)
    return WindowView(metadata, len(layout.detectors), layout.distance, tuple(selected))


@dataclass(frozen=True, slots=True)
class PackedObservations:
    bits: NDArray[np.uint8]
    detector_count: int
    shot_ids: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class CroppedObservations:
    view: WindowView
    bits: NDArray[np.bool_]
    shot_ids: NDArray[np.int64]


def _check_shot_ids(shot_ids: NDArray[np.int64], count: int) -> None:
    if shot_ids.dtype != np.int64 or shot_ids.shape != (count,):
        raise ValueError("shot IDs must be an int64 vector aligned with detector rows")
    if np.unique(shot_ids).size != count:
        raise ValueError("physical detector rows must have distinct shot IDs")


def pack_observations(bits: NDArray[np.bool_], shot_ids: NDArray[np.int64]) -> PackedObservations:
    if bits.dtype != np.bool_ or bits.ndim != 2:
        raise ValueError("detector bits must be a two-dimensional boolean array")
    _check_shot_ids(shot_ids, len(bits))
    return PackedObservations(np.packbits(bits, axis=1, bitorder="little"), bits.shape[1], shot_ids.copy())


def unpack_observations(packed: PackedObservations) -> tuple[NDArray[np.bool_], NDArray[np.int64]]:
    if (
        type(packed.detector_count) is not int or packed.detector_count < 0
        or packed.bits.ndim != 2 or packed.bits.dtype != np.uint8
        or packed.bits.shape[1] != (packed.detector_count + 7) // 8
    ):
        raise ValueError("invalid packed detector dimensions")
    _check_shot_ids(packed.shot_ids, len(packed.bits))
    bits = np.unpackbits(packed.bits, axis=1, count=packed.detector_count, bitorder="little")
    return bits.astype(np.bool_), packed.shot_ids.copy()


def crop_observations(
    view: WindowView, bits: NDArray[np.bool_], shot_ids: NDArray[np.int64],
) -> CroppedObservations:
    """Copy only permitted detector columns, before any graph is built."""
    if bits.dtype != np.bool_ or bits.ndim != 2 or bits.shape[1] != view.detector_count:
        raise ValueError("detector bits must be boolean rows matching the physical layout")
    _check_shot_ids(shot_ids, len(bits))
    return CroppedObservations(view, bits[:, view.detector_indices], shot_ids.copy())
