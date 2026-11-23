from collections import Counter
from dataclasses import replace
from itertools import product

import numpy as np

from nn_window_ablation.data import sample_chunk
from nn_window_ablation.training import MemoryObjective, sample_buffer_pair
from nn_window_ablation.windows import Buffers, buffer_lengths
from tests.p4_checks import convex_loss, scalar_trainer, setup_plan
from tests.reference.p2_observations import PILOT_SCHEDULE


class ControlledDraw:
    def __init__(self, positions):
        self.positions = positions
        self.calls = []

    def integers(self, high, *, size):
        self.calls.append((high, size))
        return np.array(self.positions, dtype=np.int64)


def test_p6_all_nine_dynamic_pairs_map_uniformly_and_preserve_targets():
    bundle, plan = setup_plan(p=0, distance=5, rounds=60, chunk=2)
    original = sample_chunk(plan, bundle, 0)
    assert not original.detectors.any()
    events = original.detectors.copy()
    events[1, [0, 999, 1300, 1415, 1429]] = True
    shots = replace(original, detectors=events)
    objective = MemoryObjective(bundle, core_context="sampled_window", fixed_buffers=Buffers(5, 5), core_weight=1)
    cores = np.array([10, 11], dtype=np.int64)
    choices = buffer_lengths(5)
    assert choices == (0, 3, 5)
    counts = Counter()
    for positions in product(range(3), repeat=2):
        rng = ControlledDraw(positions)
        pair = sample_buffer_pair(choices, rng)
        assert rng.calls == [(3, 2)]
        lengths = tuple((0, 3, 5)[index] for index in positions)
        assert (pair.left, pair.right) == lengths
        counts[lengths] += 1
        batch = objective.prepare(shots, cores, pair)
        assert [example.query.target.core_index for example in batch.examples] == [None, None, 10, 11]
        assert [example.shot_id for example in batch.examples] == [*shots.shot_ids, *shots.shot_ids]
        for row, example in enumerate(batch.examples):
            target = example.query.target.core_index
            expected = PILOT_SCHEDULE.select(target, lengths if target is not None else None)
            reported = sorted(index for slot in example.query.slots for index in slot.detector_indices)
            assert reported == list(expected)
            assert (example.query.target.first_round, example.query.target.last_round) == (
                (1, 60) if target is None else (target * 5 + 1, target * 5 + 5)
            )
        assert not batch.examples[2].query.destructive_readout
        assert batch.examples[3].query.destructive_readout
    assert counts == Counter({pair: 1 for pair in product((0, 3, 5), repeat=2)})


def test_p6_one_buffer_pair_per_logical_batch_even_with_gradient_accumulation():
    bundle, plan = setup_plan(distance=5, rounds=60, chunk=4)
    parent = scalar_trainer(bundle, plan, steps=1, batch=3)
    parent.step(convex_loss)
    plan = replace(plan, namespace=replace(plan.namespace, phase="continuation"))
    child = scalar_trainer(bundle, plan, steps=2, batch=3, micro=2)
    child.dynamic_choices = (0, 3, 5)
    child.fork_from(parent.snapshot())
    draw = ControlledDraw((1, 2))
    child.randomness.buffer = draw
    observed = []

    def objective(model, shots, indices, buffers):
        observed.append((len(shots.shot_ids), buffers))
        return convex_loss(model, shots, indices, buffers)

    for _ in range(2):
        trace = child.step(objective)
        assert trace.buffers == Buffers(3, 5)
    assert draw.calls == [(3, 2), (3, 2)]
    assert observed == [(2, Buffers(3, 5)), (1, Buffers(3, 5))] * 2
