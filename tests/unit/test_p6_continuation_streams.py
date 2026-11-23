from dataclasses import replace

import numpy as np
import pytest

from nn_window_ablation.artifacts import ArtifactError
from nn_window_ablation.data import TrajectoryID
from nn_window_ablation.streams import ChunkCursor
from tests.p4_checks import convex_loss, reference_seed, scalar_trainer, setup_plan
from tests.p6_checks import ARMS


def test_p6_stream_matches_independent_chunk_and_core_order_across_arms():
    bundle, base = setup_plan(rounds=36, chunk=3)
    parent = scalar_trainer(bundle, base, steps=2, batch=3)
    base_ids = {identity for _ in range(2) for identity in parent.step(convex_loss).trajectories}
    plan = replace(base, namespace=replace(base.namespace, phase="continuation"))
    expected_ids = [
        TrajectoryID(plan.dataset_id, chunk, int(row))
        for chunk in range(3)
        for row in np.random.default_rng(reference_seed(plan.namespace, "trajectory_order", chunk)).permutation(3)
    ][:8]
    expected_cores = np.random.default_rng(reference_seed(plan.namespace, "core_query")).integers(0, 12, size=8)
    traces = {}
    for name in ARMS:
        child = scalar_trainer(bundle, plan, steps=4, batch=2, dynamic=name == "dynamic_window")
        child.fork_from(parent.snapshot())
        if name == "dynamic_window":
            child.randomness.buffer.integers(0, 3, size=83)
        updates = [child.step(convex_loss) for _ in range(4)]
        identities = [identity for row in updates for identity in row.trajectories]
        cores = [core for row in updates for core in row.cores]
        assert identities == expected_ids and len(set(identities)) == 8
        np.testing.assert_array_equal(cores, expected_cores)
        assert child.stream.cursor == ChunkCursor(2, 2)
        assert not base_ids.intersection(identities)
        traces[name] = (identities, cores)
    assert traces["continued_full"] == traces["fixed_window"] == traces["dynamic_window"]

    wrong = scalar_trainer(bundle, base, steps=4, batch=2)
    with pytest.raises(ArtifactError, match="distinct continuation namespace"):
        wrong.fork_from(parent.snapshot())
    arm_namespace = replace(plan, namespace=replace(plan.namespace, family=plan.namespace.family + "-D"))
    arm_derived = scalar_trainer(bundle, arm_namespace, steps=4, batch=2)
    with pytest.raises(ArtifactError, match="matching base replicate"):
        arm_derived.fork_from(parent.snapshot())
    assert reference_seed(arm_namespace.namespace, "core_query") != reference_seed(plan.namespace, "core_query")

    for cursor in (ChunkCursor(0, 0), ChunkCursor(2, 0)):
        bad = scalar_trainer(bundle, plan, steps=4, batch=2)
        bad.fork_from(parent.snapshot())
        first = bad.step(convex_loss)
        bad.stream.seek(cursor)  # Replay a consumed chunk or skip the next chunk.
        second = bad.step(convex_loss)
        with pytest.raises(AssertionError):
            assert list(first.trajectories + second.trajectories) == expected_ids[:4]
