from dataclasses import replace

import pytest

from nn_window_ablation.artifacts import ArtifactError
from nn_window_ablation.checkpoints import load_checkpoint, save_checkpoint
from nn_window_ablation.config import ConfigError, resolve_config
from nn_window_ablation.execution import check_execution, check_matched_sweep
from nn_window_ablation.training import final_update_count
from tests.p4_checks import assert_exact, convex_loss, scalar_trainer, setup_plan
from tests.p6_checks import ARMS, TEST_BUDGET


def test_p6_new_update_budget_remains_distinct_from_parent_and_resume(tmp_path):
    bundle, base = setup_plan(rounds=36)
    parent = scalar_trainer(bundle, base, steps=7, batch=3, adam=True)
    for _ in range(7):
        parent.step(convex_loss)
    plan = replace(base, namespace=replace(base.namespace, phase="continuation"))
    child = scalar_trainer(bundle, plan, steps=3, batch=3, adam=True)
    child.fork_from(parent.snapshot())
    assert child.counters["cumulative_updates"] == 7 and not child.complete
    child.step(convex_loss)
    artifact = tmp_path / "ongoing.pt"
    save_checkpoint(artifact, child.snapshot())
    state = load_checkpoint(artifact)
    with pytest.raises(ArtifactError, match="fixed terminal update"):
        final_update_count(state)
    with pytest.raises(ArtifactError, match="training contract"):
        scalar_trainer(bundle, plan, steps=4, batch=3, adam=True).restore(state)

    resumed = scalar_trainer(bundle, plan, steps=3, batch=3, adam=True)
    resumed.restore(state)
    assert_exact(resumed.optimizer.state_dict(), child.optimizer.state_dict())
    assert resumed.scheduler.last_epoch == 1
    for step in (2, 3):
        assert resumed.step(convex_loss).update == step
    assert resumed.complete and final_update_count(resumed.snapshot()) == 10
    assert resumed.counters == {
        "parent_updates": 7, "phase_updates": 3, "cumulative_updates": 10, "trajectories": 9, "query_examples": 18,
    }
    assert all(int(value["step"]) == 3 for value in resumed.optimizer.state.values())
    with pytest.raises(ValueError, match="declared update budget"):
        resumed.step(convex_loss)


def test_p6_rejects_unmatched_sweeps_and_incompatible_configuration(tmp_path, compose_settings):
    def compose(arm, *overrides):
        return compose_settings(
            "runtime=test_cpu", "data.split=train", f"training={arm}", "training.max_steps=2",
            "checkpoint.init_from=base.pt", *TEST_BUDGET, *overrides,
        )

    def resolve(arm, *overrides):
        return resolve_config(compose(arm, *overrides), invocation_dir=tmp_path)

    arms = [resolve(arm) for arm in ARMS]
    for arm in arms:
        check_execution("train", arm)
    check_matched_sweep(arms)
    mismatches = (
        "training.learning_rate=0.003", "training.max_steps=4", "training.batch_size=4",
        "data.training_chunk_shots=5", "training.validation_data_id=different-val",
        "checkpoint.init_from=other.pt", "window.right_per_distance=0.5",
    )
    for change in mismatches:
        with pytest.raises(ConfigError, match=r"S\+/W/D sweep arms must share"):
            check_matched_sweep([*arms[:2], resolve("dynamic_window", change)])

    with pytest.raises(ConfigError, match="mutually exclusive"):
        resolve("dynamic_window", "checkpoint.resume_from=checkpoint.pt")
    with pytest.raises(ConfigError, match="complete 12-core"):
        check_execution("train", resolve("dynamic_window", "code.rounds_per_distance=3"))
    with pytest.raises(ConfigError, match="training regime must match"):
        check_execution("train", resolve("dynamic_window", "training.name=continued_full"))
    with pytest.raises(ConfigError, match="fixed-budget global"):
        check_execution("train", resolve("dynamic_window", "training.global_context=deployment_window"))
    assert not list(tmp_path.iterdir())
