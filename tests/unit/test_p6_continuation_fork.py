from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from nn_window_ablation.checkpoints import load_checkpoint, save_checkpoint
from nn_window_ablation.training import MemoryObjective
from nn_window_ablation.windows import Buffers
from tests.p4_checks import setup_plan
from tests.p6_checks import ARMS, decoder_trainer, reference_model_hash, tensor_inventory


def test_p6_forks_copy_every_named_tensor_and_reset_optimizer_and_scheduler(tmp_path):
    bundle, plan = setup_plan(distance=3, rounds=36)
    parent = decoder_trainer(bundle, plan, steps=2, fixture_buffer=True)
    objective = MemoryObjective(bundle, core_context="full_closed", fixed_buffers=Buffers(3, 3), core_weight=1)
    for _ in range(2):
        parent.step(objective)
    assert parent.scheduler.last_epoch == 2
    assert all(int(state["step"]) == 2 for state in parent.optimizer.state.values())
    assert any(torch.count_nonzero(state["exp_avg"]).item() for state in parent.optimizer.state.values())
    parent_path = tmp_path / "base.pt"
    save_checkpoint(parent_path, parent.snapshot())
    payload = load_checkpoint(parent_path)
    parent_state = payload["model"]["state_dict"]
    expected_inventory = tensor_inventory(parent_state)
    initial_hash = reference_model_hash(parent_state)
    assert {"gnn.empty", "query_projection.weight", "head.weight", "fixture_normalization"} <= parent_state.keys()

    continued = replace(plan, namespace=replace(plan.namespace, phase="continuation"))
    arms = {
        name: decoder_trainer(bundle, continued, fixture_buffer=True, dynamic=name == "dynamic_window")
        for name in ARMS
    }
    for child in arms.values():
        child.fork_from(payload)
        assert tensor_inventory(child.model.state_dict()) == expected_inventory
        assert reference_model_hash(child.model.state_dict()) == initial_hash
        assert child.optimizer.state_dict()["state"] == {}
        assert child.scheduler.last_epoch == 0 and child.scheduler.get_last_lr() == [0.002]
        assert child.optimizer.param_groups[0]["lr"] == 0.002

    with torch.no_grad():
        arms["continued_full"].model.head.weight.fill_(0)
        arms["continued_full"].model.fixture_normalization.add_(1)
    assert reference_model_hash(arms["continued_full"].model.state_dict()) != initial_hash
    assert tensor_inventory(arms["fixed_window"].model.state_dict()) == expected_inventory
    assert tensor_inventory(arms["dynamic_window"].model.state_dict()) == expected_inventory
    assert tensor_inventory(parent.model.state_dict()) == tensor_inventory(parent_state) == expected_inventory
    # The "fresh head in one arm" mutation fails the exact initial inventory.
    with pytest.raises(AssertionError):
        assert tensor_inventory(arms["continued_full"].model.state_dict()) == expected_inventory

    carried = decoder_trainer(bundle, continued, fixture_buffer=True)
    carried.optimizer.load_state_dict(deepcopy(payload["optimizer_state"]))
    with pytest.raises(ValueError, match="fresh trainer, optimizer, and scheduler"):
        carried.fork_from(payload)
    advanced = decoder_trainer(bundle, continued, fixture_buffer=True)
    advanced.scheduler.load_state_dict(payload["scheduler_state"])
    with pytest.raises(ValueError, match="fresh trainer, optimizer, and scheduler"):
        advanced.fork_from(payload)
