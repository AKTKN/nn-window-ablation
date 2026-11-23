"""Shared continuation fixtures and a tensor inventory independent of the workflow."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from nn_window_ablation.config import resolve_config
from nn_window_ablation.execution import train_workflow
from nn_window_ablation.streams import TrajectoryStream
from nn_window_ablation.training import Trainer, UpdateBudget, fresh_decoder


ARMS = ("continued_full", "fixed_window", "dynamic_window")
TEST_BUDGET = (
    "data.training_chunk_shots=3", "training.batch_size=2", "training.microbatch_size=1",
    "training.optimizer=adam", "training.learning_rate=0.002", "training.scheduler=step",
    "training.scheduler_step_size=1", "training.scheduler_gamma=0.75",
)


def tensor_inventory(state):
    return tuple(
        (name, str(tensor.dtype), tuple(tensor.shape), tensor.detach().cpu().numpy().tobytes())
        for name, tensor in sorted(state.items())
    )


def reference_model_hash(state):
    parts = [
        json.dumps([name, dtype, list(shape)], separators=(",", ":")).encode() + b"\n" + content
        for name, dtype, shape, content in tensor_inventory(state)
    ]
    return sha256(b"".join(parts)).hexdigest()


def decoder_trainer(bundle, plan, *, steps=1, batch=2, micro=1, dynamic=False, fixture_buffer=False):
    model = fresh_decoder(plan)
    if fixture_buffer:
        model.register_buffer("fixture_normalization", torch.tensor([0.25, -0.75], dtype=torch.float64))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.75)
    return Trainer(
        model, optimizer, scheduler, TrajectoryStream(plan, bundle), UpdateBudget(steps, batch, micro),
        architecture=model.architecture(), dynamic_choices=(0, (plan.spec.distance + 1) // 2, plan.spec.distance) if dynamic else None,
    )


@pytest.fixture
def continuation_pilot(tmp_path, compose_settings):
    def resolve(*overrides):
        return resolve_config(compose_settings("runtime=test_cpu", *overrides), invocation_dir=tmp_path)

    parent = train_workflow(resolve(
        "run_name=base", "data.split=train", "training.max_steps=3", *TEST_BUDGET,
    ))
    return resolve, parent


def run_nnwa(root: Path, command: str, *overrides: str):
    result = subprocess.run(
        [str(Path(sys.executable).with_name("nnwa")), command, *overrides],
        cwd=root, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout
