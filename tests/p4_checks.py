"""Small independent assertions and fixtures for P4."""

from dataclasses import asdict
from hashlib import sha256
import json

import numpy as np
import torch
from torch import nn

from nn_window_ablation.data import SamplingPlan, StreamNamespace
from nn_window_ablation.simulation import CircuitSpec, build_circuit_bundle
from nn_window_ablation.streams import TrajectoryStream
from nn_window_ablation.training import Trainer, UpdateBudget


def setup_plan(*, split="train", phase=None, chunk=3, total=None, distance=3, rounds=6, p=0.003):
    bundle = build_circuit_bundle(CircuitSpec(distance, rounds, distance, p))
    namespace = StreamNamespace(
        master_seed=87, family="p4-fixture", split=split, distance=distance, replicate=2,
        phase=phase or ("base" if split == "train" else "held_out"),
    )
    return bundle, SamplingPlan.from_circuit(bundle, namespace, chunk_rows=chunk, total_rows=total)


def reference_seed(namespace, role, index=0):
    # Independent byte layout for seed recipe version 1.
    payload = json.dumps([1, asdict(namespace), role, index], sort_keys=True, separators=(",", ":"))
    return int(sha256(payload.encode()).digest()[:8][::-1].hex(), 16)


def assert_exact(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_exact(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for first, second in zip(actual, expected, strict=True):
            assert_exact(first, second)
    else:
        assert actual == expected


class Scalar(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(-2.0, dtype=torch.float64))
        self.register_buffer("scale", torch.tensor(1.0, dtype=torch.float64))

    def forward(self, count):
        return self.theta.expand(count) * self.scale


def scalar_trainer(bundle, plan, *, steps, batch=3, micro=2, adam=False, dynamic=False):
    model = Scalar()
    optimizer_class = torch.optim.Adam if adam else torch.optim.SGD
    optimizer = optimizer_class(model.parameters(), lr=0.125, foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.75) if adam else None
    return Trainer(
        model, optimizer, scheduler, TrajectoryStream(plan, bundle), UpdateBudget(steps, batch, micro),
        architecture={"fixture": "scalar"}, dynamic_choices=(0, 2, 3) if dynamic else None,
    )


def convex_loss(model, shots, _cores, _buffers):
    # Numeric join keys retain the physical row ordinal in their low 32 bits.
    physical_rows = torch.from_numpy(shots.shot_ids & 0xFFFFFFFF)
    targets = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)[physical_rows % 3]
    return ((model(len(targets)) - targets) ** 2).mean() / 2
