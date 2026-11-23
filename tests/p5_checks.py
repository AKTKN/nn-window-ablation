"""Genuine P4 artifacts shared by the P5 integration checks."""

import pytest

from nn_window_ablation.config import resolve_config
from nn_window_ablation.data import open_dataset
from nn_window_ablation.execution import generate_workflow, train_workflow


@pytest.fixture
def frozen_pilot(tmp_path, compose_settings):
    def resolve(*overrides):
        return resolve_config(compose_settings("runtime=test_cpu", *overrides), invocation_dir=tmp_path)

    dataset = generate_workflow(resolve("data.id=held", "data.split=test", "data.shots=3", "data.shard_shots=2"))
    mapped = open_dataset(dataset.parent, expected_split="test")
    checkpoint = train_workflow(resolve(
        "run_name=base", "data.split=train", "training.max_steps=1", "training.batch_size=1",
        "training.optimizer=sgd", "training.scheduler=none", "training.learning_rate=0.001",
        "data.training_chunk_shots=2",
    ))
    assert mapped.plan.spec.distance == 5 and mapped.plan.spec.rounds == 60 and mapped.plan.spec.probability == 0.003
    return resolve, mapped, dataset.parent, checkpoint
