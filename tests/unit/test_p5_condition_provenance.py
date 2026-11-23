from dataclasses import asdict
from pathlib import Path

import pytest

from nn_window_ablation.artifacts import ArtifactError, write_json
from nn_window_ablation.config import ConfigError, resolve_config
from nn_window_ablation.data import open_dataset, write_dataset
from nn_window_ablation.evaluation import CheckpointProvenance
from nn_window_ablation.paired_workflows import _check_comparisons, _dataset_record, _read_manifest, check_paired_execution
from tests.p4_checks import setup_plan


@pytest.fixture
def make_provenance():
    _, test = setup_plan(split="test", total=3)

    def make(name, phase, initialization, context):
        _, train = setup_plan(phase=phase)
        parent = 4 if phase == "continuation" else 0
        payload = {
            "contract": {
                "source": {"plan": train.record()},
                "budget": {"max_steps": 1, "batch_size": 2, "microbatch_size": 1},
            },
            "run": {
                "resolved_config": {"training": {
                    "name": name, "initialization": initialization, "core_context": context,
                    "global_context": "full_closed", "data_stream_phase": phase,
                }},
                "parent": {"sha256": "b" * 64} if parent else None,
            },
            "counters": {
                "parent_updates": parent, "phase_updates": 1, "cumulative_updates": parent + 1,
                "trajectories": 2, "query_examples": 4,
            },
        }
        return payload, test

    return make


@pytest.mark.parametrize(("name", "phase", "initialization", "context", "expected"), [
    ("full_closed", "base", "random", "full_closed", "S"),
    ("continued_full", "continuation", "full_checkpoint", "full_closed", "S+"),
    ("fixed_window", "continuation", "full_checkpoint", "deployment_window", "W"),
    ("dynamic_window", "continuation", "full_checkpoint", "sampled_window", "D"),
])
def test_p5_checkpoint_training_regime_controls_condition(
    make_provenance, name, phase, initialization, context, expected,
):
    payload, test = make_provenance(name, phase, initialization, context)
    checkpoint = CheckpointProvenance.from_payload(payload, "a" * 64, test)
    assert checkpoint.condition("S") == expected
    assert checkpoint.condition("G") == ("G" if phase == "base" else f"G@{name}")
    assert checkpoint.condition("F") == ("F" if phase == "base" else f"F@{name}")
    assert checkpoint.training_seed == 87 and checkpoint.training_replicate == 2
    assert checkpoint.sha256 == "a" * 64


def test_p5_rejects_checkpoint_schema_circuit_or_training_provenance_mismatch(make_provenance):
    payload, test = make_provenance("full_closed", "base", "random", "full_closed")
    for field in ("label_schema", "feature_schema"):
        payload["contract"]["source"]["plan"][field] = "other-schema"
        with pytest.raises(ArtifactError, match="label or feature schemas differ"):
            CheckpointProvenance.from_payload(payload, "a" * 64, test)
        payload["contract"]["source"]["plan"][field] = test.record()[field]
    _, other = setup_plan(split="test", total=3, p=0)
    with pytest.raises(ArtifactError, match="physical circuits differ"):
        CheckpointProvenance.from_payload(payload, "a" * 64, other)
    payload["run"]["resolved_config"]["training"]["name"] = "dynamic_window"
    with pytest.raises(ArtifactError, match="regime contradicts"):
        CheckpointProvenance.from_payload(payload, "a" * 64, test)


def test_p5_rejects_mixed_checkpoint_and_window_conditions(make_provenance):
    payload, test = make_provenance("full_closed", "base", "random", "full_closed")
    provenance = asdict(CheckpointProvenance.from_payload(payload, "a" * 64, test))
    entry = {"checkpoint": provenance, "model_state_sha256": "e" * 64, "mode": "G", "queries": []}
    different = {"checkpoint": {**provenance, "sha256": "c" * 64}}
    with pytest.raises(ArtifactError, match="same checkpoint and model state"):
        _check_comparisons({"G": entry, "F": {**entry, **different, "mode": "F"}}, ["F_minus_G"])
    with pytest.raises(ArtifactError, match="one training seed and replicate"):
        _check_comparisons({
            "G": entry, "F": {**entry, "checkpoint": {**provenance, "training_seed": 2}},
        }, ["F_minus_G"])
    window = {**entry, "mode": "S", "checkpoint": {**provenance, "parent_sha256": "b" * 64}}
    with pytest.raises(ArtifactError, match="identical deployment queries and buffers"):
        _check_comparisons({
            "W": {**window, "queries": [{"buffers": {"left": 5, "right": 5}}]},
            "D": {**window, "queries": [{"buffers": {"left": 0, "right": 5}}]},
        }, ["W_minus_D"])


def test_p5_rejects_prediction_manifest_on_other_dataset_or_schema(tmp_path):
    bundle, plan = setup_plan(split="test", total=3)
    root = tmp_path / "data"
    write_dataset(root, plan, bundle, provenance={})
    mapped = open_dataset(root)
    path = tmp_path / "prediction.json"
    expected = _dataset_record(mapped)
    for key in ("dataset_id", "manifest_sha256", "label_schema", "feature_schema"):
        write_json(path, {"schema_version": 1, "status": "complete", "data": {**expected, key: "unrelated"}})
        with pytest.raises(ArtifactError, match="dataset identity or label/feature schema differs"):
            _read_manifest(path, mapped)


@pytest.mark.parametrize(("command", "overrides", "error"), [
    ("evaluate", ("evaluation.threshold=0.4",), "threshold"),
    ("evaluate", ("evaluation.update_model_state=true",), "frozen state"),
    ("evaluate", ("data.split=train",), "held-out split"),
    ("evaluate", ("code.rounds_per_distance=3",), "12-core"),
    ("analyze", ("analysis.bootstrap_seed=1", "analysis.sparse_rule=0"), "analysis.bootstrap_replicates"),
    ("analyze", ("analysis.bootstrap_seed=1", "analysis.bootstrap_replicates=5"), "analysis.sparse_rule"),
    ("analyze", ("analysis.sparse_rule=0", "analysis.bootstrap_replicates=5"), "analysis.bootstrap_seed"),
    ("analyze", ("analysis.sparse_rule=0", "analysis.bootstrap_replicates=5", "analysis.bootstrap_seed=1",
                 "analysis.pool_training_seeds_as_independent_shots=true"), "separate training seeds"),
])
def test_p5_rejects_invalid_execution_configuration(tmp_path, compose_settings, command, overrides, error):
    settings = compose_settings(
        "runtime=test_cpu", "data.split=test", "data.id=frozen",
        "checkpoint.path=checkpoint.pt", "analysis.predictions=[predictions/manifest.json]", *overrides,
    )
    config = resolve_config(settings, invocation_dir=tmp_path)
    with pytest.raises(ConfigError, match=error):
        check_paired_execution(command, config)
    assert list(Path(tmp_path).iterdir()) == []
