from dataclasses import astuple

import pytest

from nn_window_ablation.config import ConfigError, require_execution_config, resolve_config


def test_pilot_geometry_and_noise(compose_settings, tmp_path):
    config = resolve_config(compose_settings(), invocation_dir=tmp_path)

    assert astuple(config.geometry) == (5, 60, 5, 5, 5, 15, 12)
    assert config.noise.circuit_probabilities == {
        "after_clifford_depolarization": 0.003,
        "before_round_data_depolarization": 0.003,
        "before_measure_flip_probability": 0.003,
        "after_reset_flip_probability": 0.003,
    }
    assert config.model.hidden_size == 256
    assert config.model.gru_layers == 4
    assert config.training.initialization == "random"


@pytest.mark.parametrize(
    ("experiment", "expected"),
    [("pilot_d5", (5, 60, 5, 3, 3, 11, 12)), ("paired_d7", (7, 84, 7, 4, 4, 15, 12))],
)
def test_half_distance_buffers_round_up(compose_settings, tmp_path, experiment, expected):
    config = resolve_config(
        compose_settings(
            f"experiment={experiment}", "window.left_per_distance=0.5", "window.right_per_distance=0.5"
        ),
        invocation_dir=tmp_path,
    )
    assert astuple(config.geometry) == expected
    assert config.to_record()["window"]["left_per_distance"] == 0.5
    assert config.to_record()["geometry"]["right_buffer"] == expected[4]


def test_group_experiment_and_cli_precedence(compose_settings, tmp_path):
    experiment = resolve_config(
        compose_settings("experiment=paired_d7", "data=paired", "evaluation=F", "runtime=local_cpu"),
        invocation_dir=tmp_path,
    )
    assert (experiment.study_stage, experiment.code.distance, experiment.data.family_id) == ("main", 7, "paired_d7")
    assert (experiment.evaluation.id, experiment.evaluation.context, experiment.evaluation.aggregation) == (
        "F", "full_closed", "xor",
    )
    explicit = resolve_config(
        compose_settings("experiment=paired_d7", "data.family_id=explicit", "code.distance=9", "noise.probability=0"),
        invocation_dir=tmp_path,
    )
    assert (explicit.code.distance, explicit.data.family_id) == (9, "explicit")
    assert list(explicit.noise.circuit_probabilities.values()) == [0.0] * 4


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        (("code.distance=2",), "code.distance must be an odd integer"),
        (("code.distance=6",), "code.distance must be an odd integer"),
        (("code.distance=5.0",), "code.distance must be an integer"),
        (("code.distance=true",), "code.distance must be an integer"),
        (("code.rounds_per_distance=0",), "code.rounds_per_distance must be positive"),
        (("code.rounds_per_distance=1.5",), "code.rounds_per_distance must be an integer"),
        (("window.core_per_distance=0.5",), "whole cores that partition"),
        (("code.rounds_per_distance=11", "window.core_per_distance=2"), "whole cores that partition"),
        (("window.core_per_distance=2",), "core length equal to code.distance"),
        (("window.left_per_distance=0.25",), "window.left_per_distance must be"),
        (("window.right_per_distance=-1",), "window.right_per_distance must be"),
        (("noise.probability=-0.003",), "noise.probability in"),
        (("noise.probability=1.001",), "noise.probability in"),
        (("noise.probability=nan",), "noise.probability in"),
        (("noise.after_reset_flip_probability=0.005",), "probabilities must equal"),
        (("model.name=unavailable",), "model.name must be"),
        (("model.hidden_size=32",), "retain the specified Lange backbone"),
        (("runtime.device=cuda",), "runtime must use device=cpu"),
        (("training.max_steps=-1",), "training.max_steps must be positive"),
        (("analysis.bootstrap_replicates=0",), "analysis.bootstrap_replicates must be positive"),
        (("checkpoint.init_from=base.pt", "checkpoint.resume_from=resume.pt"), "mutually exclusive"),
    ],
)
def test_invalid_settings_raise_specific_error(compose_settings, tmp_path, overrides, error):
    with pytest.raises(ConfigError, match=error):
        resolve_config(compose_settings(*overrides), invocation_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_required_group_must_be_a_mapping(compose_settings, tmp_path):
    config = compose_settings()
    config.runtime = None
    with pytest.raises(ConfigError, match="runtime must be a mapping"):
        resolve_config(config, invocation_dir=tmp_path)


@pytest.mark.parametrize(
    ("command", "missing"),
    [
        ("analyze", "analysis.bootstrap_replicates"),
        ("evaluate", "checkpoint.path"),
        ("generate", "data.shots"),
        ("train", "training.max_steps"),
    ],
)
def test_unresolved_budgets_allow_inspection_only(compose_settings, tmp_path, command, missing):
    config = resolve_config(compose_settings(), invocation_dir=tmp_path)
    assert config.training.max_steps is None
    assert config.analysis.bootstrap_replicates is None

    with pytest.raises(ConfigError, match=rf"{command} requires explicit values.*{missing}"):
        require_execution_config(config, command)


def test_execution_guard_accepts_explicit_training_budget(compose_settings, tmp_path):
    config = resolve_config(
        compose_settings(
            "runtime=test_cpu", "training.max_steps=10", "training.batch_size=2", "training.learning_rate=0.001",
            "training.optimizer=adam", "training.scheduler=none", "data.training_chunk_shots=4",
        ),
        invocation_dir=tmp_path,
    )
    require_execution_config(config, "train")
