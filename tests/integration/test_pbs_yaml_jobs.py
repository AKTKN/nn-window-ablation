from pathlib import Path

import pytest

from nn_window_ablation.config import resolve_config
from nn_window_ablation.workflows import check_workflow
from scripts.pbs.run_from_yaml import command_line


CONFIGS = Path(__file__).resolve().parents[2] / "scripts/pbs/config"


def test_pbs_jobs_share_data_checkpoint_and_prediction_inputs(compose_settings, tmp_path):
    jobs = {}
    for name, command in [
        ("sample_train", "generate"), ("sample_test", "generate"), ("train", "train"),
        ("evaluate", "evaluate"), ("analyze", "analyze"),
    ]:
        executable, subcommand, *arguments = command_line(command, CONFIGS / f"{name}.yaml")
        assert (executable, subcommand) == ("nnwa", command)
        config = resolve_config(compose_settings(*arguments), invocation_dir=tmp_path)
        check_workflow(command, config)
        jobs[name] = config

    train_data, test_data = jobs["sample_train"], jobs["sample_test"]
    train, evaluate, analyze = jobs["train"], jobs["evaluate"], jobs["analyze"]
    assert train_data.data.split == train.data.split == "train"
    assert test_data.data.split == evaluate.data.split == analyze.data.split == "test"
    assert train_data.data.id == train.data.id != test_data.data.id
    assert test_data.data.id == evaluate.data.id == analyze.data.id
    assert len({cfg.paths.data_root for cfg in jobs.values()}) == 1
    assert train_data.data.shots == train.training.batch_size * train.training.max_steps == 10_000_128
    assert test_data.data.shots == 10_000_000
    assert train.runtime.processes == evaluate.evaluation.workers == 96
    assert evaluate.checkpoint.path == train.paths.run_root / train.run_name / "checkpoint.pt"
    assert analyze.analysis.predictions == [str(evaluate.paths.run_root / evaluate.run_name / "manifest.json")]


def test_wrong_stage_yaml_fails_before_starting_nnwa():
    with pytest.raises(ValueError, match="expected command: train"):
        command_line("train", CONFIGS / "evaluate.yaml")
