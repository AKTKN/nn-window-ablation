from dataclasses import replace
import multiprocessing
import os
import subprocess
import sys
from textwrap import dedent

import pytest
import torch

from nn_window_ablation.config import ConfigError, resolve_config, validate_cpu_budget
from nn_window_ablation.runtime import configure_cpu_runtime


def test_explicit_runtime_and_worker_counts_are_accounted_for(compose_settings, tmp_path):
    config = resolve_config(compose_settings(
        "runtime=test_cpu", "runtime.processes=2", "runtime.intraop_threads=2", "runtime.interop_threads=2",
        "runtime.allocated_cpus=14", "data.num_workers=1", "data.sampling_processes=2",
    ), invocation_dir=tmp_path)

    assert (config.runtime.processes, config.runtime.intraop_threads, config.runtime.interop_threads) == (2, 2, 2)
    assert (config.data.num_workers, config.data.sampling_processes, config.runtime.allocated_cpus) == (1, 2, 14)
    assert validate_cpu_budget(config.runtime, config.data) == 14
    with pytest.raises(ConfigError, match=r"requires 14 CPUs, exceeding runtime.allocated_cpus=13"):
        validate_cpu_budget(replace(config.runtime, allocated_cpus=13), config.data)


def test_compatible_cpu_startup_is_idempotent_without_spawned_workers(compose_settings, tmp_path):
    config = resolve_config(compose_settings("runtime=test_cpu"), invocation_dir=tmp_path)
    configure_cpu_runtime(config.runtime, config.data)
    value = torch.tensor(3.0, requires_grad=True)
    value.square().backward()
    configure_cpu_runtime(config.runtime, config.data)

    assert (torch.get_num_threads(), torch.get_num_interop_threads()) == (1, 1)
    assert os.environ["OMP_NUM_THREADS"] == os.environ["MKL_NUM_THREADS"] == "1"
    assert multiprocessing.active_children() == []


@pytest.mark.parametrize(
    ("runtime", "message"),
    [
        ("allocated_cpus=None", "runtime.allocated_cpus is required"),
        ("allocated_cpus=1, intraop_threads=2", "requires 2 CPUs, exceeding runtime.allocated_cpus=1"),
    ],
)
def test_over_budget_or_undeclared_startup_fails_before_loading_torch(tmp_path, runtime, message):
    result = subprocess.run(
        [sys.executable, "-c", dedent(f"""\
            from dataclasses import replace
            import sys
            from pathlib import Path
            from hydra import compose, initialize_config_module
            from nn_window_ablation.config import ConfigError, resolve_config
            from nn_window_ablation.runtime import configure_cpu_runtime

            with initialize_config_module(config_module="nn_window_ablation.conf", version_base="1.3"):
                config = resolve_config(
                    compose(config_name="config", overrides=["runtime=test_cpu", "run_name=negative"]),
                    invocation_dir=Path.cwd(),
                )
            try:
                configure_cpu_runtime(replace(config.runtime, {runtime}), config.data)
            except ConfigError as error:
                assert "torch" not in sys.modules
                print(error)
            else:
                raise AssertionError("invalid CPU budget was accepted")
            """)],
        cwd=tmp_path, capture_output=True, text=True, timeout=60, check=True,
    )
    assert message in result.stdout


def test_conflicting_interop_setting_reports_initialized_count(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", dedent("""\
            from pathlib import Path
            import torch
            from hydra import compose, initialize_config_module
            from nn_window_ablation.config import ConfigError, resolve_config
            from nn_window_ablation.runtime import configure_cpu_runtime

            torch.set_num_interop_threads(2)
            with initialize_config_module(config_module="nn_window_ablation.conf", version_base="1.3"):
                config = resolve_config(
                    compose(config_name="config", overrides=["runtime=test_cpu", "run_name=negative"]),
                    invocation_dir=Path.cwd(),
                )
            try:
                configure_cpu_runtime(config.runtime, config.data)
            except ConfigError as error:
                print(error)
            else:
                raise AssertionError("late inter-op change was accepted")
            """)],
        cwd=tmp_path, capture_output=True, text=True, timeout=60, check=True,
    )
    assert "runtime.interop_threads=1 cannot be applied" in result.stdout
    assert "PyTorch inter-op initialization at 2" in result.stdout


def test_unrelated_torch_startup_error_propagates(compose_settings, monkeypatch, tmp_path):
    config = resolve_config(compose_settings("runtime=test_cpu", "runtime.interop_threads=2"), invocation_dir=tmp_path)

    def failing_setter(_):
        raise RuntimeError("unrelated failure")

    # Avoid trying to start a two-thread runtime on a one-CPU fallback allocation.
    monkeypatch.setattr(torch, "set_num_interop_threads", failing_setter)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {0, 1}, raising=False)
    with pytest.raises(RuntimeError, match="^unrelated failure$"):
        configure_cpu_runtime(config.runtime, config.data)
