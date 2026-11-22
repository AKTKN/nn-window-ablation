import os
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from tests._cpu import configure_torch_threads


def test_worker_starts_with_one_numerical_thread():
    assert torch.get_num_threads() == 1
    assert torch.get_num_interop_threads() == 1
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert os.environ[name] == "1", name


def test_compatible_torch_setup_can_be_repeated():
    value = torch.tensor(2.0, requires_grad=True)
    (value * value).backward()

    configure_torch_threads()

    assert torch.get_num_interop_threads() == 1
    assert torch.get_num_threads() == 1


def test_incompatible_interop_setup_fails_in_fresh_process():
    checkout = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests._cpu import configure_torch_threads\n"
            "import torch\n"
            "torch.set_num_interop_threads(2)\n"
            "configure_torch_threads()\n",
        ],
        cwd=checkout,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode != 0
    assert "in configure_torch_threads" in result.stderr
    assert "cannot set number of interop threads" in result.stderr


def test_packaged_test_cpu_profile_composes_with_pilot():
    config_dir = files("nn_window_ablation").joinpath("conf")
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        settings = compose(
            config_name="config",
            overrides=["runtime=test_cpu", "data.num_workers=0", "data.sampling_processes=1"],
        )

    assert dict(settings.runtime) == {
        "device": "cpu",
        "dtype": "float32",
        "processes": 1,
        "intraop_threads": 1,
        "interop_threads": 1,
        "allocated_cpus": 8,
        "benchmark_threads": [1],
    }
    assert (settings.data.num_workers, settings.data.sampling_processes) == (0, 1)
    assert (settings.code.distance, settings.noise.probability) == (5, 0.003)
    assert (settings.model.hidden_size, settings.model.gru_layers) == (256, 4)
