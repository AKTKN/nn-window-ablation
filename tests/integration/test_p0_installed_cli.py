import os
from pathlib import Path
import re
import subprocess
import sys
from textwrap import dedent

from filelock import FileLock
import pytest
import yaml


TEST_REVISION = "0123456789abcdef0123456789abcdef01234567"
TEST_PROFILE = (
    "runtime=test_cpu", "runtime.processes=1", "data.num_workers=0", "data.sampling_processes=1",
)


@pytest.fixture(scope="session")
def installed_bin(tmp_path_factory, worker_id):
    """Build and install one wheel into a scratch venv shared by test workers."""
    checkout = Path(__file__).resolve().parents[2]
    base = tmp_path_factory.getbasetemp()
    shared = base if worker_id == "master" else base.parent
    work = shared / "nnwa-wheel-install"
    work.mkdir(exist_ok=True)
    environment = work / "venv"
    bin_dir = environment / "bin"
    with FileLock(work / "install.lock", timeout=240):
        if not (work / "installed").is_file():
            subprocess.run(
                [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                 "--wheel-dir", str(work), str(checkout)], check=True, timeout=120,
            )
            wheel, = work.glob("nn_window_ablation-*.whl")
            subprocess.run(
                [sys.executable, "-m", "venv", "--system-site-packages", str(environment)],
                check=True, timeout=120,
            )
            subprocess.run(
                [str(bin_dir / "python"), "-m", "pip", "install", "--no-deps", "--ignore-installed", str(wheel)],
                check=True, timeout=120,
            )
            (work / "installed").touch()
    return bin_dir


@pytest.fixture
def installed_cli(installed_bin, tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update({
        "PYTHONNOUSERSITE": "1",
        "HYDRA_FULL_ERROR": "0",
        "NNWA_DATA_ROOT": "input data",
        "NNWA_RUN_ROOT": "validation records",
        "NNWA_SOURCE_REVISION": TEST_REVISION,
    })

    def run(command, *overrides, hydra_options=()):
        return subprocess.run(
            [str(installed_bin / "nnwa"), command, *hydra_options, *TEST_PROFILE, *overrides],
            cwd=tmp_path, env=env, text=True, capture_output=True, timeout=60,
        )

    return run


def _read_yaml(path):
    return yaml.safe_load(path.read_text())


def test_wheel_validate_resolves_pilot_and_records_provenance(installed_cli, tmp_path):
    result = installed_cli("validate")
    assert result.returncode == 0, result.stderr
    parent, = (tmp_path / "validation records").iterdir()
    assert re.fullmatch(r"run-[0-9a-f]{32}", parent.name)
    output = parent / "hydra"
    assert str(output / "resolved_config.yaml") in result.stdout
    settings = _read_yaml(output / "resolved_config.yaml")
    assert settings["geometry"] == dict(
        distance=5, rounds=60, core_length=5, left_buffer=5, right_buffer=5, width=15, core_count=12,
    )
    assert settings["paths"] == {
        "data_root": str(tmp_path / "input data"), "run_root": str(tmp_path / "validation records"),
    }
    assert settings["training"]["initialization"] == "random"
    assert settings["runtime"]["device"] == "cpu"
    assert settings["training"]["max_steps"] is None
    assert not (tmp_path / "input data").exists()

    metadata = _read_yaml(output / "metadata.yaml")
    assert metadata["status"] == "configuration_validated"
    assert metadata["source"] == {"commit": TEST_REVISION, "dirty": None, "method": "supplied"}
    assert metadata["packages"]["hydra-core"] == "1.3.2"
    assert metadata["cpu"]["model"]
    assert metadata["invocation_dir"] == str(tmp_path)
    assert metadata["runtime"] == {
        **settings["runtime"], "required_cpus": 1, "data_loader_workers": 0, "sampling_processes": 1,
        "numerical_runtime_started": False,
    }


def test_wheel_multirun_saves_each_experiment_and_override(installed_cli, tmp_path):
    result = installed_cli(
        "validate", "run_name=sweep", "experiment=pilot_d5,paired_d7",
        "window.right_per_distance=0.5", hydra_options=("--multirun",),
    )
    assert result.returncode == 0, result.stderr
    sweep = tmp_path / "validation records" / "sweep" / "hydra_sweep"
    assert sorted(path.name for path in sweep.iterdir()) == ["0", "1"]
    for index, experiment, distance, right in [(0, "pilot_d5", 5, 3), (1, "paired_d7", 7, 4)]:
        child = sweep / str(index)
        settings = _read_yaml(child / "resolved_config.yaml")
        assert settings["data"]["family_id"] == experiment
        assert settings["geometry"]["distance"] == distance
        assert settings["geometry"]["right_buffer"] == right
        assert f"experiment={experiment}" in _read_yaml(child / "overrides.yaml")
        assert _read_yaml(child / "metadata.yaml")["sweep_index"] == index


def test_invalid_sweep_does_not_publish_any_validation_records(installed_cli, tmp_path):
    result = installed_cli("validate", "code.distance=5,4", hydra_options=("--multirun",))
    assert result.returncode != 0
    assert "code.distance must be an odd integer" in result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("command", "overrides", "error"),
    [
        ("analyze", (), "unavailable until P5 analysis"),
        ("evaluate", (), "unavailable until P5 evaluation"),
        ("generate", (), "generate requires explicit values"),
        ("not-a-command", (), "unknown command"),
        ("train", (), "train requires explicit values"),
        ("validate", ("runtime.intraop_threads=9",), "exceeding runtime.allocated_cpus=8"),
    ],
)
def test_cli_rejects_invalid_or_unimplemented_invocations(installed_cli, tmp_path, command, overrides, error):
    result = installed_cli(command, *overrides)
    assert result.returncode != 0
    assert error in result.stderr
    if command == "train":
        assert "training.max_steps" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_wheel_validate_rejects_existing_explicit_run_name(installed_cli, tmp_path):
    first = installed_cli("validate", "run_name=collision")
    assert first.returncode == 0, first.stderr
    record = tmp_path / "validation records" / "collision" / "hydra" / "resolved_config.yaml"
    original = record.read_bytes()

    second = installed_cli("validate", "run_name=collision", "experiment=paired_d7")
    assert second.returncode != 0
    assert "run directory already exists" in second.stderr
    assert record.read_bytes() == original


def test_import_is_inert_and_uses_wheel_install(installed_bin, tmp_path):
    result = subprocess.run(
        [str(installed_bin / "python"), "-IB", "-c", dedent("""\
            import os
            import sys
            from pathlib import Path

            def verify_import(event, args):
                if event in {"os.mkdir", "os.system", "os.fork", "subprocess.Popen"}:
                    raise AssertionError(event)
                if event == "open" and args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
                    raise AssertionError(args)

            sys.addaudithook(verify_import)
            import nn_window_ablation

            package_file = Path(nn_window_ablation.__file__).resolve()
            assert Path(sys.prefix) in package_file.parents, package_file
            assert not ({"torch", "stim", "hydra"} & sys.modules.keys())
            """)],
        cwd=tmp_path, text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []
