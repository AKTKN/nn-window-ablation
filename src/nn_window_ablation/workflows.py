"""Dispatch workflow readiness checks and publish validation records."""

from dataclasses import asdict, dataclass
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

from omegaconf import OmegaConf

from .config import ConfigError, ResolvedConfig, require_execution_config, validate_cpu_budget, validate_run_name


UNAVAILABLE_WORKFLOWS = {
    "generate": "P4 dataset generation (after P1 circuit sampling)",
    "train": "P4 training",
    "evaluate": "P5 evaluation",
    "analyze": "P5 analysis",
}


@dataclass(frozen=True)
class ValidationJob:
    config: ResolvedConfig
    overrides: tuple[str, ...]
    sweep_index: int | None


def check_workflow(command: str, config: ResolvedConfig) -> None:
    if command == "validate":
        return
    message = f"nnwa {command} is unavailable until {UNAVAILABLE_WORKFLOWS[command]} is implemented"
    try:
        require_execution_config(config, command)
    except ConfigError as exc:
        raise ConfigError(f"{message}. {exc}.") from exc
    raise ConfigError(message)


def reserve_run_directory(config: ResolvedConfig) -> Path:
    """Claim a new run name exclusively, including across concurrent invocations."""
    validate_run_name(config.run_name)
    run_dir = config.paths.run_root / config.run_name
    try:
        config.paths.run_root.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir()
    except FileExistsError as exc:
        raise ConfigError(f"run directory already exists: {run_dir}; choose a new run_name") from exc
    except OSError as exc:
        raise ConfigError(f"cannot create run directory {run_dir}: {exc}") from exc
    return run_dir


def publish_validation(
    jobs: list[ValidationJob], *, invocation_dir: Path
) -> list[Path]:
    """Write one validate record per composed Hydra job after the batch passes."""
    first = jobs[0].config
    if any(
        (job.config.paths.run_root, job.config.run_name) != (first.paths.run_root, first.run_name)
        for job in jobs
    ):
        raise ConfigError("a sweep must share paths.run_root and run_name across its jobs")
    environment = _environment_record()
    run_dir = reserve_run_directory(first)
    records = []
    try:
        for job in jobs:
            output = run_dir / "hydra"
            if job.sweep_index is not None:
                output = run_dir / "hydra_sweep" / str(job.sweep_index)
            output.mkdir(parents=True)
            config = job.config
            record = output / "resolved_config.yaml"
            OmegaConf.save(config.to_record(), record)
            OmegaConf.save(list(job.overrides), output / "overrides.yaml")
            OmegaConf.save(
                {
                    "command": "validate",
                    "status": "configuration_validated",
                    "run_name": config.run_name,
                    "sweep_index": job.sweep_index,
                    "invocation_dir": str(invocation_dir),
                    "source": environment["source"],
                    "python": environment["python"],
                    "packages": environment["packages"],
                    "cpu": environment["cpu"],
                    "runtime": {
                        **asdict(config.runtime),
                        "required_cpus": validate_cpu_budget(config.runtime, config.data),
                        "data_loader_workers": config.data.num_workers,
                        "sampling_processes": config.data.sampling_processes,
                        "numerical_runtime_started": False,
                    },
                },
                output / "metadata.yaml",
            )
            records.append(record)
    except (OSError, ValueError) as exc:
        shutil.rmtree(run_dir)
        raise ConfigError(f"cannot write validation records to {run_dir}: {exc}") from exc
    return records


def _source_record() -> dict:
    try:
        distribution = metadata.distribution("nn-window-ablation")
        direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
        commit = direct_url.get("vcs_info", {}).get("commit_id")
        if commit:
            return {"commit": commit, "dirty": None, "method": "pip_vcs"}
    except (metadata.PackageNotFoundError, ValueError):
        pass

    checkout = Path(__file__).resolve().parents[2]
    if (checkout / ".git").exists() and (checkout / "pyproject.toml").is_file():
        try:
            commit = subprocess.check_output(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            status = subprocess.check_output(
                ["git", "-C", str(checkout), "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            )
            return {"commit": commit, "dirty": bool(status), "method": "checkout"}
        except (OSError, subprocess.CalledProcessError):
            pass
    supplied = os.environ.get("NNWA_SOURCE_REVISION")
    return {"commit": supplied, "dirty": None, "method": "supplied" if supplied else "unavailable"}


def _environment_record() -> dict:
    packages = {}
    for name in (
        "nn-window-ablation", "hydra-core", "omegaconf", "torch", "torch-geometric",
        "stim", "pymatching", "numpy", "scipy", "matplotlib", "pytest", "pytest-xdist",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    cpu_model = platform.processor() or platform.machine()
    try:
        with Path("/proc/cpuinfo").open() as cpu_info:
            for line in cpu_info:
                if line.startswith("model name"):
                    cpu_model = line.partition(":")[2].strip()
                    break
    except OSError:
        pass
    return {
        "source": _source_record(),
        "python": platform.python_version(),
        "packages": packages,
        "cpu": {
            "model": cpu_model,
            "platform": platform.platform(),
            "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        },
    }
