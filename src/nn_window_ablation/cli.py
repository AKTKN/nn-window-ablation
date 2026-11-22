"""The installed nnwa entry point and Hydra orchestration."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.errors import HydraException
from hydra.types import RunMode
from omegaconf import DictConfig

from .config import ConfigError, new_run_name, resolve_config
from .workflows import UNAVAILABLE_WORKFLOWS, ValidationJob, check_workflow, publish_validation, reserve_run_directory


USAGE = "usage: nnwa {validate,generate,train,evaluate,analyze} [Hydra options and overrides]"


def _publish_jobs(command: str, jobs: list[ValidationJob], invocation_dir: Path) -> list[Path]:
    if command == "validate":
        return publish_validation(jobs, invocation_dir=invocation_dir)
    from .execution import generate_workflow, train_workflow

    if command == "generate":
        return [generate_workflow(job.config, overrides=job.overrides) for job in jobs]
    if jobs[0].sweep_index is not None:
        first = jobs[0].config
        if any(
            (job.config.paths.run_root, job.config.run_name) != (first.paths.run_root, first.run_name)
            or job.config.checkpoint.resume_from is not None
            for job in jobs
        ):
            raise ConfigError("new training sweep jobs must share their run parent; resume jobs individually")
        reserve_run_directory(first)
    return [train_workflow(job.config, overrides=job.overrides, sweep_index=job.sweep_index) for job in jobs]


def main() -> int:
    if len(sys.argv) == 1:
        print(USAGE, file=sys.stderr)
        return 2
    command, *arguments = sys.argv[1:]
    if command in ("-h", "--help"):
        print(USAGE)
        return 0
    if command not in ("validate", "generate", "train") and command not in UNAVAILABLE_WORKFLOWS:
        print(f"nnwa: unknown command {command!r}\n{USAGE}", file=sys.stderr)
        return 2

    invocation_dir = Path.cwd()
    jobs: list[ValidationJob] = []

    @hydra.main(version_base="1.3", config_path="conf", config_name="config")
    def run_job(config: DictConfig) -> None:
        try:
            resolved = resolve_config(config, invocation_dir=invocation_dir)
            check_workflow(command, resolved)
        except ConfigError as exc:
            raise HydraException(str(exc)) from None
        hydra_config = HydraConfig.get()
        index = hydra_config.job.num if hydra_config.mode == RunMode.MULTIRUN else None
        jobs.append(ValidationJob(resolved, tuple(hydra_config.overrides.task), index))

    if not any(arg.partition("=")[0].lstrip("+~") == "run_name" for arg in arguments):
        arguments.append(f"run_name={new_run_name()}")
    if command == "train" and not any(arg.partition("=")[0].lstrip("+~") == "data.split" for arg in arguments):
        arguments.append("data.split=train")

    # Hydra writes logs before the task runs. Keep them temporary until all jobs
    # resolve, then reserve the final run identity and publish application records.
    with TemporaryDirectory(prefix="nnwa-hydra-") as scratch:
        arguments.extend([
            f"hydra.run.dir={json.dumps(str(Path(scratch) / 'run'))}",
            f"hydra.sweep.dir={json.dumps(str(Path(scratch) / 'sweep'))}",
            "hydra.sweep.subdir=${hydra.job.num}",
            "hydra.job.chdir=false",
        ])
        original_argv = sys.argv
        try:
            sys.argv = [original_argv[0], *arguments]
            run_job()
        finally:
            sys.argv = original_argv

    if jobs:
        try:
            records = _publish_jobs(command, jobs, invocation_dir)
            message = {
                "validate": "Validated configuration written to",
                "generate": "Dataset manifest written to",
                "train": "Training checkpoint written to",
            }[command]
            for record in records:
                print(f"{message} {record}")
        except (ValueError, OSError) as exc:
            print(f"nnwa: {exc}", file=sys.stderr)
            return 1
    return 0
