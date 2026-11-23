"""Plan and run explicit single-node PBS jobs without submitting them."""

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from .artifacts import ArtifactError, atomic_file, file_integrity, read_json, verify_file, write_json
from .config import (
    ConfigError, evaluation_worker_threads, resolve_config, validate_cpu_budget, validate_evaluation_budget,
    validate_run_name,
)
from .workflows import check_workflow


THREAD_VARS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)
ID_PATTERN = r"[0-9]+(?:\[[0-9]*\])?(?:\.[A-Za-z0-9_.-]+)?"
STAGES = {"probe", "generate", "profile", "train", "resume", "evaluate"}


@dataclass(frozen=True, slots=True)
class PBSResources:
    ncpus: int
    memory: str
    walltime: str
    queue: str | None = None
    account: str | None = None

    def __post_init__(self) -> None:
        if type(self.ncpus) is not int or self.ncpus < 1:
            raise ConfigError("PBS resources.ncpus must be a positive integer")
        if not isinstance(self.memory, str) or not re.fullmatch(r"[1-9][0-9]*(?:kb|mb|gb|tb)", self.memory):
            raise ConfigError("PBS resources.memory must use units such as 4gb")
        if not isinstance(self.walltime, str) or not re.fullmatch(r"[0-9]{2,}:[0-5][0-9]:[0-5][0-9]", self.walltime):
            raise ConfigError("PBS resources.walltime must use HH:MM:SS")
        if not any(int(part) for part in self.walltime.split(":")):
            raise ConfigError("PBS resources.walltime must be positive")
        for name in ("queue", "account"):
            value = getattr(self, name)
            pattern = r"[A-Za-z0-9][A-Za-z0-9_.-]*(?:@[A-Za-z0-9][A-Za-z0-9_.-]*)?" if name == "queue" else r"[A-Za-z0-9][A-Za-z0-9_.-]*"
            if value is not None and (not isinstance(value, str) or not re.fullmatch(pattern, value)):
                raise ConfigError(f"invalid PBS {name}")


@dataclass(frozen=True, slots=True)
class JobSpec:
    stage: str
    job_name: str
    run_name: str
    repository: str
    source_revision: str
    conda_executable: str
    conda_prefix: str
    data_root: str
    run_root: str
    receipt_dir: str
    resources: PBSResources
    threads: int
    overrides: list[str] = field(default_factory=list)
    profile: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ConfigError(f"unsupported PBS stage {self.stage}")
        if not isinstance(self.job_name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,14}", self.job_name):
            raise ConfigError("PBS job_name must start with a letter and contain at most 15 portable characters")
        if not isinstance(self.run_name, str):
            raise ConfigError("run_name must be a portable string")
        validate_run_name(self.run_name)
        if not isinstance(self.source_revision, str) or not re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", self.source_revision):
            raise ConfigError("source_revision must be a full lowercase Git commit ID")
        for name in ("repository", "conda_executable", "conda_prefix", "data_root", "run_root", "receipt_dir"):
            path = getattr(self, name)
            if not isinstance(path, str) or not Path(path).is_absolute() or any(ord(char) < 32 for char in path):
                raise ConfigError(f"{name} must be an explicit absolute path")
        if type(self.threads) is not int or not 1 <= self.threads <= self.resources.ncpus:
            raise ConfigError("PBS threads must be positive and within resources.ncpus")
        if type(self.overrides) is not list or any(
            not isinstance(token, str) or not re.fullmatch(r"[+~]?[A-Za-z0-9_.]+=[^\r\n\0]+", token)
            for token in self.overrides
        ):
            raise ConfigError("PBS overrides must be a list of single Hydra field assignments")
        owned = {"run_name", "paths", "paths.data_root", "paths.run_root", "runtime.intraop_threads", "runtime.allocated_cpus"}
        keys = {token.partition("=")[0].lstrip("+~") for token in self.overrides}
        if keys & owned or any(key.startswith("hydra") for key in keys):
            raise ConfigError("PBS plan owns run_name, roots, allocated CPUs, threads, and Hydra output settings")
        if any(token == "runtime=test_cpu" for token in self.overrides) and (self.resources.ncpus > 8 or self.threads != 1):
            raise ConfigError("runtime=test_cpu uses at most eight allocated CPUs and one numerical thread")
        values = {"batch_size": 2, "warmup": 1, "steps": 2}
        if type(self.profile) is not dict or not set(self.profile) <= values.keys():
            raise ConfigError("profile fields are batch_size, warmup, and steps")
        if self.stage != "profile" and self.profile:
            raise ConfigError("profile settings apply to the profile stage")
        if any(type(value) is not int or value < (0 if name == "warmup" else 1) for name, value in self.profile.items()):
            raise ConfigError("profile batch_size and steps must be positive and warmup nonnegative")
        if self.stage in ("profile", "probe") and keys - {"runtime"}:
            raise ConfigError("profile and probe use the explicit PBS CPU and profile settings")

    @classmethod
    def from_record(cls, record: dict) -> "JobSpec":
        try:
            return cls(**{**record, "resources": PBSResources(**record["resources"])})
        except (KeyError, TypeError) as exc:
            raise ConfigError(f"missing or invalid PBS job field: {exc}") from exc


def _overrides(job: JobSpec) -> list[str]:
    settings = ["runtime=cpu", *job.overrides]
    keys = {token.partition("=")[0].lstrip("+~") for token in job.overrides}
    split = {"train": "train", "resume": "train", "evaluate": "test"}.get(job.stage)
    if split is not None and "data.split" not in keys:
        settings.append(f"data.split={split}")
    return [
        *settings, f"run_name={job.run_name}", f"paths.data_root={json.dumps(job.data_root)}",
        f"paths.run_root={json.dumps(job.run_root)}", f"runtime.intraop_threads={job.threads}",
        f"runtime.allocated_cpus={job.resources.ncpus}",
    ]


def _configuration(job: JobSpec):
    from hydra import compose, initialize_config_module
    from hydra.errors import HydraException

    try:
        with initialize_config_module(config_module="nn_window_ablation.conf", version_base="1.3"):
            settings = compose(config_name="config", overrides=_overrides(job))
        config = resolve_config(settings, invocation_dir=Path(job.repository))
    except HydraException as exc:
        raise ConfigError(f"invalid PBS workflow overrides: {exc}") from exc
    if job.stage in ("probe", "profile"):
        validate_cpu_budget(config.runtime, config.data)
        return config
    command = "train" if job.stage == "resume" else job.stage
    check_workflow(command, config)
    if job.stage == "generate" and config.data.id is None:
        raise ConfigError("PBS generate requires an explicit data.id for its output receipt")
    if (config.checkpoint.resume_from is not None) != (job.stage == "resume"):
        raise ConfigError("PBS resume requires checkpoint.resume_from and the original run_name")
    if job.stage == "resume":
        path = config.checkpoint.resume_from
        if path.name != "checkpoint.pt" or not path.is_relative_to(config.paths.run_root / config.run_name):
            raise ConfigError("checkpoint.resume_from must locate checkpoint.pt in the selected original run")
    if config.runtime.interop_threads != 1:
        raise ConfigError("PBS workflows require one inter-operation thread")
    return config


def render_script(job: JobSpec) -> str:
    """Validate and render content without invoking any scheduler command."""
    _configuration(job)
    resources = job.resources
    lines = [
        "#!/bin/bash", f"#PBS -N {job.job_name}",
        f"#PBS -l select=1:ncpus={resources.ncpus}:mem={resources.memory}",
        f"#PBS -l walltime={resources.walltime}", "#PBS -j oe",
    ]
    if resources.queue is not None:
        lines.append(f"#PBS -q {resources.queue}")
    if resources.account is not None:
        lines.append(f"#PBS -A {resources.account}")
    command = [
        job.conda_executable, "run", "--no-capture-output", "-p", job.conda_prefix,
        "python", "-m", "nn_window_ablation.pbs", json.dumps(asdict(job), separators=(",", ":")),
    ]
    return "\n".join([
        *lines, "", "set -euo pipefail", "umask 077",
        *(f"export {name}={job.threads}" for name in THREAD_VARS),
        "exec " + shlex.join(command), "",
    ])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def planned_job(job: JobSpec) -> dict:
    return {"schema_version": 1, "status": "planned", "plan": asdict(job)}


def start_job(record: dict, observed: dict) -> dict:
    if record.get("status") != "planned":
        raise ValueError("only a planned job can start")
    return {**record, "status": "running", "observed": observed, "started_at": _now()}


def _artifact_receipt(artifact: dict | None) -> bool:
    from .continuation import is_digest

    valid = (
        isinstance(artifact, dict) and isinstance(artifact.get("path"), str) and bool(artifact["path"])
        and type(artifact.get("bytes")) is int and artifact["bytes"] > 0 and is_digest(artifact.get("sha256"))
    )
    if not valid:
        return False
    try:
        verify_file(Path(artifact["path"]), {key: artifact[key] for key in ("bytes", "sha256")})
    except ArtifactError:
        return False
    return True


def finish_job(record: dict, *, exit_code: int | None, artifact: dict | None = None, error: str | None = None) -> dict:
    if record.get("status") != "running":
        raise ValueError("only a running job can finish")
    status = "incomplete"
    if error is not None or (type(exit_code) is int and exit_code != 0):
        status = "failed"
    elif type(exit_code) is int and exit_code == 0 and _artifact_receipt(artifact):
        status = "complete"
    return {**record, "status": status, "ended_at": _now(), "exit_code": exit_code, "artifact": artifact, "error": error}


def scheduler_status(job_id: str, qstat: str, *, receipt: dict | None = None, transport=subprocess.run) -> dict:
    """Query OpenPBS explicitly and distinguish scheduler exit from artifact completion."""
    if not re.fullmatch(ID_PATTERN, job_id):
        raise ConfigError("invalid PBS job ID")
    completed = transport([qstat, "-x", "-f", "-F", "json", job_id], capture_output=True, text=True, check=True)
    try:
        entry = json.loads(completed.stdout)["Jobs"][job_id]
        state = entry["job_state"]
    except (KeyError, ValueError, TypeError) as exc:
        raise ArtifactError("qstat response lacks the requested job and state") from exc
    status = {"Q": "planned", "H": "planned", "W": "planned", "R": "running", "E": "running"}.get(state, "unknown")
    exit_code = entry.get("Exit_status")
    if state in ("F", "C"):
        status = "incomplete"
        if type(exit_code) is int and exit_code != 0:
            status = "failed"
        elif receipt is not None and receipt.get("observed", {}).get("pbs_job_id") == job_id:
            if receipt.get("status") == "failed":
                status = "failed"
            elif (type(exit_code) is int and exit_code == 0 and receipt.get("status") == "complete"
                    and receipt.get("exit_code") == 0):
                if _artifact_receipt(receipt.get("artifact")):
                    status = "complete"
    return {
        "job_id": job_id, "pbs_state": state, "status": status, "exit_code": exit_code,
        "reported_request": entry.get("Resource_List"), "resources_used": entry.get("resources_used"),
        "exec_vnode": entry.get("exec_vnode"),
    }


def _observed() -> dict:
    import platform

    slots = None
    nodefile = os.environ.get("PBS_NODEFILE")
    if nodefile and Path(nodefile).is_file():
        slots = dict(Counter(line for line in Path(nodefile).read_text().splitlines() if line))
    count = os.environ.get("PBS_NCPUS", "")
    return {
        "pbs_job_id": os.environ.get("PBS_JOBID"), "pbs_queue": os.environ.get("PBS_QUEUE"),
        "pbs_submission_directory": os.environ.get("PBS_O_WORKDIR"),
        "hostname": platform.node(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "pbs_ncpus": int(count) if count.isdecimal() else None, "pbs_nodefile_slots": slots,
        "native_threads": {name: os.environ.get(name) for name in THREAD_VARS},
    }


def _artifact_path(job: JobSpec, config) -> Path:
    if job.stage in ("profile", "probe"):
        return Path(job.receipt_dir) / f"{job.stage}.json"
    if job.stage == "generate":
        return config.paths.data_root / config.data.id / "manifest.json"
    if job.stage == "resume":
        return config.checkpoint.resume_from
    return config.paths.run_root / config.run_name / ("checkpoint.pt" if job.stage == "train" else "manifest.json")


def _command(job: JobSpec, artifact: Path) -> list[str]:
    if job.stage == "profile":
        profile = {"batch_size": 2, "warmup": 1, "steps": 2, **job.profile}
        return [
            sys.executable, "-m", "nn_window_ablation.profiling",
            "--threads", str(job.threads), "--allocated-cpus", str(job.resources.ncpus),
            "--batch-size", str(profile["batch_size"]), "--warmup", str(profile["warmup"]),
            "--steps", str(profile["steps"]), "--output", str(artifact),
        ]
    if job.stage == "probe":
        return [
            sys.executable, "-c",
            "import torch, torch_geometric, stim, pymatching, scipy; "
            "print(torch.__version__, torch.version.cuda, torch_geometric.__version__, stim.__version__, "
            "pymatching.__version__, scipy.__version__)",
        ]
    command = "train" if job.stage == "resume" else job.stage
    return [str(Path(sys.executable).with_name("nnwa")), command, *_overrides(job)]


def _verified_artifact(job: JobSpec, config, path: Path) -> tuple[dict, dict]:
    intraop = config.runtime.intraop_threads
    if job.stage in ("train", "resume"):
        from .checkpoints import load_checkpoint
        from .execution import _resume_contract
        from .training import final_update_count

        state = load_checkpoint(path)
        final_update_count(state)
        receipt = read_json(path.with_name("run.json"))
        if (receipt.get("status") != "complete" or receipt.get("checkpoint") != file_integrity(path)
                or state["run"]["resume_contract"] != _resume_contract(config.to_record())):
            raise ArtifactError("training did not publish the expected terminal run")
        actual = state["run"]["runtime"]
    elif job.stage == "generate":
        from .data import open_dataset

        mapped = open_dataset(path.parent, expected_split=config.data.split)
        mapped.verify_payloads()
        provenance = read_json(path)["provenance"]
        if mapped.plan.total_rows != config.data.shots or provenance["resolved_config"] != config.to_record():
            raise ArtifactError("generation did not publish the expected dataset")
        actual = provenance["runtime"]
    elif job.stage == "evaluate":
        from .paired_workflows import _dataset, _evaluation_request, _read_manifest, _read_shard, _verify_commit

        _, mapped = _dataset(config)
        manifest = _read_manifest(path, mapped)
        checkpoint_digest = file_integrity(config.checkpoint.path)["sha256"]
        if (
            _evaluation_request(manifest["resolved_config"]) != _evaluation_request(config.to_record())
            or any(
                item["checkpoint"]["sha256"] != checkpoint_digest
                for item in manifest["conditions"].values() if item["checkpoint"] is not None
            )
        ):
            raise ArtifactError("evaluation manifest does not match the planned run")
        if "contract_sha256" in manifest and {p.name for p in (path.parent / "shards").iterdir()} != {
            str(desc.index) for desc in mapped.shards
        }:
            raise ArtifactError("evaluation artifact has missing or unexpected committed shard directories")
        for descriptor in mapped.shards:
            with mapped.open_shard(descriptor.index) as shard:
                if "contract_sha256" in manifest:
                    entry = _verify_commit(
                        path.parent / "shards" / str(descriptor.index), descriptor, shard,
                        manifest["conditions"], manifest["contract_sha256"],
                    )
                    if entry != manifest["shards"][descriptor.index]:
                        raise ArtifactError("evaluation inventory disagrees with the committed shard receipt")
                else:
                    for _ in _read_shard(path, manifest, shard):
                        pass
        actual = manifest["runtime"]
        intraop = evaluation_worker_threads(config.runtime, config.evaluation)
    elif job.stage == "profile":
        profile = read_json(path)
        if profile.get("runtime", {}).get("torch_intraop") != job.threads or "mean_seconds" not in profile:
            raise ArtifactError("profiling did not publish the expected measurements")
        actual = profile["runtime"]
    else:
        actual = {}
    if job.stage != "probe" and (
        actual["torch_intraop"], actual["torch_interop"]
    ) != (intraop, config.runtime.interop_threads):
        raise ArtifactError("workflow PyTorch threads differ from the planned runtime")
    return {"path": str(path), **file_integrity(path)}, actual


def execute_job(job: JobSpec) -> int:
    """Run inside the selected Conda environment and record actual execution evidence."""
    config = _configuration(job)
    directory = Path(job.receipt_dir)
    directory.mkdir()
    status = directory / "job.json"
    plan = planned_job(job)
    write_json(status, plan)
    observed = _observed()
    record = start_job(plan, observed)
    record["resolved_runtime"] = {
        "torch_intraop_requested": config.runtime.intraop_threads,
        "torch_interop_requested": config.runtime.interop_threads,
        "workers": config.data.num_workers, "sampling_processes": config.data.sampling_processes,
    }
    if job.stage == "evaluate":
        record["resolved_runtime"].update(
            evaluation_workers=config.evaluation.workers,
            evaluation_worker_threads=evaluation_worker_threads(config.runtime, config.evaluation),
            evaluation_max_in_flight=config.evaluation.max_in_flight,
        )
    write_json(status, record)
    code = None
    artifact = _artifact_path(job, config)
    try:
        root = Path(job.repository)
        revision = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
        if revision != job.source_revision or subprocess.check_output(["git", "-C", root, "status", "--porcelain"]):
            raise ArtifactError("require a clean checkout at the planned Git revision")
        if Path(sys.prefix).resolve() != Path(job.conda_prefix).resolve():
            raise ArtifactError("compute process is outside the planned Conda prefix")
        from .execution import _package_source_sha256

        if _package_source_sha256(root / "src/nn_window_ablation") != _package_source_sha256():
            raise ArtifactError("installed package files differ from the planned checkout")
        if any(not path.is_dir() for path in (config.paths.data_root, config.paths.run_root)):
            raise ArtifactError("planned data and run roots must be visible directories on the compute node")
        required = validate_evaluation_budget(config) if job.stage == "evaluate" else validate_cpu_budget(config.runtime, config.data)
        available = observed["pbs_ncpus"]
        if required > len(observed["cpu_affinity"]) or (available is not None and required > available):
            raise ArtifactError("numerical CPU budget exceeds observed compute allocation")
        resuming = job.stage == "resume" or (job.stage == "evaluate" and config.evaluation.resume)
        if not resuming and os.path.lexists(artifact):
            raise ArtifactError("job output exists before execution")
        with (directory / "stdout.log").open("xb") as log:
            result = subprocess.run(_command(job, artifact), cwd=root, stdout=log, stderr=subprocess.STDOUT)
        code = result.returncode
        if code:
            raise ArtifactError(f"workflow exited with code {code}")
        if job.stage == "probe":
            from .workflows import _environment_record

            write_json(artifact, {"schema_version": 1, "status": "complete", "environment": _environment_record()})
        evidence, actual = _verified_artifact(job, config, artifact)
        record["workflow_runtime"] = actual
        terminal = finish_job(record, exit_code=code, artifact=evidence)
        if terminal["status"] != "complete":
            raise ArtifactError("workflow artifact changed before completion could be recorded")
        write_json(status, terminal)
        print(f"Completed PBS workflow. Receipt at {status}")
        return 0
    except (ValueError, OSError, subprocess.SubprocessError, KeyError) as exc:
        write_json(status, finish_job(record, exit_code=code, error=str(exc)))
        print(f"nnwa PBS: {exc}. Receipt at {status}", file=sys.stderr)
        return code if type(code) is int and code > 0 else 1


def write_script(job: JobSpec, output: Path) -> dict:
    script = render_script(job)
    if os.path.lexists(output):
        raise FileExistsError(output)
    with atomic_file(output) as file:
        file.write(script.encode())
    return planned_job(job)


if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise ConfigError("invoke the compute runner through a rendered PBS script")
        sys.exit(execute_job(JobSpec.from_record(json.loads(sys.argv[1]))))
    except (ValueError, OSError) as exc:
        print(f"nnwa PBS: {exc}", file=sys.stderr)
        sys.exit(1)
