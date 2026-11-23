"""Publish and verify paired predictions and whole-trajectory analysis records."""

from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from importlib import metadata
from io import StringIO
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING
from uuid import uuid4
import csv
import fcntl
import os
import re
import shutil
import sys

from .artifacts import (
    ArtifactError, _sync_directory, atomic_file, canonical_json, exclusive_lock, file_integrity, publish_directory,
    read_json, read_json_artifact, staged_directory, verify_file, write_json,
)
from .config import (
    ConfigError, ResolvedConfig, evaluation_worker_threads, require_execution_config, validate_evaluation_budget,
    validate_run_name,
)
from .evaluation_pool import append_event, cpu_affinity, observe_process, run_pool, stop_on_signal
from .runtime import (
    NATIVE_THREAD_VARIABLES, configure_cpu_runtime, native_thread_limits, reject_excessive_native_threads,
    runtime_observation,
)
from .workflows import _environment_record

if TYPE_CHECKING:
    from .data import MappedDataset
    from .evaluation import FrozenEvaluator, MatchingReference


PREDICTION_SCHEMA = 3
EVALUATION_CONTRACT_SCHEMA = 2
PRIMARY_CONTRASTS = ("S_minus_G", "F_minus_G", "S_minus_F")
EVALUATION_PRESETS = {
    "G": ("global", "full_closed", "identity"),
    "F": ("all_cores", "full_closed", "xor"),
    "S": ("all_cores", "deployment_window", "xor"),
    "paired": ("global_and_all_cores", "paired", "identity_and_xor"),
}


def check_paired_execution(command: str, config: ResolvedConfig) -> None:
    require_execution_config(config, command)
    validate_run_name(config.data.id)
    if config.data.split not in ("test", "validation") or config.geometry.core_count != 12:
        raise ConfigError("paired workflows require a held-out split and the complete 12-core partition")
    if (
        (config.runtime.processes, config.data.num_workers, config.data.sampling_processes) != (1, 0, 1)
        or any(path is not None for path in (config.checkpoint.init_from, config.checkpoint.resume_from))
    ):
        raise ConfigError("paired workflows require a single process and no training checkpoint operations")
    if command == "evaluate":
        settings = config.evaluation
        validate_evaluation_budget(config)
        if (
            EVALUATION_PRESETS.get(settings.id) != (settings.query, settings.context, settings.aggregation)
            or settings.threshold != 0.5 or not settings.reset_recurrent_state_per_query or settings.update_model_state
        ):
            raise ConfigError("evaluation must use its declared G/F/S query, context, threshold, and frozen state")
    else:
        settings = config.analysis
        if (
            (settings.unit, settings.confidence_level, settings.ler_interval, settings.paired_interval,
             settings.sparse_interval, settings.pool_training_seeds_as_independent_shots)
            != ("whole_trajectory", 0.95, "wilson", "paired_bootstrap", "conservative_exact_discordances", False)
        ):
            raise ConfigError("analysis must use the whole-trajectory 95% interval policy and separate training seeds")
        if not settings.predictions or len(set(settings.predictions)) != len(settings.predictions):
            raise ConfigError("analysis.predictions must list distinct evaluation manifest paths")
        if not settings.contrasts or len(set(settings.contrasts)) != len(settings.contrasts):
            raise ConfigError("analysis.contrasts must select distinct paired contrasts")


def _run_path(config: ResolvedConfig, sweep_index: int | None) -> Path:
    root = config.paths.run_root / config.run_name
    if sweep_index is not None:
        root = root / "hydra_sweep" / str(sweep_index)
    root.parent.mkdir(parents=True, exist_ok=True)
    return root


def _dataset(config: ResolvedConfig):
    from .data import BITORDER, LABEL_SCHEMA, open_dataset
    from .simulation import CircuitSpec

    geometry = config.geometry
    data_root = config.paths.data_root / config.data.id
    mapped = open_dataset(data_root, expected_split=config.data.split)
    expected = CircuitSpec(geometry.distance, geometry.rounds, geometry.core_length, config.noise.probability)
    if mapped.plan.spec != expected:
        raise ArtifactError("dataset circuit differs from the resolved evaluation configuration")
    if (config.data.label_schema, config.data.bitorder, config.data.storage) != (LABEL_SCHEMA, BITORDER, "packed_npy"):
        raise ConfigError("dataset configuration differs from the packed label schema")
    return data_root, mapped


def _dataset_record(mapped) -> dict:
    recipe = mapped.plan.record()
    return {
        "dataset_id": mapped.plan.dataset_id, "manifest_sha256": mapped.manifest_sha256,
        "split": mapped.plan.namespace.split, "core_count": mapped.plan.core_count,
        "label_schema": recipe["label_schema"], "feature_schema": recipe["feature_schema"],
        "bitorder": recipe["bitorder"], "shots": mapped.plan.total_rows,
    }


def _write_array(path: Path, values) -> dict:
    import numpy as np

    with atomic_file(path) as file:
        np.save(file, values, allow_pickle=False)
    return {**file_integrity(path), "shape": list(values.shape), "dtype": values.dtype.name}


def _query_record(view) -> dict:
    metadata = view.metadata
    return {
        "core_index": metadata.target.core_index,
        "target_first_round": metadata.target.first_round, "target_last_round": metadata.target.last_round,
        "first_observed_round": metadata.first_observed_round, "last_observed_round": metadata.last_observed_round,
        "release": {"round": metadata.release.round, "phase": metadata.release.phase.name.lower()},
        "preparation": metadata.preparation, "destructive_readout": metadata.destructive_readout,
        "buffers": None if metadata.buffers is None else asdict(metadata.buffers),
        "detector_count": len(view.detectors),
    }


def evaluate_workflow(
    config: ResolvedConfig, *, overrides: tuple[str, ...] = (), sweep_index: int | None = None,
) -> Path:
    started = perf_counter()
    check_paired_execution("evaluate", config)
    settings = config.evaluation
    threads = evaluation_worker_threads(config.runtime, settings)
    required = validate_evaluation_budget(config)
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    if affinity is not None and required > len(affinity):
        raise ConfigError(f"evaluation requires {required} CPUs, exceeding process CPU affinity")
    if settings.workers == 1:
        with stop_on_signal() as check_stop:
            return _run_evaluation(config, overrides, sweep_index, started, threads, (), None, check_stop)
    if not sys.platform.startswith("linux") or affinity is None:
        raise ConfigError("parallel evaluation requires Linux process affinity and parent-death signals")
    reject_excessive_native_threads(threads, "evaluation worker")
    allocation = config.runtime.allocated_cpus
    if allocation > len(affinity):
        raise ConfigError("evaluation allocation exceeds process CPU affinity")
    slots = tuple(tuple(affinity[i:i + threads]) for i in range(0, required, threads))
    # The coordinator can share a worker CPU when the allocation is tight.
    coordinator_cpu = affinity[required] if allocation > required else affinity[0]
    with stop_on_signal() as check_stop, cpu_affinity(affinity[:allocation]), native_thread_limits(1):
        return _run_evaluation(config, overrides, sweep_index, started, threads, slots, coordinator_cpu, check_stop)


def _evaluation_runtime(config, threads):
    import torch

    return {
        **runtime_observation(config.data), "torch_intraop": threads,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
    }


def _evaluation_inputs(config, mapped, data_root, modes, expected_checkpoint=None):
    import stim

    from .checkpoints import load_checkpoint
    from .continuation import model_tensors_sha256
    from .evaluation import CheckpointProvenance
    from .windows import Buffers, build_observation_layout, select_window

    checkpoint = config.checkpoint.path
    receipt = expected_checkpoint if expected_checkpoint is not None else file_integrity(checkpoint)
    if expected_checkpoint is not None:
        verify_file(checkpoint, receipt)
    payload = load_checkpoint(checkpoint)
    verify_file(checkpoint, receipt)
    provenance = CheckpointProvenance.from_payload(payload, receipt["sha256"], mapped.plan)
    model_payload = payload["model"]
    del payload
    state_digest = model_tensors_sha256(model_payload["state_dict"])
    layout = build_observation_layout(mapped.bundle)
    geometry = config.geometry
    buffers = Buffers(geometry.left_buffer, geometry.right_buffer)
    conditions = {}
    for mode in modes:
        targets = (layout.global_query,) if mode == "G" else layout.queries
        conditions[mode] = {
            "mode": mode, "condition": provenance.condition(mode), "checkpoint": asdict(provenance),
            "model_state_sha256": state_digest,
            "queries": [_query_record(select_window(layout, target, buffers if mode == "S" else None)) for target in targets],
        }
    dem = None
    if config.evaluation.matching:
        dem_path = data_root / "global.dem"
        dem_receipt = file_integrity(dem_path)
        dem = stim.DetectorErrorModel(dem_path.read_text())
        verify_file(dem_path, dem_receipt)
        if (dem.num_detectors, dem.num_observables) != (mapped.plan.detector_count, 1):
            raise ArtifactError("matching requires the original global-only DEM and its detector dimensions")
        conditions["MWPM"] = {
            "mode": "MWPM", "condition": "MWPM", "checkpoint": None, "model_state_sha256": None, "queries": [],
            "baseline": {
                "implementation": "pymatching", "version": metadata.version("pymatching"), "dem": dem_receipt,
                "decompose_errors": True, "num_observables": dem.num_observables,
                "num_detectors": dem.num_detectors, "options": "PyMatching defaults",
            },
        }
    return model_payload, receipt, conditions, buffers, dem


@dataclass(frozen=True, slots=True)
class EvaluationTask:
    index: int
    attempt: str
    contract_sha256: str


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    task: EvaluationTask
    receipt: dict[str, int | str]
    rows: int
    output_bytes: int
    output_files: int
    worker_pid: int
    started: float


@dataclass(frozen=True, slots=True)
class EvaluationFailure:
    task: EvaluationTask
    error: str


@dataclass(slots=True)
class _TaskRunner:
    mapped: "MappedDataset"
    evaluator: "FrozenEvaluator"
    matching: "MatchingReference | None"
    modes: tuple[str, ...]
    batch_size: int
    state_digest: str
    contract_sha256: str
    stage: Path
    log: Path

    def run(self, task: EvaluationTask, *, check_stop=lambda: None) -> EvaluationResult:
        from .evaluation import model_state_sha256

        if task.attempt != self.stage.name or task.contract_sha256 != self.contract_sha256:
            raise ArtifactError(f"evaluation task {task} has incompatible attempt or run identity")
        index = task.index
        descriptor = self.mapped.shards[index]
        started = perf_counter()
        timings = defaultdict(float)
        before = observe_process()
        append_event(self.log, "task_start", index=index, shots=descriptor.rows, process=before)
        mappings = active_fds = None
        opened = perf_counter()
        try:
            with self.mapped.open_shard(index) as shard:
                timings["source_verify_open"] += perf_counter() - opened
                mappings = shard.mapped_arrays
                active_fds = observe_process()["fd_count"]
                entry = _evaluate_shard(
                    self.stage / str(index), descriptor, shard, self.evaluator, self.modes, self.matching,
                    self.batch_size, self.contract_sha256, timings, check_stop,
                )
            started_hash = perf_counter()
            if model_state_sha256(self.evaluator.model) != self.state_digest:
                raise ArtifactError("frozen evaluation changed model parameters or buffers")
            timings["model_state_verification"] = perf_counter() - started_hash
        except BaseException:
            append_event(
                self.log, "task_failed", index=index, process=observe_process(), dataset_active=self.mapped._active,
            )
            raise
        after = observe_process()
        receipt = file_integrity(self.stage / str(index) / "receipt.json")
        resources = {
            "index": index, "shots": entry["rows"], "started": started, "wall_seconds": perf_counter() - started,
            "timings": dict(timings), "before": before, "after": after, "active_mappings": mappings,
            "active_fd_count": active_fds, "dataset_active_after": self.mapped._active,
            "source_bytes": descriptor.manifest_bytes + sum(array.bytes for array in descriptor.arrays),
            "output_bytes": receipt["bytes"] + sum(file["bytes"] for file in entry["files"].values()),
            "output_files": 1 + len(entry["files"]),
        }
        append_event(self.log, "task_done", **resources)
        return EvaluationResult(
            task, receipt, entry["rows"], resources["output_bytes"], resources["output_files"], os.getpid(), started,
        )


def _build_task_runner(config, mapped, model_payload, buffers, dem, modes, conditions, digest, stage, log):
    from .evaluation import FrozenEvaluator, MatchingReference, model_state_sha256
    from .models.gnn_rnn import QueryDecoder

    model = QueryDecoder.from_checkpoint(model_payload)
    evaluator = FrozenEvaluator(model, mapped.bundle, buffers)
    state_digest = conditions[modes[0]]["model_state_sha256"]
    if model_state_sha256(model) != state_digest:
        raise ArtifactError("loaded evaluation model state differs from the checkpoint")
    matching = None if dem is None else MatchingReference(dem, detector_count=mapped.plan.detector_count)
    return _TaskRunner(mapped, evaluator, matching, modes, config.evaluation.batch_size, state_digest, digest, stage, log)


_worker_runner: _TaskRunner | None = None


def _initialize_evaluation_worker(config, semantic, modes, threads, stage, log_dir):
    global _worker_runner
    start = perf_counter()
    # A worker's budget is its own affinity slice. The coordinator validates the full allocation.
    local_runtime = replace(config.runtime, allocated_cpus=threads, intraop_threads=threads)
    configure_cpu_runtime(local_runtime, config.data)
    import torch

    from .execution import _package_source_sha256

    numerical = semantic["runtime"]
    torch.use_deterministic_algorithms(numerical["deterministic_algorithms"], warn_only=numerical["deterministic_warn_only"])
    torch.set_float32_matmul_precision(numerical["float32_matmul_precision"])
    torch.backends.mkldnn.enabled = numerical["mkldnn_enabled"]
    if _evaluation_runtime(config, torch.get_num_threads()) != numerical:
        raise ArtifactError("evaluation worker numerical settings differ from the coordinator")
    if _package_source_sha256() != semantic["environment"]["package_source_sha256"]:
        raise ArtifactError("evaluation worker source differs from the coordinator")
    metadata_start = perf_counter()
    data_root, mapped = _dataset(config)
    if _dataset_record(mapped) != semantic["data"]:
        raise ArtifactError("evaluation worker dataset differs from the coordinator")
    checkpoint_start = perf_counter()
    model_payload, _, conditions, buffers, dem = _evaluation_inputs(
        config, mapped, data_root, modes, expected_checkpoint=semantic["checkpoint"],
    )
    if conditions != semantic["conditions"]:
        raise ArtifactError("evaluation worker checkpoint or query policy differs from the coordinator")
    model_start = perf_counter()
    _worker_runner = _build_task_runner(
        config, mapped, model_payload, buffers, dem, modes, conditions, _semantic_digest(semantic),
        stage, log_dir / f"worker-{os.getpid()}.jsonl",
    )
    return {
        "model_loads": 1, "runtime": numerical, "native_env": {name: os.environ[name] for name in NATIVE_THREAD_VARIABLES},
        "timings": {
            "runtime": metadata_start - start, "dataset_metadata": checkpoint_start - metadata_start,
            "checkpoint_and_queries": model_start - checkpoint_start, "model_initialization": perf_counter() - model_start,
        },
    }


def _run_worker_task(task: EvaluationTask) -> EvaluationResult | EvaluationFailure:
    # Keep both outcomes (including executor framing) below Linux PIPE_BUF.
    # A worker killed during delivery cannot leave a partial result frame.
    try:
        return _worker_runner.run(task)
    except BaseException as exc:
        message = f"{type(exc).__name__}: {exc}"
        return EvaluationFailure(task, message.encode("utf-8")[:1024].decode("utf-8", errors="ignore"))


def _run_evaluation(config, overrides, sweep_index, started, threads, slots, coordinator_cpu, check_stop):
    configure_cpu_runtime(config.runtime, config.data)
    from .execution import _package_source_sha256, _resume_environment

    settings = config.evaluation
    destination = _run_path(config, sweep_index)
    if not settings.resume:
        destination.mkdir()
        _sync_directory(destination.parent)
    elif not destination.is_dir() or destination.is_symlink():
        raise ArtifactError("evaluation resume requires the original run directory")

    with exclusive_lock(destination / ".evaluation.lock", "evaluation"):
        contract_path = destination / "contract.json"
        final_path = destination / "manifest.json"
        shards_dir = destination / "shards"
        if shards_dir.is_symlink():
            raise ArtifactError("evaluation shard root must be an owned directory")
        _clean_evaluation_json_temps(destination)
        saved = read_json(contract_path) if contract_path.exists() else None
        if saved is not None and saved.get("schema_version") != EVALUATION_CONTRACT_SCHEMA:
            raise ArtifactError("evaluation resume contract differs or has a legacy schema; use a new run name for legacy results")
        if saved is None and (
            final_path.exists() or {p.name for p in destination.iterdir()} - {".evaluation.lock", "shards"}
            or (shards_dir.exists() and any(shards_dir.iterdir()))
        ):
            raise ArtifactError("evaluation run lacks a valid resumable contract; use a new run name for legacy results")

        metadata_start = perf_counter()
        data_root, mapped = _dataset(config)
        checkpoint_start = perf_counter()
        modes = ("G", "F", "S") if settings.id == "paired" else (settings.id,)
        model_payload, checkpoint, conditions, buffers, dem = _evaluation_inputs(config, mapped, data_root, modes)
        if slots:
            model_payload = dem = None
        environment = {**_environment_record(), "package_source_sha256": _package_source_sha256()}
        numerical_environment = _resume_environment(environment)
        numerical_environment["cpu"] = {key: environment["cpu"][key] for key in ("model", "platform")}
        for package in ("scipy", "pymatching") if settings.matching else ("scipy",):
            numerical_environment["packages"][package] = environment["packages"][package]
        runtime = _evaluation_runtime(config, threads)
        resolved = config.to_record()
        semantic = {
            "prediction_schema": PREDICTION_SCHEMA, "data": _dataset_record(mapped), "conditions": conditions,
            "request": _evaluation_request(resolved), "checkpoint": checkpoint, "runtime": runtime,
            "environment": numerical_environment,
        }
        if saved is not None and saved.get("semantic") != semantic:
            raise ArtifactError("evaluation resume contract differs from the requested data, model, policy, or environment")
        if saved is None:
            saved = {
                "schema_version": EVALUATION_CONTRACT_SCHEMA, "semantic": semantic, "resolved_config": resolved,
                "overrides": list(overrides), "environment": environment,
            }
            write_json(contract_path, saved)
        shards_dir.mkdir(exist_ok=True)
        _sync_directory(destination)

        staging_root = destination / ".evaluation-staging"
        _clean_staging(staging_root)
        sealed = _read_manifest(final_path, mapped) if final_path.exists() else None
        if sealed is not None and sealed.get("contract") != semantic:
            raise ArtifactError("completed evaluation manifest differs from the resumable contract")
        committed = _evaluation_inventory(shards_dir, len(mapped.shards))
        if sealed is not None and len(committed) != len(mapped.shards):
            raise ArtifactError("completed evaluation is missing committed shards")

        digest = _semantic_digest(semantic)
        attempt = uuid4().hex
        log_dir = destination / "attempts" / attempt
        log_dir.mkdir(parents=True)
        log = log_dir / "coordinator.jsonl"
        limit = (settings.max_in_flight or 2 * settings.workers) if slots else 1
        append_event(
            log, "attempt_start", attempt=attempt, contract_sha256=digest, process=observe_process(),
            resume=settings.resume, workers=settings.workers, worker_threads=threads, max_in_flight=limit,
            worker_affinity=slots, coordinator_commit_cpu=coordinator_cpu, allocated_cpus=config.runtime.allocated_cpus,
            committed=sorted(committed), source_checkpoint=str(config.checkpoint.path),
            source_dataset=str(data_root), timings={
                "setup": metadata_start - started, "dataset_metadata": checkpoint_start - metadata_start,
                "checkpoint_queries_and_contract": perf_counter() - checkpoint_start,
            },
        )
        status = "interrupted_or_failed"
        try:
            inventory = {}
            verify_started = perf_counter()
            for index in sorted(committed):
                check_stop()
                descriptor = mapped.shards[index]
                with mapped.open_shard(index) as shard:
                    entry = _verify_commit(shards_dir / str(index), descriptor, shard, conditions, digest)
                if sealed is not None and sealed["shards"][index] != entry:
                    raise ArtifactError("completed evaluation inventory differs from its shard receipt")
                inventory[index] = entry
            append_event(log, "verified_commits", count=len(inventory), wall_seconds=perf_counter() - verify_started)
            if len(inventory) < len(mapped.shards):
                with _task_staging(staging_root, attempt) as stage:
                    inline_start = perf_counter()
                    runner = None if slots else _build_task_runner(
                        config, mapped, model_payload, buffers, dem, modes, conditions, digest, stage, log,
                    )
                    del model_payload
                    if runner is not None:
                        append_event(
                            log, "inline_model_ready", model_loads=1, process=observe_process(),
                            wall_seconds=perf_counter() - inline_start,
                        )
                    _evaluate_missing(
                        config, mapped, modes, semantic, digest, inventory, stage, log_dir, shards_dir,
                        slots, coordinator_cpu, threads, limit, runner, check_stop,
                    )
            check_stop()
            if sealed is None:
                if set(inventory) != set(range(len(mapped.shards))):
                    raise ArtifactError("evaluation did not commit exact shard coverage")
                publication = perf_counter()
                _sync_directory(shards_dir)
                write_json(final_path, {
                    "schema_version": PREDICTION_SCHEMA, "status": "complete", "data": semantic["data"],
                    "conditions": conditions, "shards": [inventory[index] for index in range(len(mapped.shards))],
                    "resolved_config": saved["resolved_config"], "overrides": saved["overrides"],
                    "environment": saved["environment"], "runtime": runtime,
                    "contract": semantic, "contract_sha256": digest,
                })
                append_event(log, "manifest_published", wall_seconds=perf_counter() - publication)
            status = "validated_complete" if sealed is not None else "complete"
        finally:
            append_event(log, "attempt_end", status=status, wall_seconds=perf_counter() - started, process=observe_process())
    return final_path


def _evaluate_missing(
    config, mapped, modes, semantic, digest, inventory, stage, log_dir, shards_dir,
    slots, coordinator_cpu, threads, limit, runner, check_stop,
):
    log = log_dir / "coordinator.jsonl"
    tasks = (
        EvaluationTask(desc.index, stage.name, digest) for desc in mapped.shards if desc.index not in inventory
    )

    def submitted(task, pending):
        append_event(log, "task_submitted", index=task.index, pending=pending)

    def commit(task, result, submitted_at, pending):
        check_stop()
        begin = perf_counter()
        if isinstance(result, EvaluationFailure) and result.task == task:
            raise ArtifactError(f"evaluation task {task} failed: {result.error}")
        if (not isinstance(result, EvaluationResult) or result.task != task
                or task.index in inventory or type(result.receipt) is not dict):
            raise ArtifactError(f"invalid receipt or duplicate commit for evaluation task {task}")
        directory = stage / str(task.index)
        descriptor = mapped.shards[task.index]
        with cpu_affinity({coordinator_cpu}) if slots else nullcontext():
            with mapped.open_shard(task.index, labels_only=True) as shard:
                entry = _verify_commit(directory, descriptor, shard, semantic["conditions"], digest, result.receipt)
            output_bytes = result.receipt["bytes"] + sum(file["bytes"] for file in entry["files"].values())
            if (entry["rows"], 1 + len(entry["files"]), output_bytes) != (
                result.rows, result.output_files, result.output_bytes,
            ):
                raise ArtifactError(f"worker receipt differs from staged evaluation shard {task.index}")
            verified = perf_counter()
            publish_directory(directory, shards_dir / str(task.index))
            _sync_directory(stage)
        inventory[task.index] = entry
        append_event(
            log, "task_committed", index=task.index, pending=pending - 1, worker_pid=result.worker_pid,
            queue_seconds=result.started - submitted_at, verification_seconds=verified - begin,
            publication_seconds=perf_counter() - verified,
        )

    if slots:
        pool_start = perf_counter()
        try:
            run_pool(
                tasks, slots=slots, threads=threads, limit=limit, lease=stage / "writers.lock", log_dir=log_dir,
                initialize=_initialize_evaluation_worker, initializer_args=(config, semantic, modes, threads, stage, log_dir),
                work=_run_worker_task, check_stop=check_stop, on_submit=submitted, on_result=commit,
            )
        finally:
            append_event(log, "pool_closed", wall_seconds=perf_counter() - pool_start, process=observe_process())
    else:
        for task in tasks:
            check_stop()
            submitted_at = perf_counter()
            submitted(task, 1)
            commit(task, runner.run(task, check_stop=check_stop), submitted_at, 1)


@contextmanager
def _task_staging(root: Path, attempt: str):
    root.mkdir(exist_ok=True)
    stage = root / attempt
    stage.mkdir()
    (stage / "writers.lock").touch(exist_ok=False)
    try:
        yield stage
    finally:
        _retire_staging(stage)
        root.rmdir()


def _retire_staging(stage: Path) -> None:
    lease = stage / "writers.lock"
    if lease.is_symlink() or (lease.exists() and not lease.is_file()):
        raise ArtifactError("invalid evaluation worker lease")
    if not lease.exists():
        # A coordinator can die before creating the lease or while retiring
        # an already quiescent attempt. No worker opens the lease with O_CREAT.
        shutil.rmtree(stage)
        return
    with lease.open("rb") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ArtifactError(f"evaluation attempt {stage.name} still has live writers; resume after they exit") from exc
        shutil.rmtree(stage)


def _clean_staging(root: Path) -> None:
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ArtifactError("invalid evaluation staging root")
    if root.is_dir():
        for stage in root.iterdir():
            if not re.fullmatch(r"[a-f0-9]{32}", stage.name) or stage.is_symlink() or not stage.is_dir():
                raise ArtifactError("invalid evaluation staging attempt")
            _retire_staging(stage)
        root.rmdir()


def _semantic_digest(record: dict) -> str:
    return sha256(canonical_json(record)).hexdigest()


def _evaluation_request(resolved: dict, *, schema: int = PREDICTION_SCHEMA) -> dict:
    runtime = resolved["runtime"]
    settings = resolved["evaluation"]
    omitted = {"resume"}
    if schema >= 3:
        threads = settings.get("worker_threads")
        if threads is None:
            threads = runtime["intraop_threads"] if settings.get("workers", 1) == 1 else 1
        runtime = {
            "device": runtime["device"], "dtype": runtime["dtype"], "intraop_threads": threads,
            "interop_threads": runtime["interop_threads"],
        }
        omitted.update(("workers", "worker_threads", "max_in_flight"))
    return {
        **{key: resolved[key] for key in ("code", "noise", "window", "model", "geometry")},
        "runtime": runtime,
        "execution": {key: resolved["data"][key] for key in ("num_workers", "sampling_processes")},
        "evaluation": {key: value for key, value in settings.items() if key not in omitted},
    }


def _clean_evaluation_json_temps(directory: Path) -> None:
    for path in directory.iterdir():
        if re.fullmatch(r"\.(?:contract|manifest)\.json\.[a-z0-9_]{8}", path.name) and path.is_file():
            path.unlink()


def _evaluation_inventory(root: Path, count: int) -> set[int]:
    indices = set()
    for path in root.iterdir():
        if not path.is_dir() or path.is_symlink() or not re.fullmatch(r"0|[1-9][0-9]*", path.name):
            raise ArtifactError("unexpected evaluation shard directory")
        index = int(path.name)
        if index >= count:
            raise ArtifactError("out-of-range evaluation shard directory")
        indices.add(index)
    return indices


def _verify_commit(directory, descriptor, shard, conditions, digest, expected_receipt=None) -> dict:
    if directory.is_symlink() or (directory / "receipt.json").is_symlink():
        raise ArtifactError(f"invalid evaluation receipt location for shard {descriptor.index}")
    receipt, integrity = read_json_artifact(directory / "receipt.json")
    if expected_receipt is not None and integrity != expected_receipt:
        raise ArtifactError(f"worker receipt differs from staged evaluation shard {descriptor.index}")
    if (
        type(receipt.get("schema_version")) is not int or receipt["schema_version"] != 1
        or receipt.get("contract_sha256") != digest
        or receipt.get("source") != descriptor.source_record()
    ):
        raise ArtifactError(f"committed evaluation shard {descriptor.index} has incompatible source or run identity")
    entry = {key: value for key, value in receipt.items() if key not in ("schema_version", "contract_sha256")}
    _verify_prediction_files(directory, entry, shard, conditions)
    return entry


def _verify_prediction_files(directory, entry, shard, conditions) -> None:
    paths = list(directory.iterdir())
    expected = set(entry.get("files", {})) | {"receipt.json"}
    if {path.name for path in paths} != expected or any(path.is_symlink() or not path.is_file() for path in paths):
        raise ArtifactError(f"missing or unexpected prediction payloads for shard {shard.index}")
    for _ in _read_prediction_shard(directory, entry, shard, conditions):
        pass


def _evaluate_shard(stage, descriptor, shard, evaluator, modes, matching, batch_size, digest, timings, check_stop) -> dict:
    import numpy as np

    from .evaluation import compose_core_decisions

    stage.mkdir()
    count = shard.rows
    logits = {mode: np.empty((count, len(evaluator.views(mode))), dtype=np.float32) for mode in modes}
    baselines = np.empty((count, 1), dtype=bool) if matching is not None else None
    for start in range(0, count, batch_size):
        check_stop()
        rows = slice(start, min(count, start + batch_size))
        begin = perf_counter()
        shots = shard.read(rows).shots
        timings["source_read_unpack"] += perf_counter() - begin
        for mode in modes:
            logits[mode][rows] = evaluator.logits(mode, shots.detectors, shots.shot_ids, timings=timings)
        if matching is not None:
            begin = perf_counter()
            baselines[rows] = matching.decisions(shots.detectors)
            timings["forward_MWPM"] += perf_counter() - begin
    check_stop()
    begin = perf_counter()
    files = {"shot_ids.npy": _write_array(stage / "shot_ids.npy", shard.shot_ids)}
    decisions = {mode: scores >= 0 for mode, scores in logits.items()}
    if matching is not None:
        decisions["MWPM"] = baselines
    for mode, bits in decisions.items():
        files[f"{mode}.decisions.npy"] = {
            **_write_array(stage / f"{mode}.decisions.npy", np.packbits(bits, axis=1, bitorder="little")),
            "valid_bits": bits.shape[1],
        }
        if mode in logits:
            files[f"{mode}.logits.npy"] = _write_array(stage / f"{mode}.logits.npy", logits[mode])
        if mode in ("F", "S"):
            final = compose_core_decisions(bits).astype(np.uint8)[:, None]
            files[f"{mode}.final.npy"] = {**_write_array(stage / f"{mode}.final.npy", final), "valid_bits": 1}
    entry = {"index": descriptor.index, "rows": count, "files": files, "source": descriptor.source_record()}
    write_json(stage / "receipt.json", {"schema_version": 1, "contract_sha256": digest, **entry})
    timings["prediction_serialize_write"] += perf_counter() - begin
    return entry


def _read_manifest(path: Path, mapped, *, verify_queries: bool = True) -> dict:
    return _validate_manifest(read_json(path), mapped, verify_queries=verify_queries)


def _validate_manifest(record: dict, mapped, *, verify_queries: bool = True) -> dict:
    from .continuation import is_digest
    from .evaluation import CheckpointProvenance
    from .windows import Buffers, build_observation_layout, select_window

    schema = record.get("schema_version")
    if (type(schema) is not int or schema not in (1, 2, PREDICTION_SCHEMA)
            or record.get("status") != "complete"):
        raise ArtifactError("incomplete or incompatible prediction manifest")
    if record.get("data") != _dataset_record(mapped):
        raise ArtifactError("prediction dataset identity or label/feature schema differs from the selected dataset")
    try:
        if len(record["shards"]) != len(mapped.shards) or not record["conditions"]:
            raise ArtifactError("prediction manifest is missing shards or conditions")
        if schema == 1 and {"contract", "contract_sha256"} & record.keys():
            raise ArtifactError("incompatible prediction manifest mixes legacy and resumable schemas")
        if schema >= 2:
            semantic = record["contract"]
            if (
                record["contract_sha256"] != _semantic_digest(semantic)
                or semantic["prediction_schema"] != schema
                or any(semantic[key] != record[key] for key in ("data", "conditions", "runtime"))
                or semantic["request"] != _evaluation_request(record["resolved_config"], schema=schema)
            ):
                raise ArtifactError("prediction manifest has an incompatible semantic contract")
            for descriptor, entry in zip(mapped.shards, record["shards"], strict=True):
                if (
                    type(entry["index"]) is not int or type(entry["rows"]) is not int
                    or (descriptor.index, descriptor.rows) != (entry["index"], entry["rows"])
                    or entry["source"] != descriptor.source_record()
                ):
                    raise ArtifactError("prediction manifest has incompatible source shard identity or coverage")
        resolved = record["resolved_config"]
        settings = resolved["evaluation"]
        expected_modes = set("GFS") if settings["id"] == "paired" else {settings["id"]}
        if settings["matching"]:
            expected_modes.add("MWPM")
        if (
            record["conditions"].keys() != expected_modes
            or EVALUATION_PRESETS.get(settings["id"]) != (settings["query"], settings["context"], settings["aggregation"])
            or settings["threshold"] != 0.5 or not settings["reset_recurrent_state_per_query"]
            or settings["update_model_state"]
        ):
            raise ArtifactError("prediction inference settings differ from the frozen G/F/S policy")
        geometry = resolved["geometry"]
        layout = build_observation_layout(mapped.bundle) if verify_queries else None
        buffers = Buffers(geometry["left_buffer"], geometry["right_buffer"])
        for mode, entry in record["conditions"].items():
            if entry["mode"] != mode:
                raise ArtifactError("prediction inference mode differs from its manifest identity")
            if mode == "MWPM":
                baseline = entry["baseline"]
                if (
                    entry["condition"] != "MWPM" or entry["checkpoint"] is not None or entry["queries"]
                    or entry["model_state_sha256"] is not None or baseline["implementation"] != "pymatching"
                    or baseline["num_observables"] != 1 or baseline["num_detectors"] != mapped.plan.detector_count
                ):
                    raise ArtifactError("invalid matching baseline provenance")
                continue
            provenance = CheckpointProvenance(**entry["checkpoint"])
            indices = [None] if mode == "G" else list(range(mapped.plan.core_count))
            if (
                provenance.condition(mode) != entry["condition"] or not is_digest(entry["model_state_sha256"])
                or [query["core_index"] for query in entry["queries"]] != indices
            ):
                raise ArtifactError("prediction condition, query partition or checkpoint provenance differs")
            if layout is not None:
                targets = (layout.global_query,) if mode == "G" else layout.queries
                queries = [_query_record(select_window(layout, target, buffers if mode == "S" else None)) for target in targets]
                if entry["queries"] != queries:
                    raise ArtifactError("prediction condition, query partition or checkpoint provenance differs")
        return record
    except ArtifactError:
        raise
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise ArtifactError("invalid prediction manifest contents") from exc


def _read_array(path: Path, entry: dict, *, dtype: str, shape: tuple):
    from .data import ArrayDescriptor, _map_array

    if entry["dtype"] != dtype or entry["shape"] != list(shape):
        raise ArtifactError(f"prediction array dimensions or dtype differ for {path.name}")
    spec = ArrayDescriptor(path.name, path, shape, dtype, None, entry["bytes"], entry["sha256"])
    with _map_array(spec) as array:
        return array.copy()


def _read_shard(path: Path, record: dict, shard):
    entry = record["shards"][shard.index]
    yield from _read_prediction_shard(path.parent / "shards" / str(shard.index), entry, shard, record["conditions"])


def _read_prediction_shard(directory, entry, shard, conditions):
    import numpy as np

    from .analysis import require_paired_ids
    from .evaluation import compose_core_decisions

    try:
        row_count = shard.rows
        if (
            type(entry["index"]) is not int or type(entry["rows"]) is not int
            or (entry["index"], entry["rows"]) != (shard.index, row_count)
        ):
            raise ArtifactError("missing, duplicated or reordered prediction shards")
        files = entry["files"]
        expected = {"shot_ids.npy"}
        for mode in conditions:
            expected.add(f"{mode}.decisions.npy")
            if mode != "MWPM":
                expected.add(f"{mode}.logits.npy")
            if mode in ("F", "S"):
                expected.add(f"{mode}.final.npy")
        if set(files) != expected:
            raise ArtifactError("prediction inventory differs from the declared modes")
        ids = _read_array(directory / "shot_ids.npy", files["shot_ids.npy"], dtype="int64", shape=(row_count,))
        require_paired_ids(shard.shot_ids, ids)
        for mode, condition in conditions.items():
            width = shard.plan.core_count if mode in ("F", "S") else 1
            name = f"{mode}.decisions.npy"
            packed = _read_array(directory / name, files[name], dtype="uint8", shape=(row_count, (width + 7) // 8))
            if files[name]["valid_bits"] != width or (width % 8 and np.any(packed[:, -1] >> (width % 8))):
                raise ArtifactError("prediction valid bits or padding differ")
            decisions = np.unpackbits(packed, count=width, axis=1, bitorder="little").astype(bool)
            scores = None
            if mode != "MWPM":
                name = f"{mode}.logits.npy"
                scores = _read_array(directory / name, files[name], dtype="float32", shape=(row_count, width))
                if not np.isfinite(scores).all() or not np.array_equal(decisions, scores >= 0):
                    raise ArtifactError("prediction decisions differ from the recorded finite logits and threshold")
            final = decisions[:, 0]
            if mode in ("F", "S"):
                name = f"{mode}.final.npy"
                composed = _read_array(directory / name, files[name], dtype="uint8", shape=(row_count, 1))
                final = compose_core_decisions(decisions)
                if files[name]["valid_bits"] != 1 or not np.array_equal(final, composed[:, 0]):
                    raise ArtifactError("final predictions disagree with the recorded core-decision XOR")
            yield condition["condition"], decisions, scores, final
    except ArtifactError:
        raise
    except (ValueError, KeyError, TypeError, EOFError, OSError) as exc:
        raise ArtifactError(f"invalid prediction rows in shard {shard.index}: {exc}") from exc


def _check_comparisons(conditions: dict, names: list[str]) -> list[tuple[str, str, str]]:
    requested = []
    namespaces = {
        (cp["training_seed"], cp["training_replicate"], cp["training_family"])
        for item in conditions.values() if (cp := item["checkpoint"]) is not None
    }
    if len(namespaces) > 1:
        raise ArtifactError("analyze one training seed and replicate at a time")
    primary = {
        (conditions[mode]["checkpoint"]["sha256"], conditions[mode]["model_state_sha256"])
        for mode in ("G", "F", "S") if mode in conditions
    }
    if len(primary) > 1:
        raise ArtifactError("primary G/F/S contrasts require the same checkpoint and model state")
    for name in names:
        parts = name.split("_minus_")
        if len(parts) != 2:
            raise ConfigError(f"invalid contrast {name!r}; use A_minus_B")
        first, second = ("S+" if part == "Splus" else part for part in parts)
        if first == second or first not in conditions or second not in conditions:
            raise ArtifactError(f"contrast {name} requires two distinct recorded conditions")
        left = conditions[first]["checkpoint"]
        right = conditions[second]["checkpoint"]
        if {first, second} <= {"S+", "W", "D"} and left["parent_sha256"] != right["parent_sha256"]:
            raise ArtifactError("adapted comparisons require a common base checkpoint")
        if conditions[first]["mode"] == conditions[second]["mode"] == "S" and (
            conditions[first]["queries"] != conditions[second]["queries"]
        ):
            raise ArtifactError("window comparisons require identical deployment queries and buffers")
        if {first, second} <= {"S+", "W", "D"}:
            for field in ("initial_model_sha256", "continuation_controls_sha256", "continuation_input_sha256"):
                if left.get(field) is None or left[field] != right.get(field):
                    raise ArtifactError("adapted comparisons require matched initialization, controls, and trajectory/core draws")
        requested.append((name, first, second))
    return requested


def _write_csv(path: Path, rows: list[dict]) -> None:
    contents = StringIO(newline="")
    writer = csv.DictWriter(contents, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    with atomic_file(path) as file:
        file.write(contents.getvalue().encode("utf-8"))


def summarize_predictions(mapped, inputs: list[tuple[Path, dict, dict]], policy, contrast_names: list[str]) -> dict:
    """Calculate P5 statistics given verified label and prediction receipts."""
    import numpy as np

    from .analysis import (
        CoreDiagnostics, PairedCounts, _count_pairs, binomial_interval, core_disagreements, summarize_pair,
    )
    shots = mapped.plan.total_rows
    conditions = {}
    for _, manifest, _ in inputs:
        for entry in manifest["conditions"].values():
            label = entry["condition"]
            if label in conditions:
                raise ArtifactError(f"duplicate prediction condition {label}")
            conditions[label] = entry
    comparisons = _check_comparisons(conditions, contrast_names)
    core = {label: CoreDiagnostics() for label, entry in conditions.items() if entry["mode"] in ("F", "S")}
    failures = dict.fromkeys(conditions, 0)
    losses = {label: np.zeros(12 if label in core else 1) for label, entry in conditions.items() if entry["mode"] != "MWPM"}
    paired = {name: np.zeros(4, dtype=np.int64) for name, _, _ in comparisons}
    for descriptor in mapped.shards:
        with mapped.open_shard(descriptor.index, labels_only=True) as shard:
            global_labels, core_labels = shard.read_labels()
            labels = global_labels[:, None]
            shard_failures = {}
            for path, manifest, _ in inputs:
                for condition, decisions, scores, final in _read_shard(path, manifest, shard):
                    target = labels
                    if condition in core:
                        core[condition].add(core_disagreements(decisions, core_labels, labels[:, 0]))
                        target = core_labels
                    error = final ^ labels[:, 0]
                    shard_failures[condition] = error
                    failures[condition] += int(error.sum())
                    if scores is not None:
                        signed = np.where(target, -scores, scores).astype(np.float64)
                        losses[condition] += np.logaddexp(0, signed).sum(axis=0)
            for name, first, second in comparisons:
                paired[name] += _count_pairs(shard_failures[first], shard_failures[second]).categories

    for path, _, receipt in inputs:
        verify_file(path, receipt)
    rates = {}
    for label, count in failures.items():
        rates[label] = {
            "shots": shots, "failures": count, "ler": count / shots,
            "wilson_interval": list(binomial_interval(count, shots, confidence=0.95, method="wilson")),
        }
        if label in losses:
            rates[label]["mean_query_cross_entropy"] = (losses[label] / shots).tolist()
        if label in core:
            rates[label]["core_diagnostics"] = core[label].report()
    rng = policy.rng()
    contrasts = {
        name: {"first": first, "second": second, **summarize_pair(PairedCounts(*map(int, paired[name])), policy, rng)}
        for name, first, second in comparisons
    }
    decomposition = None
    if set(PRIMARY_CONTRASTS) <= contrasts.keys():
        def net(name):
            row = contrasts[name]
            return row["n10"] - row["n01"]

        direct = net("S_minus_G")
        sum_of_parts = net("F_minus_G") + net("S_minus_F")
        if direct != sum_of_parts:
            raise ArtifactError("primary paired contrast decomposition disagrees at the count level")
        decomposition = {"direct_numerator": direct, "sum_of_parts_numerator": sum_of_parts, "denominator": shots}
    return {
        "conditions": conditions, "rates": rates, "contrasts": contrasts, "primary_decomposition": decomposition,
        "analysis": {**policy.record(), "contrast_order": contrast_names},
    }


def write_analysis_tables(root: Path, summary: dict) -> None:
    rates, contrasts = summary["rates"], summary["contrasts"]
    _write_csv(root / "rates.csv", [
        {"condition": label, "shots": row["shots"], "failures": row["failures"], "ler": row["ler"],
         "lower": row["wilson_interval"][0], "upper": row["wilson_interval"][1]} for label, row in rates.items()
    ])
    _write_csv(root / "contrasts.csv", [
        {"contrast": name, "n10": row["n10"], "n01": row["n01"], "difference": row["difference"],
         "lower": row["interval"][0], "upper": row["interval"][1], "interval_method": row["interval_method"]}
        for name, row in contrasts.items()
    ])


def analyze_workflow(
    config: ResolvedConfig, *, overrides: tuple[str, ...] = (), sweep_index: int | None = None,
) -> Path:
    check_paired_execution("analyze", config)
    configure_cpu_runtime(config.runtime, config.data)
    from .analysis import AnalysisPolicy

    _, mapped = _dataset(config)
    settings = config.analysis
    policy = AnalysisPolicy(
        settings.bootstrap_replicates, settings.bootstrap_seed, settings.sparse_rule,
        settings.ratio_min_denominator_failures,
    )
    inputs = []
    for name in settings.predictions:
        path = Path(name)
        record, receipt = read_json_artifact(path)
        inputs.append((path, _validate_manifest(record, mapped), receipt))
    summary = summarize_predictions(mapped, inputs, policy, settings.contrasts)
    destination = _run_path(config, sweep_index)
    with staged_directory(destination) as staging:
        write_analysis_tables(staging, summary)
        write_json(staging / "report.json", {
            "schema_version": 1, "status": "complete", "data": _dataset_record(mapped), **summary,
            "inputs": [{"path": str(path), **receipt} for path, _, receipt in inputs],
            "resolved_config": config.to_record(), "overrides": list(overrides), "environment": _environment_record(),
        })
    return destination / "report.json"
