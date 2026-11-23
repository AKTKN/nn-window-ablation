"""Deterministic shard tasks and all-or-nothing saved-dataset publication."""

from contextlib import closing, contextmanager, nullcontext
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from time import perf_counter
from uuid import uuid4

import stim

from .artifacts import (
    ArtifactError, _sync_directory, create_writer_lease, exclusive_lock, publish_directory, retire_staging,
)
from .config import ConfigError, ResolvedConfig
from .data import (
    ARRAY_FILES, OpenShard, SamplingPlan, _describe_shard, _write_dataset_manifest, _write_physical,
    _write_shard, sample_chunk,
)
from .evaluation_pool import append_event, cpu_affinity, observe_process, run_pool, stop_on_signal
from .runtime import NATIVE_THREAD_VARIABLES, _cpu_quota
from .simulation import CircuitBundle, build_circuit_bundle
from .timing import capture_spans
from .workflows import _environment_record


@dataclass(frozen=True, slots=True)
class GenerationTask:
    index: int
    dataset_id: str
    start: int
    rows: int
    seed: int
    attempt: str


@dataclass(frozen=True, slots=True)
class GenerationResult:
    task: GenerationTask
    manifest: dict
    pid: int
    started: float


@dataclass(frozen=True, slots=True)
class GenerationFailure:
    task: GenerationTask
    error: str


def _task(plan: SamplingPlan, index: int, attempt: str) -> GenerationTask:
    start, rows = plan.chunk_bounds(index)
    return GenerationTask(index, plan.dataset_id, start, rows, plan.namespace.seed("circuit", index), attempt)


@dataclass(slots=True)
class _Sampler:
    plan: SamplingPlan
    bundle: CircuitBundle
    shards: Path
    attempt: str
    log: Path

    def run(self, task: GenerationTask) -> GenerationResult:
        if _task(self.plan, task.index, self.attempt) != task:
            raise ArtifactError(f"generation task {task.index} has incompatible plan, interval, seed or attempt")
        started = perf_counter()
        append_event(self.log, "task_start", index=task.index, shots=task.rows, process=observe_process())
        with capture_spans() as timings:
            shots = sample_chunk(self.plan, self.bundle, task.index)
            receipt = _write_shard(self.shards / str(task.index), self.plan, task.index, shots)
            del shots
        append_event(
            self.log, "task_done", index=task.index, shots=task.rows,
            timings=timings, wall_seconds=perf_counter() - started, process=observe_process(),
        )
        return GenerationResult(task, receipt["manifest"], os.getpid(), started)


_sampler: _Sampler | None = None


def _initialize_sampler(recipe, backend, shards, attempt, log_dir):
    global _sampler
    started = perf_counter()
    if len(os.sched_getaffinity(0)) != 1 or any(os.environ.get(name) != "1" for name in NATIVE_THREAD_VARIABLES):
        raise ArtifactError("generation worker must have one CPU and one native thread")
    plan = SamplingPlan.from_record(recipe)
    bundle = build_circuit_bundle(plan.spec)
    plan.assert_circuit(bundle)
    if stim.Circuit.__module__ != backend:
        raise ArtifactError("generation worker has a different Stim SIMD backend")
    _sampler = _Sampler(plan, bundle, shards, attempt, log_dir / f"worker-{os.getpid()}.jsonl")
    return {
        "circuit_builds": 1, "circuit_seconds": perf_counter() - started, "stim_backend": backend,
        "native_threads": 1, "native_env": {name: os.environ[name] for name in NATIVE_THREAD_VARIABLES},
    }


def _run_sampler(task: GenerationTask) -> GenerationResult | GenerationFailure:
    try:
        return _sampler.run(task)
    except BaseException as exc:
        # Keep result-queue frames atomic if a sampler is terminated during delivery.
        error = f"{type(exc).__name__}: {exc}".encode()[:768].decode(errors="ignore")
        return GenerationFailure(task, error)


def _verify_result(shards, plan, task, result):
    if isinstance(result, GenerationFailure) and result.task == task:
        raise ArtifactError(f"generation task {task.index} failed: {result.error}")
    if (
        not isinstance(result, GenerationResult) or result.task != task or type(result.manifest) is not dict
        or task != _task(plan, task.index, task.attempt)
    ):
        raise ArtifactError(f"invalid generation receipt for task {task.index}")
    # The coordinator derives all paths from its own plan and attempt.
    directory = shards / str(task.index)
    if (
        shards.is_symlink() or directory.is_symlink() or not directory.is_dir()
        or directory.resolve().parent != shards
    ):
        raise ArtifactError(f"invalid staged generation path for shard {task.index}")
    files = list(directory.iterdir())
    if (
        {item.name for item in files} != {"shard.json", *ARRAY_FILES.values()}
        or any(item.is_symlink() or not item.is_file() for item in files)
    ):
        raise ArtifactError(f"invalid staged file inventory for generation shard {task.index}")
    entry = {"index": task.index, "manifest": result.manifest}
    try:
        descriptor = _describe_shard(directory, plan, task.index, entry)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ArtifactError(f"malformed staged manifest for generation shard {task.index}") from exc
    with closing(OpenShard(descriptor, plan, labels_only=False)) as shard:
        mappings = shard.mapped_arrays
    return entry, mappings, descriptor.manifest_bytes + sum(array.bytes for array in descriptor.arrays)


def _generation_slots(config: ResolvedConfig):
    workers = config.data.sampling_processes
    if not hasattr(os, "sched_getaffinity") or not sys.platform.startswith("linux"):
        if workers > 1:
            raise ConfigError("parallel generation requires Linux process affinity and parent-death signals")
        return (), None
    affinity = sorted(os.sched_getaffinity(0))
    allocated = config.runtime.allocated_cpus
    if allocated > len(affinity) or workers > allocated:
        raise ConfigError("generation allocation does not fit observed process affinity")
    quota = _cpu_quota()
    if quota is not None and workers > quota:
        raise ConfigError(f"generation requires {workers} CPUs, exceeding visible cgroup CPU quota {quota:g}")
    coordinator = affinity[workers] if allocated > workers else affinity[0]
    return tuple((cpu,) for cpu in affinity[:workers]), coordinator


@contextmanager
def _owned_staging(directory: Path, log_dir: Path):
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise ArtifactError("invalid generation staging root")
    if directory.exists():
        if {item.name for item in directory.iterdir()} - {"writers.lock", "dataset"}:
            raise ArtifactError("unexpected data in the generation staging root")
        retire_staging(directory, "generation", log_dir / "previous-writers.lock")
    directory.mkdir()
    try:
        create_writer_lease(directory, log_dir.name)
        yield directory
    except BaseException as error:
        try:
            retire_staging(directory, "generation", log_dir / "writers.lock")
        except (ArtifactError, OSError) as cleanup:
            error.add_note(f"generation staging cleanup: {cleanup}")
        raise
    else:
        retire_staging(directory, "generation", log_dir / "writers.lock")


def generate_saved_dataset(config: ResolvedConfig, overrides, started: float) -> Path:
    from .execution import _circuit_and_namespace, _package_source_sha256

    slots, coordinator = _generation_slots(config)
    workers = config.data.sampling_processes
    limit = (config.data.sampling_max_in_flight or 2 * workers) if workers > 1 else 1
    prepare = perf_counter()
    bundle, namespace = _circuit_and_namespace(config)
    plan = SamplingPlan.from_circuit(bundle, namespace, chunk_rows=config.data.shard_shots, total_rows=config.data.shots)
    circuit_seconds = perf_counter() - prepare
    root = config.paths.data_root
    path = root / (config.data.id or plan.dataset_id)
    root.mkdir(parents=True, exist_ok=True)
    attempt = uuid4().hex
    with stop_on_signal("generation") as check_stop, exclusive_lock(root / f".{path.name}.generation.lock", "generation"):
        if os.path.lexists(path):
            raise FileExistsError(path)
        log_dir = root / ".generation-attempts" / path.name / attempt
        log_dir.mkdir(parents=True)
        log = log_dir / "coordinator.jsonl"
        runtime = {
            "sampling_processes": workers, "sampling_threads": 1, "data_loader_workers": 0,
            "max_in_flight": limit, "worker_affinities": slots if workers > 1 else (),
            "stim_backend": stim.Circuit.__module__, "allocated_cpus": config.runtime.allocated_cpus,
        }
        append_event(
            log, "attempt_start", attempt=attempt, dataset_id=plan.dataset_id, shots=plan.total_rows,
            shard_rows=plan.chunk_rows, circuit_seconds=circuit_seconds, runtime=runtime, process=observe_process(),
        )
        complete = False
        try:
            with _owned_staging(root / f".{path.name}.generation", log_dir) as stage:
                worktree = stage / "dataset"
                shards = worktree / "shards"
                shards.mkdir(parents=True)
                physical_start = perf_counter()
                physical = _write_physical(worktree, bundle)
                append_event(log, "physical_written", wall_seconds=perf_counter() - physical_start)
                count = (plan.total_rows + plan.chunk_rows - 1) // plan.chunk_rows
                inventory = _generate_shards(
                    plan, bundle, attempt, shards, stage / "writers.lock", log_dir, count, slots if workers > 1 else (),
                    coordinator, limit, check_stop,
                )
                check_stop()
                publication = perf_counter()
                provenance = {
                    "command": "generate", "resolved_config": config.to_record(), "overrides": list(overrides),
                    "environment": {**_environment_record(), "package_source_sha256": _package_source_sha256()},
                    "runtime": runtime, "attempt": attempt,
                }
                _write_dataset_manifest(worktree, plan, physical, inventory, provenance)
                _sync_directory(shards)
                publish_directory(worktree, path)
                complete = True
                append_event(log, "manifest_published", wall_seconds=perf_counter() - publication)
        finally:
            append_event(
                log, "attempt_end", status="complete" if complete else "interrupted_or_failed",
                wall_seconds=perf_counter() - started, process=observe_process(),
            )
    return path / "manifest.json"


def _generate_shards(plan, bundle, attempt, shards, lease, log_dir, count, slots, coordinator, limit, check_stop):
    log = log_dir / "coordinator.jsonl"
    inventory = {}

    def submitted(task, pending):
        append_event(log, "task_submitted", index=task.index, pending=pending)

    def verified(task, result, submitted_at, pending):
        check_stop()
        if task.index in inventory:
            raise ArtifactError(f"duplicate generation receipt for shard {task.index}")
        verification = perf_counter()
        with cpu_affinity({coordinator}) if slots else nullcontext():
            record, mappings, output_bytes = _verify_result(shards, plan, task, result)
        inventory[task.index] = record
        append_event(
            log, "shard_verified", index=task.index, pending=pending - 1, worker_pid=result.pid,
            queue_seconds=result.started - submitted_at, verification_seconds=perf_counter() - verification,
            active_mappings=mappings, output_bytes=output_bytes, output_files=1 + len(ARRAY_FILES),
            process=observe_process(),
        )

    tasks = (_task(plan, index, attempt) for index in range(count))
    if slots:
        with cpu_affinity({coordinator, *(cpu for group in slots for cpu in group)}):
            run_pool(
                tasks, slots=slots, threads=1, limit=limit, lease=lease, log_dir=log_dir,
                initialize=_initialize_sampler, initializer_args=(plan.record(), stim.Circuit.__module__, shards, attempt, log_dir),
                work=_run_sampler, check_stop=check_stop, on_submit=submitted, on_result=verified, operation="generation",
            )
    else:
        sampler = _Sampler(plan, bundle, shards, attempt, log)
        for task in tasks:
            check_stop()
            submitted_at = perf_counter()
            submitted(task, 1)
            verified(task, sampler.run(task), submitted_at, 1)
    if len(inventory) != count or {item.name for item in shards.iterdir()} != {str(index) for index in range(count)}:
        raise ArtifactError("generation did not verify exact shard coverage")
    return [inventory[index] for index in range(count)]


def dataset_semantics(manifest: dict) -> dict:
    """Project a sealed dataset manifest onto immutable scientific content."""
    return {key: manifest[key] for key in ("schema_version", "status", "dataset_id", "recipe", "physical", "shards")}
