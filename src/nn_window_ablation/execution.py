"""P4 dataset and training workflows over resolved configuration and explicit paths."""

from contextlib import contextmanager
import fcntl
from hashlib import sha256
from pathlib import Path

from .artifacts import ArtifactError, file_integrity, write_json
from .config import ConfigError, ResolvedConfig, require_execution_config, validate_run_name
from .runtime import configure_cpu_runtime
from .workflows import _environment_record, reserve_run_directory


def check_execution(command: str, config: ResolvedConfig) -> None:
    """Check P4 configuration before reserving any artifact directories."""
    require_execution_config(config, command)
    data = config.data
    if (config.runtime.processes, data.num_workers, data.sampling_processes) != (1, 0, 1):
        raise ConfigError("P4 execution requires one application process, zero loader workers, and inline sampling")
    if data.split not in ("train", "validation", "test"):
        raise ConfigError("data.split must be train, validation, or test")
    if (data.storage, data.bitorder, data.label_schema) != ("packed_npy", "little", "clean_cut_core_parity_v1"):
        raise ConfigError("P4 requires the packed little-endian data and clean-cut core label schema")
    if data.id is not None:
        validate_run_name(data.id)
    if not 0 <= config.seed < 2**64 or not data.family_id.strip():
        raise ConfigError("P4 requires an unsigned 64-bit seed and a nonempty dataset family")
    training = config.training
    if data.split == "train" and training.data_stream_phase not in ("base", "continuation"):
        raise ConfigError("a training stream requires base or continuation phase")
    if command == "generate":
        return
    if data.split != "train":
        raise ConfigError("optimization requires data.split=train")
    if data.train_source not in ("generated_chunks", "stored_shards"):
        raise ConfigError("data.train_source must be generated_chunks or stored_shards")
    if training.optimizer not in ("sgd", "adam") or training.scheduler not in ("none", "step"):
        raise ConfigError("choose training.optimizer=sgd or adam and training.scheduler=none or step")
    if training.microbatch_size is not None and training.microbatch_size > training.batch_size:
        raise ConfigError("microbatch size exceeds the effective batch size")
    if config.checkpoint.resume_from is None and not training.reset_optimizer_at_start:
        raise ConfigError("a new training phase requires a fresh optimizer")
    if (
        training.global_context != "full_closed" or training.core_loss_weight != 1.0
        or training.loss_normalization != "one_plus_core_loss_weight"
        or training.sampled_cores_per_trajectory != 1 or training.core_sampling != "uniform"
        or training.checkpoint_policy != "fixed_updates"
    ):
        raise ConfigError("training must retain the fixed-budget global and sampled-core objective")
    if training.initialization == "random":
        if training.data_stream_phase != "base" or training.core_context != "full_closed" or config.checkpoint.init_from:
            raise ConfigError("fresh base training requires random initialization and full context")
    elif (
        training.initialization != "full_checkpoint" or training.data_stream_phase != "continuation"
        or training.core_context not in ("full_closed", "deployment_window", "sampled_window")
    ):
        raise ConfigError("continuation requires a base checkpoint and a supported core context")
    dynamic = training.dynamic_buffers
    if dynamic.enabled != (training.core_context == "sampled_window"):
        raise ConfigError("sampled-window context and dynamic buffer mode must agree")
    if dynamic.enabled and (
        dynamic.ratios != [0, 0.5, 1] or dynamic.distribution != "independent_uniform" or dynamic.sampling_unit != "minibatch"
    ):
        raise ConfigError("dynamic buffer draws require the prescribed uniform logical-minibatch policy")
    if config.checkpoint.path is not None:
        raise ConfigError("checkpoint.path selects evaluation weights; training uses init_from or resume_from")


def _circuit_and_namespace(config: ResolvedConfig):
    from .data import StreamNamespace
    from .simulation import CircuitSpec, build_circuit_bundle

    geometry = config.geometry
    bundle = build_circuit_bundle(CircuitSpec(
        geometry.distance, geometry.rounds, geometry.core_length, config.noise.probability,
    ))
    split = config.data.split
    phase = config.training.data_stream_phase if split == "train" else "held_out"
    namespace = StreamNamespace(config.seed, config.data.family_id, split, geometry.distance, config.data.replicate, phase)
    return bundle, namespace


def generate_workflow(config: ResolvedConfig, *, overrides: tuple[str, ...] = ()) -> Path:
    check_execution("generate", config)
    configure_cpu_runtime(config.runtime, config.data)
    from .data import SamplingPlan, write_dataset

    bundle, namespace = _circuit_and_namespace(config)
    plan = SamplingPlan.from_circuit(bundle, namespace, chunk_rows=config.data.shard_shots, total_rows=config.data.shots)
    config.paths.data_root.mkdir(parents=True, exist_ok=True)
    path = config.paths.data_root / (config.data.id or plan.dataset_id)
    write_dataset(path, plan, bundle, provenance={
        "command": "generate", "resolved_config": config.to_record(), "overrides": list(overrides),
        "environment": _environment_record(),
    })
    return path / "manifest.json"


def _training_stream(config: ResolvedConfig):
    from .data import SamplingPlan, open_dataset
    from .streams import TrajectoryStream

    bundle, namespace = _circuit_and_namespace(config)
    dataset = None
    if config.data.train_source == "stored_shards":
        dataset = open_dataset(config.paths.data_root / config.data.id, expected_split="train")
        plan = SamplingPlan.from_circuit(
            bundle, namespace, chunk_rows=dataset.plan.chunk_rows, total_rows=dataset.plan.total_rows,
        )
        if plan != dataset.plan:
            raise ArtifactError("stored training dataset differs from the selected circuit or stream namespace")
    else:
        plan = SamplingPlan.from_circuit(bundle, namespace, chunk_rows=config.data.training_chunk_shots)
    return TrajectoryStream(plan, bundle, dataset)


def _resume_contract(record: dict) -> dict:
    return {**record, "checkpoint": {**record["checkpoint"], "resume_from": None, "init_from": None}}


def _resume_environment(record: dict) -> dict:
    # Targets and execution settings are checked separately. Test/tool-only
    # package changes are irrelevant to the recorded numerical environment.
    return {
        "source": record["source"], "package_source_sha256": record["package_source_sha256"],
        "python": record["python"], "cpu": record["cpu"],
        "packages": {name: record["packages"].get(name) for name in (
            "nn-window-ablation", "torch", "torch-geometric", "stim", "numpy",
        )},
    }


def _package_source_sha256() -> str:
    root = Path(__file__).resolve().parent
    digest = sha256()
    for path in sorted(item for item in root.rglob("*") if item.suffix in (".py", ".yaml")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(sha256(path.read_bytes()).digest())
    return digest.hexdigest()


@contextmanager
def _training_lock(path: Path):
    with (path / ".training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ArtifactError("another process is updating the selected training run") from exc
        yield


def train_workflow(
    config: ResolvedConfig, *, overrides: tuple[str, ...] = (), sweep_index: int | None = None,
    stop_after: int | None = None,
) -> Path:
    """Train to the fixed budget, with an optional phase-local pause for bounded checks."""
    check_execution("train", config)
    configure_cpu_runtime(config.runtime, config.data)
    import torch

    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        return _run_training(config, overrides=overrides, sweep_index=sweep_index, stop_after=stop_after)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _run_training(
    config: ResolvedConfig, *, overrides: tuple[str, ...], sweep_index: int | None, stop_after: int | None,
) -> Path:
    import torch

    from .checkpoints import load_checkpoint, save_checkpoint
    from .training import MemoryObjective, Trainer, UpdateBudget, fresh_decoder
    from .windows import Buffers, buffer_lengths

    settings = config.training
    stop = settings.max_steps if stop_after is None else stop_after
    if type(stop) is not int or not 0 <= stop <= settings.max_steps:
        raise ConfigError("pause must be within the declared phase update budget")
    budget = UpdateBudget(settings.max_steps, settings.batch_size, settings.microbatch_size or settings.batch_size)
    stream = _training_stream(config)
    model = fresh_decoder(stream.plan)
    optimizer_type = torch.optim.Adam if settings.optimizer == "adam" else torch.optim.SGD
    optimizer = optimizer_type(model.parameters(), lr=settings.learning_rate, foreach=False)
    scheduler = (
        torch.optim.lr_scheduler.StepLR(optimizer, settings.scheduler_step_size, gamma=settings.scheduler_gamma)
        if settings.scheduler == "step" else None
    )
    choices = buffer_lengths(config.geometry.distance) if settings.dynamic_buffers.enabled else None
    trainer = Trainer(model, optimizer, scheduler, stream, budget, architecture=model.architecture(), dynamic_choices=choices)
    objective = MemoryObjective(
        stream.bundle, core_context=settings.core_context,
        fixed_buffers=Buffers(config.geometry.left_buffer, config.geometry.right_buffer), core_weight=settings.core_loss_weight,
    )
    declared = config.to_record()
    environment = {**_environment_record(), "package_source_sha256": _package_source_sha256()}
    root = config.paths.run_root / config.run_name
    resume = config.checkpoint.resume_from
    if resume is not None:
        directory = resume.parent
        relative = directory.relative_to(config.paths.run_root) if directory.is_relative_to(config.paths.run_root) else None
        if resume.name != "checkpoint.pt" or not directory.is_relative_to(root) or relative is None:
            raise ArtifactError("resume must address checkpoint.pt in the selected original run")
    else:
        parent = config.checkpoint.init_from
        parent_record = None
        if parent is not None:
            parent_state = load_checkpoint(parent)
            trainer.fork_from(parent_state)
            parent_record = {"path": str(parent), **file_integrity(parent)}
        if sweep_index is None:
            directory = reserve_run_directory(config)
        else:
            directory = root / "hydra_sweep" / str(sweep_index)
            directory.mkdir(parents=True)
        relative = directory.relative_to(config.paths.run_root)
    checkpoint = directory / "checkpoint.pt"

    with _training_lock(directory):
        if resume is not None:
            state = load_checkpoint(resume)
            run = state.get("run", {})
            if (
                run.get("directory") != relative.as_posix()
                or run.get("resume_contract") != _resume_contract(declared)
                or run.get("replay_environment") != _resume_environment(environment)
            ):
                raise ArtifactError("checkpoint belongs to another run, resolved configuration, or environment")
            trainer.restore(state)
            if stop < trainer.phase_updates:
                raise ConfigError("pause cannot precede the restored phase update")
        else:
            run = {
                "directory": relative.as_posix(), "resolved_config": declared,
                "resume_contract": _resume_contract(declared), "overrides": list(overrides),
                "environment": environment, "replay_environment": _resume_environment(environment),
                "runtime": {
                    "torch_intraop": torch.get_num_threads(), "torch_interop": torch.get_num_interop_threads(),
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                },
                "parent": parent_record,
            }
            save_checkpoint(checkpoint, {**trainer.snapshot(), "run": run})
        write_json(directory / "run.json", {
            "schema_version": 1, "status": "running", "counters": trainer.counters, "run": run,
        })
        while trainer.phase_updates < stop:
            trainer.step(objective)
            save_checkpoint(checkpoint, {**trainer.snapshot(), "run": run})
        write_json(directory / "run.json", {
            "schema_version": 1, "status": "complete" if trainer.complete else "paused",
            "counters": trainer.counters, "checkpoint": file_integrity(checkpoint), "run": run,
        })
    return checkpoint
