"""Dataset and training workflows over resolved configuration and explicit paths."""

from hashlib import sha256
from pathlib import Path
import re
import resource
from time import perf_counter, process_time

from .artifacts import ArtifactError, exclusive_lock, file_integrity, verify_file, write_json
from .config import ConfigError, ResolvedConfig, require_execution_config, validate_run_name
from .continuation import CONTINUATION_ARMS, TRAINING_REGIMES
from .runtime import configure_cpu_runtime
from .workflows import _environment_record, reserve_run_directory


def check_execution(command: str, config: ResolvedConfig) -> None:
    """Check workflow configuration before reserving any artifact directories."""
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
    regime = TRAINING_REGIMES.get(training.name)
    if regime is None or regime[:3] != (training.data_stream_phase, training.initialization, training.core_context):
        raise ConfigError("training regime must match its initialization, context, and stream phase")
    if training.data_stream_phase == "base" and config.checkpoint.init_from is not None:
        raise ConfigError("fresh base training requires random initialization and full context")
    if training.data_stream_phase == "continuation" and config.geometry.core_count != 12:
        raise ConfigError("continuation training requires the complete 12-core partition")
    if training.validation_data_id is not None:
        validate_run_name(training.validation_data_id)
        if training.data_stream_phase != "continuation":
            raise ConfigError("training.validation_data_id selects terminal continuation validation")
    dynamic = training.dynamic_buffers
    if dynamic.enabled != (training.core_context == "sampled_window"):
        raise ConfigError("sampled-window context and dynamic buffer mode must agree")
    if dynamic.enabled and (
        dynamic.ratios != [0, 0.5, 1] or dynamic.distribution != "independent_uniform" or dynamic.sampling_unit != "minibatch"
    ):
        raise ConfigError("dynamic buffer draws require the prescribed uniform logical-minibatch policy")
    if config.checkpoint.path is not None:
        raise ConfigError("checkpoint.path selects evaluation weights; training uses init_from or resume_from")


def check_matched_sweep(configs: list[ResolvedConfig]) -> None:
    """Preflight S+/W/D sweep groups before creating run directories."""
    groups = {}
    for config in configs:
        if config.training.name in CONTINUATION_ARMS:
            key = config.seed, config.data.replicate, config.geometry.distance
            groups.setdefault(key, []).append(config)
    for arms in groups.values():
        if {config.training.name for config in arms} != CONTINUATION_ARMS:
            continue
        expected = None
        for config in arms:
            record = config.to_record()
            shared = {key: record[key] for key in ("code", "noise", "window", "model", "data", "runtime")}
            shared["training"] = {
                key: value for key, value in record["training"].items()
                if key not in ("name", "core_context", "dynamic_buffers")
            }
            shared["parent"] = record["checkpoint"]["init_from"]
            shared["data_root"] = record["paths"]["data_root"]
            if expected is not None and shared != expected:
                raise ConfigError("S+/W/D sweep arms must share their parent, data, optimizer, and budget controls")
            expected = shared


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
    from .runtime import runtime_observation

    bundle, namespace = _circuit_and_namespace(config)
    plan = SamplingPlan.from_circuit(bundle, namespace, chunk_rows=config.data.shard_shots, total_rows=config.data.shots)
    config.paths.data_root.mkdir(parents=True, exist_ok=True)
    path = config.paths.data_root / (config.data.id or plan.dataset_id)
    write_dataset(path, plan, bundle, provenance={
        "command": "generate", "resolved_config": config.to_record(), "overrides": list(overrides),
        "environment": _environment_record(), "runtime": runtime_observation(config.data),
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


def _validation_dataset(config: ResolvedConfig, plan):
    from .data import open_dataset

    name = config.training.validation_data_id
    if name is None:
        return None
    mapped = open_dataset(config.paths.data_root / name, expected_split="validation")
    if (mapped.plan.spec, mapped.plan.original_sha256, mapped.plan.teacher_sha256) != (
        plan.spec, plan.original_sha256, plan.teacher_sha256,
    ):
        raise ArtifactError("validation dataset physical circuit differs from training")
    return mapped


def _continuation_controls(trainer, objective, parent: dict, initial_model: str, validation, config: ResolvedConfig) -> dict:
    return {
        "parent_sha256": parent["sha256"], "initial_model_sha256": initial_model,
        "trainer": {key: value for key, value in trainer.contract().items() if key != "dynamic_choices"},
        "global_context": "full_closed", "core_loss_weight": objective.core_weight,
        "loss_normalization": "one_plus_core_loss_weight", "checkpoint_policy": "fixed_updates",
        "sampled_cores_per_trajectory": 1, "core_sampling": "uniform", "optimizer_reset": True,
        "deployment_buffers": [objective.fixed_buffers.left, objective.fixed_buffers.right],
        "execution": [config.runtime.intraop_threads, config.runtime.interop_threads, config.runtime.allocated_cpus],
        "validation": None if validation is None else {
            "schedule": "terminal_only", "phase_update": trainer.budget.max_steps,
            "dataset_id": validation.plan.dataset_id, "manifest_sha256": validation.manifest_sha256,
            "shots": validation.plan.total_rows,
        },
    }


def _terminal_validation(trainer, objective, mapped) -> dict:
    import numpy as np

    from .evaluation import FrozenEvaluator, compose_core_decisions, model_state_sha256

    before = model_state_sha256(trainer.model)
    evaluator = FrozenEvaluator(trainer.model, mapped.bundle, objective.fixed_buffers)
    failures = 0
    batch_size = trainer.budget.microbatch_size
    for descriptor in mapped.shards:
        with mapped.open_shard(descriptor.index) as shard:
            for start in range(0, shard.rows, batch_size):
                shots = shard.read(slice(start, start + batch_size)).shots
                predictions = evaluator.logits("S", shots.detectors, shots.shot_ids) >= 0
                failures += int(np.count_nonzero(compose_core_decisions(predictions) ^ shots.global_labels))
    if model_state_sha256(trainer.model) != before:
        raise ArtifactError("terminal validation changed model parameters or buffers")
    return {
        "phase_update": trainer.phase_updates, "shots": mapped.plan.total_rows, "failures": failures,
        "ler": failures / mapped.plan.total_rows,
    }


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


def _package_source_sha256(root: Path | None = None) -> str:
    if root is None:
        root = Path(__file__).resolve().parent
    digest = sha256()
    for path in sorted(item for item in root.rglob("*") if item.suffix in (".py", ".yaml")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(sha256(path.read_bytes()).digest())
    return digest.hexdigest()


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
    with _training_stream(config) as stream:
        return _train_stream(config, stream, overrides=overrides, sweep_index=sweep_index, stop_after=stop_after)


def _train_stream(config, stream, *, overrides, sweep_index, stop_after) -> Path:
    import torch

    from .checkpoints import load_checkpoint
    from .continuation import checked_audit, model_tensors_sha256, record_update, start_audit
    from .training import MemoryObjective, Trainer, UpdateBudget, fresh_decoder
    from .windows import Buffers, buffer_lengths

    settings = config.training
    stop = settings.max_steps if stop_after is None else stop_after
    if type(stop) is not int or not 0 <= stop <= settings.max_steps:
        raise ConfigError("pause must be within the declared phase update budget")
    budget = UpdateBudget(settings.max_steps, settings.batch_size, settings.microbatch_size or settings.batch_size)
    validation = _validation_dataset(config, stream.plan)
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
            parent_record = {"path": str(parent), **file_integrity(parent)}
            parent_state = load_checkpoint(parent)
            verify_file(parent, {key: parent_record[key] for key in ("bytes", "sha256")})
            trainer.fork_from(parent_state)
        if sweep_index is None:
            directory = reserve_run_directory(config)
        else:
            directory = root / "hydra_sweep" / str(sweep_index)
            directory.mkdir(parents=True)
        relative = directory.relative_to(config.paths.run_root)
    checkpoint = directory / "checkpoint.pt"
    audit = None

    with exclusive_lock(directory / ".training.lock", "training"):
        for entry in directory.iterdir():
            if re.fullmatch(r"\.(?:checkpoint\.pt|run\.json)\.[a-z0-9_]{8}", entry.name) and entry.is_file():
                entry.unlink()
        if resume is not None:
            state = load_checkpoint(resume)
            run = state.get("run", {})
            if run.get("replay_environment") != _resume_environment(environment):
                raise ArtifactError(
                    "checkpoint resume environment or source differs; use the original checkout and environment for exact replay"
                )
            if (
                run.get("directory") != relative.as_posix()
                or run.get("resume_contract") != _resume_contract(declared)
            ):
                raise ArtifactError("checkpoint belongs to another run, resolved configuration, or environment")
            trainer.restore(state)
            if trainer.parent_updates:
                audit = checked_audit(state.get("continuation_audit"), updates=trainer.phase_updates)
                controls = _continuation_controls(
                    trainer, objective, run["parent"], audit["controls"]["initial_model_sha256"], validation, config,
                )
                if audit["controls"] != controls:
                    raise ArtifactError("checkpoint continuation controls differ from this run")
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
            if parent_record is not None:
                initial = model_tensors_sha256(model.state_dict())
                audit = start_audit(_continuation_controls(trainer, objective, parent_record, initial, validation, config))
            _save_training_checkpoint(checkpoint, trainer, run, audit)
        write_json(directory / "run.json", {
            "schema_version": 1, "status": "running", "counters": trainer.counters, "run": run,
        })
        cadence = config.checkpoint.save_every_updates
        while trainer.phase_updates < stop:
            wall, cpu = perf_counter(), process_time()
            trace = trainer.step(objective)
            if audit is not None:
                record_update(
                    audit, trace, detector_tokens=objective.detector_tokens(trace.cores, trace.buffers),
                    wall=perf_counter() - wall, cpu=process_time() - cpu,
                    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    allocated_cpus=config.runtime.allocated_cpus,
                )
            if trace.update == stop or (cadence is not None and trace.update % cadence == 0):
                _save_training_checkpoint(checkpoint, trainer, run, audit)
        stream.close()
        if validation is not None and trainer.complete and audit["validation"] is None:
            audit["validation"] = _terminal_validation(trainer, objective, validation)
            _save_training_checkpoint(checkpoint, trainer, run, audit)
        write_json(directory / "run.json", {
            "schema_version": 1, "status": "complete" if trainer.complete else "paused",
            "counters": trainer.counters, "checkpoint": file_integrity(checkpoint), "run": run,
            **({"continuation_audit": audit} if audit is not None else {}),
        })
    return checkpoint


def _save_training_checkpoint(checkpoint: Path, trainer, run: dict, audit: dict | None) -> None:
    from .checkpoints import save_checkpoint

    state = {**trainer.snapshot(), "run": run}
    if audit is not None:
        state["continuation_audit"] = audit
    save_checkpoint(checkpoint, state)
