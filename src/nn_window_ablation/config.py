"""Resolve composed Hydra settings to typed Python records."""

from dataclasses import asdict, dataclass
from fractions import Fraction
from math import ceil, isfinite
from pathlib import Path
import re
from typing import cast
from uuid import uuid4

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException


class ConfigError(ValueError):
    """A configuration cannot be used with the current study."""


@dataclass
class CodeConfig:
    family: str
    layout: str
    memory_basis: str
    distance: int
    rounds_per_distance: int


@dataclass
class NoiseConfig:
    name: str
    probability: float
    after_clifford_depolarization: float
    before_round_data_depolarization: float
    before_measure_flip_probability: float
    after_reset_flip_probability: float

    @property
    def circuit_probabilities(self) -> dict[str, float]:
        return {
            "after_clifford_depolarization": self.after_clifford_depolarization,
            "before_round_data_depolarization": self.before_round_data_depolarization,
            "before_measure_flip_probability": self.before_measure_flip_probability,
            "after_reset_flip_probability": self.after_reset_flip_probability,
        }


@dataclass
class WindowConfig:
    strategy: str
    core_per_distance: float
    left_per_distance: float
    right_per_distance: float
    reset_recurrent_state: bool
    final_readout_policy: str
    crop_before_features: bool


@dataclass(frozen=True)
class WindowGeometry:
    distance: int
    rounds: int
    core_length: int
    left_buffer: int
    right_buffer: int
    width: int
    core_count: int


@dataclass
class ModelConfig:
    name: str
    upstream_commit: str
    graph_channels: list[int]
    feature_group_rounds: int
    hidden_size: int
    gru_layers: int
    spatial_aggregation: str
    query_projection: str
    head: str
    empty_observation_embedding: str


@dataclass
class RuntimeConfig:
    device: str
    dtype: str
    processes: int
    intraop_threads: int
    interop_threads: int
    allocated_cpus: int | None
    benchmark_threads: list[int]


@dataclass
class DataConfig:
    family_id: str
    id: str | None
    split: str
    shots: int | None
    shard_shots: int | None
    training_chunk_shots: int | None
    num_workers: int
    sampling_processes: int
    train_source: str
    storage: str
    bitorder: str
    keep_zero_event_shots: bool
    label_schema: str


@dataclass
class DynamicBuffersConfig:
    enabled: bool
    ratios: list[float]
    distribution: str
    sampling_unit: str


@dataclass
class TrainingConfig:
    name: str
    initialization: str
    global_context: str
    core_context: str
    core_loss_weight: float
    loss_normalization: str
    sampled_cores_per_trajectory: int
    core_sampling: str
    data_stream_phase: str
    max_steps: int | None
    batch_size: int | None
    learning_rate: float | None
    optimizer: str | None
    scheduler: str | None
    reset_optimizer_at_start: bool
    checkpoint_policy: str
    dynamic_buffers: DynamicBuffersConfig


@dataclass
class EvaluationConfig:
    id: str
    query: str
    context: str
    aggregation: str
    threshold: float
    reset_recurrent_state_per_query: bool
    update_model_state: bool


@dataclass
class AnalysisConfig:
    unit: str
    confidence_level: float
    ler_interval: str
    paired_interval: str
    sparse_interval: str
    sparse_rule: str | None
    bootstrap_replicates: int | None
    contrasts: list[str]
    pool_training_seeds_as_independent_shots: bool


@dataclass
class PathsConfig:
    data_root: Path
    run_root: Path


@dataclass
class CheckpointConfig:
    path: Path | None
    init_from: Path | None
    resume_from: Path | None


@dataclass
class ResolvedConfig:
    schema_version: int
    seed: int
    run_name: str
    study_stage: str
    code: CodeConfig
    noise: NoiseConfig
    window: WindowConfig
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    evaluation: EvaluationConfig
    analysis: AnalysisConfig
    runtime: RuntimeConfig
    paths: PathsConfig
    checkpoint: CheckpointConfig

    @property
    def geometry(self) -> WindowGeometry:
        distance = self.code.distance
        rounds = distance * self.code.rounds_per_distance
        core = distance * Fraction(str(self.window.core_per_distance))
        if core <= 0 or core.denominator != 1 or rounds % core:
            raise ConfigError("window.core_per_distance must yield whole cores that partition the rounds")
        if core != distance:
            raise ConfigError("window.core_per_distance must give a core length equal to code.distance")

        left = ceil(distance * Fraction(self.window.left_per_distance))
        right = ceil(distance * Fraction(self.window.right_per_distance))
        return WindowGeometry(distance, rounds, int(core), left, right, int(core) + left + right, rounds // int(core))

    def to_record(self) -> dict:
        """Return a portable record with source fields and derived geometry."""
        record = asdict(self)
        record["geometry"] = asdict(self.geometry)
        for group in ("paths", "checkpoint"):
            record[group] = {
                key: str(value) if value is not None else None
                for key, value in record[group].items()
            }
        return record


def new_run_name() -> str:
    return f"run-{uuid4().hex}"


def validate_run_name(name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name) or name.endswith("."):
        raise ConfigError("run_name must be a portable name of at most 128 letters, digits, dots, underscores or hyphens")


def resolve_path(value: Path, invocation_dir: Path, *, directory: bool = False) -> Path:
    try:
        path = (invocation_dir / value.expanduser()).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ConfigError(f"invalid path {value}: {exc}") from exc
    if directory:
        for candidate in (path, *path.parents):
            if candidate.exists() and not candidate.is_dir():
                raise ConfigError(f"expected a directory at {candidate}")
    return path


def resolve_config(config: DictConfig, *, invocation_dir: Path) -> ResolvedConfig:
    """Validate a composed task configuration without creating run artifacts."""
    try:
        raw = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        for group in ("code", "runtime", "data", "training", "analysis", "paths", "checkpoint"):
            if not isinstance(raw.get(group), dict):
                raise ConfigError(f"{group} must be a mapping")
        for group, names in {
            "code": ("distance", "rounds_per_distance"),
            "runtime": ("processes", "intraop_threads", "interop_threads", "allocated_cpus"),
            "data": ("num_workers", "sampling_processes", "shots", "shard_shots", "training_chunk_shots"),
            "training": ("max_steps", "batch_size"),
            "analysis": ("bootstrap_replicates",),
        }.items():
            for name in names:
                value = raw[group][name]
                if value is not None and type(value) is not int:
                    raise ConfigError(f"{group}.{name} must be an integer")
        if type(raw["run_name"]) is not str:
            raise ConfigError("run_name must be a string")
        validate_run_name(raw["run_name"])
        for group in ("paths", "checkpoint"):
            for name, value in raw[group].items():
                if group == "checkpoint" and value is None:
                    continue
                if type(value) is not str or not value.strip() or "\0" in value:
                    raise ConfigError(f"{group}.{name} must be a nonempty path")
        typed = OmegaConf.merge(OmegaConf.structured(ResolvedConfig), raw)
        resolved = cast(ResolvedConfig, OmegaConf.to_object(typed))
    except (OmegaConfBaseException, KeyError) as exc:
        raise ConfigError(f"invalid configuration: {exc}") from exc

    _validate_settings(resolved)
    for field in ("data_root", "run_root"):
        value = getattr(resolved.paths, field)
        setattr(resolved.paths, field, resolve_path(value, invocation_dir, directory=True))
    for field in ("path", "init_from", "resume_from"):
        value = getattr(resolved.checkpoint, field)
        if value is not None:
            setattr(resolved.checkpoint, field, resolve_path(value, invocation_dir))
    return resolved


def _validate_settings(config: ResolvedConfig) -> None:
    code, noise, window, model = config.code, config.noise, config.window, config.model
    if config.schema_version != 1:
        raise ConfigError("schema_version must be 1")
    if (code.family, code.layout, code.memory_basis) != ("surface_code", "rotated", "Z"):
        raise ConfigError("code must be rotated surface-code Z memory")
    if code.distance < 3 or code.distance % 2 != 1:
        raise ConfigError("code.distance must be an odd integer of at least 3")
    if code.rounds_per_distance <= 0:
        raise ConfigError("code.rounds_per_distance must be positive")
    if not isfinite(window.core_per_distance):
        raise ConfigError("window.core_per_distance must be finite")
    for field in ("left_per_distance", "right_per_distance"):
        if getattr(window, field) not in (0, 0.5, 1):
            raise ConfigError(f"window.{field} must be 0, 0.5 or 1")
    # Force validation of the complete temporal partition before any path is created.
    config.geometry
    if noise.name != "uniform_circuit" or not 0 <= noise.probability <= 1:
        raise ConfigError("noise must be uniform_circuit with noise.probability in [0, 1]")
    if any(value != noise.probability for value in noise.circuit_probabilities.values()):
        raise ConfigError("the four circuit-noise probabilities must equal noise.probability")
    if model.name != "query_gnn_rnn":
        raise ConfigError("model.name must be query_gnn_rnn")
    if (
        (model.hidden_size, model.gru_layers, model.graph_channels, model.feature_group_rounds)
        != (256, 4, [3, 64, 256], 2)
        or (model.spatial_aggregation, model.query_projection, model.head)
        != ("distance_dependent_lange", "additive", "shared_scalar")
    ):
        raise ConfigError("model must retain the specified Lange backbone and shared query head")
    if not (
        window.strategy == "independent_core_parity"
        and window.reset_recurrent_state
        and window.final_readout_policy == "final_core_only"
        and window.crop_before_features
        and config.data.keep_zero_event_shots
    ):
        raise ConfigError("window and data settings must preserve the study's observation policy")
    if config.checkpoint.init_from is not None and config.checkpoint.resume_from is not None:
        raise ConfigError("checkpoint.init_from and checkpoint.resume_from are mutually exclusive")
    _validate_budgets(config)
    validate_cpu_budget(config.runtime, config.data)


def _validate_budgets(config: ResolvedConfig) -> None:
    for name, value in (
        ("data.shots", config.data.shots),
        ("data.shard_shots", config.data.shard_shots),
        ("data.training_chunk_shots", config.data.training_chunk_shots),
        ("training.max_steps", config.training.max_steps),
        ("training.batch_size", config.training.batch_size),
        ("training.learning_rate", config.training.learning_rate),
        ("analysis.bootstrap_replicates", config.analysis.bootstrap_replicates),
    ):
        if value is not None and (value <= 0 or isinstance(value, float) and not isfinite(value)):
            raise ConfigError(f"{name} must be positive when provided")
    if not 0 < config.analysis.confidence_level < 1:
        raise ConfigError("analysis.confidence_level must be in (0, 1)")


def validate_cpu_budget(runtime: RuntimeConfig, data: DataConfig) -> int:
    """Return the CPU requirement under the configured concurrency model."""
    if (runtime.device, runtime.dtype) != ("cpu", "float32"):
        raise ConfigError("runtime must use device=cpu and dtype=float32")
    counts = (
        ("runtime.processes", runtime.processes),
        ("runtime.intraop_threads", runtime.intraop_threads),
        ("runtime.interop_threads", runtime.interop_threads),
        ("data.sampling_processes", data.sampling_processes),
    )
    for name, value in counts:
        if value < 1:
            raise ConfigError(f"{name} must be positive")
    if data.num_workers < 0:
        raise ConfigError("data.num_workers must be nonnegative")
    if not runtime.benchmark_threads or any(value < 1 for value in runtime.benchmark_threads):
        raise ConfigError("runtime.benchmark_threads must contain positive thread counts")

    # A single sampler runs inline. Parallel sampler and loader workers get one CPU each.
    sampler_workers = data.sampling_processes if data.sampling_processes > 1 else 0
    per_process = runtime.intraop_threads * runtime.interop_threads + data.num_workers + sampler_workers
    required = runtime.processes * per_process
    if runtime.allocated_cpus is not None:
        if runtime.allocated_cpus < 1:
            raise ConfigError("runtime.allocated_cpus must be positive")
        if required > runtime.allocated_cpus:
            raise ConfigError(
                f"runtime requires {required} CPUs, exceeding runtime.allocated_cpus={runtime.allocated_cpus}"
            )
    return required


def require_execution_config(config: ResolvedConfig, command: str) -> None:
    """Reject unresolved fields required before a scientific workflow starts."""
    fields = {
        "generate": ("data.shots", "data.shard_shots"),
        "train": (
            "training.max_steps", "training.batch_size", "training.learning_rate",
            "training.optimizer", "training.scheduler",
            "data.training_chunk_shots" if config.data.train_source == "generated_chunks" else "data.id",
        ),
        "evaluate": ("data.id", "checkpoint.path"),
        "analyze": ("analysis.bootstrap_replicates", "analysis.sparse_rule"),
    }[command]
    if command == "train" and config.training.initialization == "full_checkpoint":
        if config.checkpoint.init_from is None and config.checkpoint.resume_from is None:
            fields += ("checkpoint.init_from",)

    missing = []
    for field in ("runtime.allocated_cpus", *fields):
        group, name = field.split(".")
        value = getattr(getattr(config, group), name)
        if value is None or isinstance(value, str) and not value.strip():
            missing.append(field)
    if missing:
        raise ConfigError(f"{command} requires explicit values for {', '.join(missing)}")
