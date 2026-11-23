"""Frozen global and core-query inference on complete detector trajectories."""

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from numpy.typing import NDArray
import stim
import torch

from .artifacts import ArtifactError, canonical_json
from .data import SamplingPlan
from .features import batch_features, build_feature_geometry, build_features
from .models.gnn_rnn import QueryDecoder, prepare_model_batch
from .simulation import CircuitBundle
from .training import final_update_count
from .windows import Buffers, WindowView, build_observation_layout, crop_observations, select_window


REGIMES = {
    "full_closed": ("base", "random", "full_closed", "S"),
    "continued_full": ("continuation", "full_checkpoint", "full_closed", "S+"),
    "fixed_window": ("continuation", "full_checkpoint", "deployment_window", "W"),
    "dynamic_window": ("continuation", "full_checkpoint", "sampled_window", "D"),
}


def condition_id(mode: str, training_regime: str) -> str:
    if mode not in ("G", "F", "S") or training_regime not in REGIMES:
        raise ValueError("unknown inference mode or checkpoint training regime")
    if mode == "S":
        return REGIMES[training_regime][3]
    return mode if training_regime == "full_closed" else f"{mode}@{training_regime}"


@dataclass(frozen=True, slots=True)
class CheckpointProvenance:
    sha256: str
    training_regime: str
    training_seed: int
    training_replicate: int
    training_family: str
    parent_sha256: str | None
    cumulative_updates: int

    def __post_init__(self) -> None:
        if self.training_regime not in REGIMES or not _is_digest(self.sha256):
            raise ValueError("invalid checkpoint provenance regime or digest")
        if self.training_regime == "full_closed":
            if self.parent_sha256 is not None:
                raise ValueError("a base checkpoint cannot have a parent digest")
        elif not _is_digest(self.parent_sha256):
            raise ValueError("a continuation requires a valid parent digest")
        if (
            type(self.training_seed) is not int or not 0 <= self.training_seed < 2**64
            or type(self.training_replicate) is not int or self.training_replicate < 0
            or not isinstance(self.training_family, str) or not self.training_family.strip()
            or type(self.cumulative_updates) is not int or self.cumulative_updates < 1
        ):
            raise ValueError("invalid checkpoint training seed, replicate, family, or update count")

    def condition(self, mode: str) -> str:
        return condition_id(mode, self.training_regime)

    @classmethod
    def from_payload(cls, payload: dict, digest: str, dataset: SamplingPlan) -> "CheckpointProvenance":
        """Check P4 provenance against the held-out recipe before running a model."""
        try:
            recipe = payload["contract"]["source"]["plan"]
            dataset_record = dataset.record()
            if any(recipe[key] != dataset_record[key] for key in ("label_schema", "feature_schema")):
                raise ArtifactError("checkpoint and dataset label or feature schemas differ")
            plan = SamplingPlan.from_record(recipe)
            if (plan.spec, plan.original_sha256, plan.teacher_sha256) != (
                dataset.spec, dataset.original_sha256, dataset.teacher_sha256,
            ):
                raise ArtifactError("checkpoint and dataset physical circuits differ")
            run = payload["run"]
            training = run["resolved_config"]["training"]
            regime = training["name"]
            namespace = plan.namespace
            regime_fields = namespace.phase, training["initialization"], training["core_context"]
            if (
                regime not in REGIMES or REGIMES[regime][:3] != regime_fields
                or training["data_stream_phase"] != namespace.phase
                or training["global_context"] != "full_closed" or namespace.split != "train"
            ):
                raise ArtifactError("checkpoint training regime contradicts its recorded context or phase")
            parent = run["parent"]
            if (parent is None) != (namespace.phase == "base"):
                raise ArtifactError("checkpoint parent provenance differs from its training phase")
            parent_digest = None if parent is None else parent["sha256"]
            return cls(
                digest, regime, namespace.master_seed, namespace.replicate, namespace.family,
                parent_digest, final_update_count(payload),
            )
        except ArtifactError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError("checkpoint lacks compatible P4 training provenance") from exc


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def model_state_sha256(model: QueryDecoder) -> str:
    """Hash parameter and buffer names, dtypes, shapes, and CPU tensor bytes."""
    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(canonical_json([name, str(tensor.dtype), list(tensor.shape)]) + b"\n")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def binary_array(values: NDArray, *, name: str) -> NDArray[np.bool_]:
    if values.dtype.kind not in "biu" or np.any((values != 0) & (values != 1)):
        raise ValueError(f"{name} must contain binary integers or booleans")
    return values.astype(np.bool_, copy=False)


def compose_core_decisions(decisions: NDArray, *, core_count: int = 12) -> NDArray[np.bool_]:
    if decisions.ndim != 2 or decisions.shape[1] != core_count or core_count < 1 or len(decisions) == 0:
        raise ValueError(f"core decisions must have one complete row of {core_count} cores per trajectory")
    return np.logical_xor.reduce(binary_array(decisions, name="core decisions"), axis=1)


class FrozenEvaluator:
    def __init__(self, model: QueryDecoder, bundle: CircuitBundle, buffers: Buffers):
        if model.distance != bundle.spec.distance:
            raise ValueError("model and circuit distances differ")
        self.model = model.eval()
        self.layout = build_observation_layout(bundle)
        self.geometry = build_feature_geometry(self.layout)
        self._views = {
            "G": (select_window(self.layout, self.layout.global_query),),
            "F": tuple(select_window(self.layout, target) for target in self.layout.queries),
            "S": tuple(select_window(self.layout, target, buffers) for target in self.layout.queries),
        }

    def views(self, mode: str) -> tuple[WindowView, ...]:
        if mode not in self._views:
            raise ValueError("expected G, F, or S inference mode")
        return self._views[mode]

    @torch.inference_mode()
    def logits(self, mode: str, detectors: NDArray[np.bool_], shot_ids: NDArray[np.int64]) -> NDArray[np.float32]:
        """Return rows ordered by trajectory, and columns by global/core query."""
        if self.model.training or any(module.training for module in self.model.modules()):
            raise ValueError("frozen evaluation requires evaluation mode for every model module")
        output = []
        for view in self.views(mode):
            cropped = crop_observations(view, detectors, shot_ids)
            examples = build_features(cropped, self.geometry)
            inputs = prepare_model_batch(batch_features(examples), distance=self.layout.distance)
            output.append(self.model(inputs).numpy())
        logits = np.column_stack(output)
        if logits.shape != (len(shot_ids), len(self.views(mode))) or not np.isfinite(logits).all():
            raise ValueError("model returned incompatible or nonfinite query logits")
        return logits


class MatchingReference:
    def __init__(self, dem: stim.DetectorErrorModel, *, detector_count: int):
        import pymatching

        if (dem.num_detectors, dem.num_observables) != (detector_count, 1):
            raise ValueError("matching requires the original global-only DEM and its detector dimensions")
        self.detector_count = detector_count
        self.version = pymatching.__version__
        self.decoder = pymatching.Matching.from_detector_error_model(dem)

    def decisions(self, detectors: NDArray[np.bool_]) -> NDArray[np.bool_]:
        if detectors.ndim != 2 or detectors.shape[1] != self.detector_count or detectors.dtype != np.bool_:
            raise ValueError("matching detector rows disagree with the original DEM")
        result = self.decoder.decode_batch(detectors)
        if result.shape != (len(detectors), 1):
            raise ValueError("matching must return one physical logical decision per trajectory")
        return binary_array(result, name="matching decisions")
