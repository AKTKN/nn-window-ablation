"""Fixed-update training with independent RNG streams and replayable state."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import random
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor, nn

from .artifacts import ArtifactError
from .data import SamplingPlan, StreamNamespace, TrajectoryID
from .features import FeatureBatch, batch_features, build_feature_geometry, build_features
from .models.gnn_rnn import QueryDecoder, prepare_model_batch
from .models.objective import query_loss, sample_core_indices
from .simulation import CircuitBundle, ShotBatch
from .streams import ChunkCursor, TrajectoryStream
from .windows import Buffers, WindowView, build_observation_layout, crop_observations, select_window


Loss = Callable[[nn.Module, ShotBatch, NDArray[np.int64], Buffers | None], Tensor]


class IntegerDraw(Protocol):
    def integers(self, high: int, *, size: int) -> NDArray[np.int64]: ...


def sample_buffer_pair(choices: tuple[int, ...], rng: IntegerDraw) -> Buffers:
    """Draw two independent uniform indices for one effective minibatch."""
    left, right = rng.integers(len(choices), size=2)
    return Buffers(choices[left], choices[right])


@dataclass(frozen=True, slots=True)
class UpdateBudget:
    max_steps: int
    batch_size: int
    microbatch_size: int

    def __post_init__(self) -> None:
        values = (self.max_steps, self.batch_size, self.microbatch_size)
        if any(type(value) is not int or value < 1 for value in values) or self.microbatch_size > self.batch_size:
            raise ValueError("positive update and batch sizes required, with microbatch no larger than effective batch")


@dataclass(frozen=True, slots=True)
class StepTrace:
    update: int
    loss: float
    trajectories: tuple[TrajectoryID, ...]
    cores: tuple[int, ...]
    buffers: Buffers | None


class TrainingRandomness:
    """Own the state of independent streams, including process-global model RNGs."""

    def __init__(self, namespace: StreamNamespace):
        self.core = np.random.Generator(np.random.PCG64(namespace.seed("core_query")))
        self.buffer = np.random.Generator(np.random.PCG64(namespace.seed("dynamic_buffer")))
        self.torch_state = torch.Generator().manual_seed(namespace.seed("model_stochastic")).get_state()
        self.python_state = random.Random(namespace.seed("python_stochastic")).getstate()

    @contextmanager
    def model_stream(self) -> Iterator[None]:
        previous_python = random.getstate()
        with torch.random.fork_rng(devices=[]):
            try:
                torch.set_rng_state(self.torch_state)
                random.setstate(self.python_state)
                yield
            finally:
                self.torch_state = torch.get_rng_state()
                self.python_state = random.getstate()
                random.setstate(previous_python)

    def record(self) -> dict:
        return {
            "core_query": deepcopy(self.core.bit_generator.state),
            "dynamic_buffer": deepcopy(self.buffer.bit_generator.state),
            "torch_cpu": self.torch_state.clone(),
            "python": self.python_state,
        }

    def restore(self, state: dict) -> None:
        core = np.random.PCG64()
        buffer = np.random.PCG64()
        core.state = state["core_query"]
        buffer.state = state["dynamic_buffer"]
        torch_state = state["torch_cpu"]
        python_state = state["python"]
        torch.Generator().set_state(torch_state)
        random.Random().setstate(python_state)
        self.core = np.random.Generator(core)
        self.buffer = np.random.Generator(buffer)
        self.torch_state = torch_state.clone()
        self.python_state = python_state


def fresh_decoder(plan: SamplingPlan) -> QueryDecoder:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(plan.namespace.seed("initialization"))
        return QueryDecoder(plan.spec.distance)


def _take_rows(shots: ShotBatch, rows: slice) -> ShotBatch:
    return ShotBatch(
        shots.detectors[rows], shots.global_labels[rows], shots.core_labels[rows], shots.shot_ids[rows], None,
    )


def _counters(budget: UpdateBudget, parent: int, local: int) -> dict[str, int]:
    return {
        "parent_updates": parent, "phase_updates": local, "cumulative_updates": parent + local,
        "trajectories": local * budget.batch_size, "query_examples": 2 * local * budget.batch_size,
    }


class Trainer:
    def __init__(
        self, model: nn.Module, optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.StepLR | None, stream: TrajectoryStream, budget: UpdateBudget,
        *, architecture: dict, dynamic_choices: tuple[int, ...] | None = None,
    ):
        if budget.max_steps * budget.batch_size > stream.capacity:
            raise ValueError("training source cannot supply the declared fixed-update budget")
        if dynamic_choices is not None and (
            not dynamic_choices or any(type(value) is not int or value < 0 for value in dynamic_choices)
        ):
            raise ValueError("dynamic buffers require nonnegative candidate lengths")
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.stream = stream
        self.budget = budget
        self.architecture = deepcopy(architecture)
        self.dynamic_choices = dynamic_choices
        self.randomness = TrainingRandomness(stream.plan.namespace)
        self.parent_updates = 0
        self.phase_updates = 0
        self.last_update = None
        self._usable = True

    @property
    def counters(self) -> dict[str, int]:
        return _counters(self.budget, self.parent_updates, self.phase_updates)

    @property
    def complete(self) -> bool:
        return self.phase_updates == self.budget.max_steps

    def contract(self) -> dict:
        return {
            "budget": asdict(self.budget),
            "source": self.stream.record(),
            "architecture": self.architecture,
            "optimizer": {"type": type(self.optimizer).__name__, "defaults": self.optimizer.defaults},
            "scheduler": None if self.scheduler is None else {
                "type": type(self.scheduler).__name__, "step_size": self.scheduler.step_size, "gamma": self.scheduler.gamma,
            },
            "dynamic_choices": self.dynamic_choices,
        }

    def step(self, objective: Loss) -> StepTrace:
        if not self._usable:
            raise RuntimeError("reload a completed update checkpoint after a failed training action")
        self._check_parent()
        if self.complete:
            raise ValueError("the phase has reached its declared update budget")
        self._usable = False
        batch = self.stream.take(self.budget.batch_size)
        indices = sample_core_indices(self.budget.batch_size, self.stream.plan.core_count, rng=self.randomness.core)
        buffers = None
        if self.dynamic_choices is not None:
            buffers = sample_buffer_pair(self.dynamic_choices, self.randomness.buffer)
        accumulated = 0.0
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with self.randomness.model_stream():
            for start in range(0, self.budget.batch_size, self.budget.microbatch_size):
                stop = min(start + self.budget.microbatch_size, self.budget.batch_size)
                rows = slice(start, stop)
                mean_loss = objective(self.model, _take_rows(batch.shots, rows), indices[rows], buffers)
                if mean_loss.ndim != 0 or not bool(torch.isfinite(mean_loss)):
                    raise ValueError("each microbatch must supply a finite scalar mean loss")
                weighted = mean_loss * ((stop - start) / self.budget.batch_size)
                weighted.backward()
                accumulated += weighted.detach().item()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in self.model.parameters()):
                raise ValueError("nonfinite gradient in the logical minibatch")
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
        self.phase_updates += 1
        trace = StepTrace(self.phase_updates, accumulated, batch.identities, tuple(map(int, indices)), buffers)
        self.last_update = asdict(trace)
        self._usable = True
        return trace

    def snapshot(self) -> dict:
        """Return a checkpoint payload to serialize before the next update."""
        if not self._usable:
            raise RuntimeError("a failed training action has no completed checkpoint")
        self._check_parent()
        return {
            "contract": self.contract(), "counters": self.counters, "cursor": asdict(self.stream.cursor),
            "model": {"architecture": self.architecture, "state_dict": self.model.state_dict()},
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": None if self.scheduler is None else self.scheduler.state_dict(),
            "random_states": self.randomness.record(), "last_update": self.last_update,
        }

    def _check_parent(self) -> None:
        if self.stream.plan.namespace.phase == "continuation" and self.parent_updates == 0:
            raise ValueError("initialize a continuation from its completed base checkpoint first")

    def restore(self, payload: dict) -> None:
        """Restore a checkpoint into a reconstructed trainer for the same run."""
        self._usable = False
        try:
            if payload["contract"] != self.contract() or payload["model"]["architecture"] != self.architecture:
                raise ArtifactError("checkpoint training contract or model schema differs")
            parent, local = _check_counters(payload, self.budget)
            phase = self.stream.plan.namespace.phase
            if (phase == "base") != (parent == 0):
                raise ArtifactError("checkpoint parent counter is incompatible with the training phase")
            cursor = ChunkCursor(**payload["cursor"])
            if self.stream.validate_cursor(cursor) != local * self.budget.batch_size:
                raise ArtifactError("checkpoint data cursor disagrees with consumed trajectory count")
            trace = payload["last_update"]
            if (local == 0 and trace is not None) or (local > 0 and (not isinstance(trace, dict) or trace["update"] != local)):
                raise ArtifactError("checkpoint update trace disagrees with its counter")
            scheduler_state = payload["scheduler_state"]
            if (self.scheduler is None) != (scheduler_state is None):
                raise ArtifactError("checkpoint scheduler state does not match the run")
            self.model.load_state_dict(payload["model"]["state_dict"], strict=True)
            self.optimizer.load_state_dict(payload["optimizer_state"])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(scheduler_state)
            self.stream.seek(cursor)
            self.parent_updates, self.phase_updates = parent, local
            self.last_update = trace
            self.randomness.restore(payload["random_states"])
        except ArtifactError:
            raise
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise ArtifactError("invalid checkpoint optimizer, scheduler, model, or RNG state") from exc
        self._usable = True

    def fork_from(self, payload: dict) -> None:
        """Initialize a continuation from a completed base model with fresh training state."""
        if (
            self.phase_updates or self.parent_updates or self.stream.position or self.optimizer.state
            or (self.scheduler is not None and self.scheduler.last_epoch != 0)
        ):
            raise ValueError("a fork requires a fresh trainer, optimizer, and scheduler")
        parent_plan = SamplingPlan.from_record(payload["contract"]["source"]["plan"])
        current = self.stream.plan
        if (
            parent_plan.namespace.phase != "base"
            or current.namespace.phase != "continuation"
            or replace(parent_plan.namespace, phase="continuation") != current.namespace
        ):
            raise ArtifactError("fork requires the matching base replicate and a distinct continuation namespace")
        if (parent_plan.original_sha256, parent_plan.teacher_sha256) != (current.original_sha256, current.teacher_sha256):
            raise ArtifactError("fork parent circuit differs from the continuation circuit")
        if payload["model"]["architecture"] != self.architecture:
            raise ArtifactError("fork parent model schema differs")
        parent_count = final_update_count(payload)
        if payload["counters"]["parent_updates"] != 0:
            raise ArtifactError("a full-context base checkpoint must have no parent updates")
        self.model.load_state_dict(payload["model"]["state_dict"], strict=True)
        self.parent_updates = parent_count


def _check_counters(payload: dict, budget: UpdateBudget) -> tuple[int, int]:
    counters = payload["counters"]
    parent, local = counters["parent_updates"], counters["phase_updates"]
    if (
        type(parent) is not int or type(local) is not int or parent < 0 or not 0 <= local <= budget.max_steps
        or counters != _counters(budget, parent, local)
    ):
        raise ArtifactError("checkpoint update or example counters are inconsistent")
    return parent, local


def final_update_count(payload: dict) -> int:
    budget = UpdateBudget(**payload["contract"]["budget"])
    parent, local = _check_counters(payload, budget)
    if local != budget.max_steps:
        raise ArtifactError("checkpoint has not reached its fixed terminal update")
    return parent + local


class MemoryObjective:
    """Construct the two P1–P3 queries per trajectory using the requested context."""

    def __init__(
        self, bundle: CircuitBundle, *, core_context: str, fixed_buffers: Buffers, core_weight: float,
    ):
        self.layout = build_observation_layout(bundle)
        self.geometry = build_feature_geometry(self.layout)
        self.global_view = select_window(self.layout, self.layout.global_query)
        self.core_context = core_context
        self.fixed_buffers = fixed_buffers
        self.core_weight = core_weight
        self._core_views: dict[tuple[int, Buffers | None], WindowView] = {}

    def core_view(self, index: int, buffers: Buffers | None) -> WindowView:
        if self.core_context == "full_closed":
            selected = None
        elif self.core_context == "deployment_window":
            selected = self.fixed_buffers
        elif self.core_context == "sampled_window" and buffers is not None:
            selected = buffers
        else:
            raise ValueError("missing or unsupported core observation context")
        key = (index, selected)
        if key not in self._core_views:
            self._core_views[key] = select_window(self.layout, self.layout.core_query(index), selected)
        return self._core_views[key]

    def detector_tokens(self, cores: tuple[int, ...], buffers: Buffers | None) -> int:
        """Count raw detector columns selected by each global and core query."""
        return len(cores) * len(self.global_view.detectors) + sum(len(self.core_view(index, buffers).detectors) for index in cores)

    def prepare(self, shots: ShotBatch, cores: NDArray[np.int64], buffers: Buffers | None) -> FeatureBatch:
        if cores.dtype != np.int64 or cores.shape != shots.shot_ids.shape:
            raise ValueError("each trajectory requires one int64 core index")
        examples = build_features(crop_observations(self.global_view, shots.detectors, shots.shot_ids), self.geometry)
        core_examples = []
        for row, core in enumerate(cores):
            view = self.core_view(int(core), buffers)
            sample = slice(row, row + 1)
            core_examples.extend(build_features(
                crop_observations(view, shots.detectors[sample], shots.shot_ids[sample]), self.geometry,
            ))
        return batch_features(examples + tuple(core_examples))

    def __call__(self, model: nn.Module, shots: ShotBatch, cores: NDArray[np.int64], buffers: Buffers | None) -> Tensor:
        features = self.prepare(shots, cores, buffers)
        batch = prepare_model_batch(features, distance=self.layout.distance)
        logits = model(batch)
        count = len(shots.shot_ids)
        return query_loss(
            logits[:count], torch.from_numpy(shots.global_labels), logits[count:], torch.from_numpy(shots.core_labels),
            core_indices=torch.from_numpy(cores), core_weight=self.core_weight,
        ).total
