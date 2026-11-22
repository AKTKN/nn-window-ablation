"""The shared global/core objective and explicit per-trajectory core sampling."""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits


@dataclass(frozen=True, slots=True)
class QueryLoss:
    total: Tensor
    global_term: Tensor
    core_term: Tensor


def sample_core_indices(
    trajectories: int, core_count: int, *, rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Draw one zero-based uniform core index per trajectory."""
    if type(trajectories) is not int or trajectories < 1 or type(core_count) is not int or core_count < 1:
        raise ValueError("trajectory and core counts must be positive integers")
    return rng.integers(0, core_count, size=trajectories, dtype=np.int64)


def query_loss(
    global_logits: Tensor,
    global_labels: Tensor,
    core_logits: Tensor,
    core_labels: Tensor,
    *,
    core_indices: Tensor | None = None,
    core_weight: float = 1.0,
) -> QueryLoss:
    """Average across trajectories with either all core logits or one sampled logit each.

    The full label matrix is always ordered by trajectory and zero-based core.
    Labels and sampling choices are supplied by the caller.
    """
    if not isfinite(core_weight) or core_weight < 0:
        raise ValueError("core weight must be finite and nonnegative")
    if global_logits.ndim != 1 or not global_logits.numel() or global_labels.shape != global_logits.shape:
        raise ValueError("global logits and labels must be aligned nonempty vectors")
    trajectories = global_logits.shape[0]
    if core_labels.ndim != 2 or core_labels.shape[0] != trajectories or core_labels.shape[1] == 0:
        raise ValueError("core labels must be a nonempty trajectory-by-core matrix")

    if core_indices is None:
        if core_logits.shape != core_labels.shape:
            raise ValueError("full core logits must match the core label matrix")
        targets = core_labels
    else:
        if core_logits.shape != (trajectories,) or core_indices.shape != (trajectories,) or core_indices.dtype != torch.long:
            raise ValueError("sampled core logits and int64 indices require one entry per trajectory")
        if bool(((core_indices < 0) | (core_indices >= core_labels.shape[1])).any()):
            raise ValueError("sampled core index is outside the core label matrix")
        indices = core_indices.to(device=core_labels.device)
        targets = core_labels.gather(1, indices[:, None]).squeeze(1)

    global_term = binary_cross_entropy_with_logits(global_logits, global_labels.to(global_logits))
    core_term = binary_cross_entropy_with_logits(core_logits, targets.to(core_logits))
    return QueryLoss((global_term + core_weight * core_term) / (1 + core_weight), global_term, core_term)
