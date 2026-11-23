"""Whole-trajectory composition, estimators, and uncertainty."""

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binomtest

from .evaluation import binary_array, compose_core_decisions


@dataclass(frozen=True, slots=True)
class PairedCounts:
    n00: int
    n01: int
    n10: int
    n11: int

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 0 for value in self.categories) or not self.total:
            raise ValueError("paired counts must be nonnegative integers for at least one trajectory")

    @property
    def categories(self) -> tuple[int, int, int, int]:
        return self.n00, self.n01, self.n10, self.n11

    @property
    def total(self) -> int:
        return sum(self.categories)

    @property
    def difference(self) -> float:
        return (self.n10 - self.n01) / self.total

    @property
    def discordances(self) -> int:
        return self.n10 + self.n01


def require_paired_ids(first: NDArray[np.int64], second: NDArray[np.int64]) -> None:
    for ids in (first, second):
        if ids.ndim != 1 or ids.dtype != np.int64 or not len(ids) or np.any(ids < 0):
            raise ValueError("shot IDs must be a nonempty vector of nonnegative int64 identities")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("duplicate shot identities")
    if len(first) != len(second):
        raise ValueError("missing or extra shot identities")
    if not np.array_equal(first, second):
        raise ValueError("shot identities differ or are reordered")


def paired_counts(first: NDArray, second: NDArray, first_ids: NDArray, second_ids: NDArray) -> PairedCounts:
    require_paired_ids(first_ids, second_ids)
    if first.shape != first_ids.shape or second.shape != second_ids.shape:
        raise ValueError("failure vectors must provide one result per shot identity")
    return _count_pairs(binary_array(first, name="failures"), binary_array(second, name="failures"))


def _count_pairs(first: NDArray[np.bool_], second: NDArray[np.bool_]) -> PairedCounts:
    return PairedCounts(*map(int, np.bincount(2 * first + second, minlength=4)))


def binomial_interval(count: int, total: int, *, confidence: float, method: str) -> tuple[float, float]:
    if type(count) is not int or type(total) is not int or total < 1 or not 0 <= count <= total:
        raise ValueError("binomial interval requires 0 <= count <= total and total > 0")
    if isinstance(confidence, bool) or not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between zero and one")
    if method not in ("wilson", "exact"):
        raise ValueError("expected Wilson or exact Clopper–Pearson interval")
    interval = binomtest(count, total).proportion_ci(confidence_level=confidence, method=method)
    return float(interval.low), float(interval.high)


def sparse_difference_interval(counts: PairedCounts) -> tuple[float, float]:
    """At least 95% coverage via two 97.5% exact marginal intervals."""
    lower_10, upper_10 = binomial_interval(counts.n10, counts.total, confidence=0.975, method="exact")
    lower_01, upper_01 = binomial_interval(counts.n01, counts.total, confidence=0.975, method="exact")
    return lower_10 - upper_01, upper_10 - lower_01


def bootstrap_counts(counts: PairedCounts, replicates: int, rng: np.random.Generator) -> NDArray[np.int64]:
    if type(replicates) is not int or replicates < 1:
        raise ValueError("bootstrap replicates must be a positive integer")
    return rng.multinomial(counts.total, np.asarray(counts.categories) / counts.total, size=replicates)


@dataclass(frozen=True, slots=True)
class AnalysisPolicy:
    bootstrap_replicates: int
    bootstrap_seed: int
    sparse_rule: int
    ratio_min_denominator_failures: int | None = None

    def __post_init__(self) -> None:
        if type(self.bootstrap_replicates) is not int or self.bootstrap_replicates < 1:
            raise ValueError("bootstrap replicates must be a positive integer")
        if type(self.bootstrap_seed) is not int or not 0 <= self.bootstrap_seed < 2**64:
            raise ValueError("bootstrap seed must be an unsigned 64-bit integer")
        if type(self.sparse_rule) is not int or self.sparse_rule < 0:
            raise ValueError("sparse rule must be a nonnegative inclusive discordance count")
        minimum = self.ratio_min_denominator_failures
        if minimum is not None and (type(minimum) is not int or minimum < 1):
            raise ValueError("ratio minimum denominator failures must be a positive integer")

    def rng(self) -> np.random.Generator:
        return np.random.Generator(np.random.PCG64(self.bootstrap_seed))

    def record(self) -> dict:
        return {
            **asdict(self), "confidence": 0.95, "unit": "whole_trajectory", "ler_interval": "wilson",
            "bootstrap": "four_category_multinomial_percentile", "quantile": "linear", "rng": "numpy.PCG64",
            "sparse_switch": "discordances <= sparse_rule", "sparse_exact_binomial_confidence": 0.975,
            "ratio": "first_rate / second_rate", "zero_bootstrap_denominator": "unresolved",
        }


def _percentile(values: NDArray) -> list[float]:
    return np.quantile(values, [0.025, 0.975], method="linear").tolist()


def summarize_pair(counts: PairedCounts, policy: AnalysisPolicy, rng: np.random.Generator) -> dict:
    minimum = policy.ratio_min_denominator_failures
    denominator_count = counts.n01 + counts.n11
    sparse = counts.discordances <= policy.sparse_rule
    draws = None
    if not sparse or (minimum is not None and denominator_count >= minimum):
        draws = bootstrap_counts(counts, policy.bootstrap_replicates, rng)
    if sparse:
        interval = list(sparse_difference_interval(counts))
        method = "conservative_exact_discordances"
    else:
        interval = _percentile((draws[:, 2] - draws[:, 1]) / counts.total)
        method = "paired_bootstrap_percentile"
    ratio = None
    if minimum is not None:
        ratio = {
            "status": "unresolved", "estimate": None, "interval": None,
            "denominator_failures": denominator_count, "undefined_bootstrap_replicates": None,
        }
        if denominator_count < minimum:
            ratio["reason"] = "insufficient_denominator_failures"
        else:
            denominators = draws[:, 1] + draws[:, 3]
            undefined = int(np.count_nonzero(denominators == 0))
            ratio["undefined_bootstrap_replicates"] = undefined
            if undefined:
                ratio["reason"] = "zero_denominator_in_bootstrap"
            else:
                ratio.update(
                    status="resolved", estimate=(counts.n10 + counts.n11) / denominator_count,
                    interval=_percentile((draws[:, 2] + draws[:, 3]) / denominators),
                )
    return {
        **asdict(counts), "shots": counts.total, "discordance_rate": counts.discordances / counts.total,
        "difference": counts.difference, "interval": interval, "interval_method": method, "ratio": ratio,
    }


def core_disagreements(decisions: NDArray, core_labels: NDArray, global_labels: NDArray) -> NDArray[np.bool_]:
    if decisions.shape != core_labels.shape or global_labels.shape != (len(core_labels),):
        raise ValueError("predictions and labels must have matching trajectory and core dimensions")
    composed = compose_core_decisions(decisions)
    labels = binary_array(core_labels, name="core labels")
    global_bits = binary_array(global_labels, name="global labels")
    if not np.array_equal(compose_core_decisions(labels), global_bits):
        raise ValueError("true core labels disagree with the global label")
    errors = binary_array(decisions, name="core decisions") ^ labels
    if not np.array_equal(compose_core_decisions(errors), composed ^ global_bits):
        raise ValueError("whole-trajectory failure disagrees with core-error parity")
    return errors


class CoreDiagnostics:
    """Accumulate core diagnostics one complete-trajectory shard at a time."""

    def __init__(self):
        self.counts = np.zeros(12, dtype=np.int64)
        self.histogram = np.zeros(13, dtype=np.int64)
        self.adjacent_joint = np.zeros(11, dtype=np.int64)

    def add(self, disagreements: NDArray[np.bool_]) -> None:
        if disagreements.ndim != 2 or disagreements.shape[1] != len(self.counts):
            raise ValueError("core diagnostics require the complete core partition")
        errors = binary_array(disagreements, name="core disagreements")
        self.counts += errors.sum(axis=0)
        self.histogram += np.bincount(errors.sum(axis=1), minlength=len(self.histogram))
        self.adjacent_joint += (errors[:, :-1] & errors[:, 1:]).sum(axis=0)

    def report(self) -> dict:
        shots = int(self.histogram.sum())
        if not shots:
            raise ValueError("core diagnostics require trajectories")
        rates = self.counts / shots
        left, right = rates[:-1], rates[1:]
        denominator = np.sqrt(left * (1 - left) * right * (1 - right))
        covariance = self.adjacent_joint / shots - left * right
        return {
            "per_core_counts": self.counts.tolist(), "per_core_rates": rates.tolist(),
            "disagreement_count_histogram": self.histogram.tolist(),
            "odd_disagreement_rate": float(self.histogram[1::2].sum() / shots),
            "adjacent_joint_counts": self.adjacent_joint.tolist(),
            "adjacent_correlations": [float(c / d) if d else None for c, d in zip(covariance, denominator, strict=True)],
            "position_rates": {"first": float(rates[0]), "interior": float(rates[1:-1].mean()), "last": float(rates[-1])},
        }
