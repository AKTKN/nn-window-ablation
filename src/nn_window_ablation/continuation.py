"""Portable continuation controls and a bounded audit of committed updates."""

from dataclasses import asdict
from hashlib import sha256
from math import isfinite

from .artifacts import ArtifactError, canonical_json


TRAINING_REGIMES = {
    "full_closed": ("base", "random", "full_closed", "S"),
    "continued_full": ("continuation", "full_checkpoint", "full_closed", "S+"),
    "fixed_window": ("continuation", "full_checkpoint", "deployment_window", "W"),
    "dynamic_window": ("continuation", "full_checkpoint", "sampled_window", "D"),
}
CONTINUATION_ARMS = frozenset(name for name, regime in TRAINING_REGIMES.items() if regime[0] == "continuation")
TRACE_PREFIX_UPDATES = 16
EMPTY_INPUT_SHA256 = sha256(b"nnwa-continuation-input-v1").hexdigest()


def model_tensors_sha256(state: dict) -> str:
    """Hash model state tensor names, dtypes, shapes and exact CPU bytes."""
    digest = sha256()
    for name, tensor in sorted(state.items()):
        digest.update(canonical_json([name, str(tensor.dtype), list(tensor.shape)]) + b"\n")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def is_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def next_input_digest(previous: str, update: dict) -> str:
    identity = [update[field] for field in ("update", "trajectories", "cores")]
    return sha256(bytes.fromhex(previous) + canonical_json(identity)).hexdigest()


def start_audit(controls: dict) -> dict:
    return {
        "schema_version": 1, "controls": controls, "controls_sha256": sha256(canonical_json(controls)).hexdigest(),
        "input_sha256": EMPTY_INPUT_SHA256, "updates": 0, "trace_prefix": [], "validation": None,
        "resources": {
            "detector_tokens": 0, "step_wall_seconds": 0.0, "process_cpu_seconds": 0.0,
            "allocated_cpu_core_hours": 0.0, "peak_process_rss_kib": 0,
        },
    }


def record_update(
    audit: dict, trace, *, detector_tokens: int, wall: float, cpu: float, rss: int, allocated_cpus: int,
) -> None:
    update = asdict(trace)
    if update["update"] != audit["updates"] + 1:
        raise ValueError("continuation audit must record consecutive updates")
    audit["updates"] = update["update"]
    audit["input_sha256"] = next_input_digest(audit["input_sha256"], update)
    if len(audit["trace_prefix"]) < TRACE_PREFIX_UPDATES:
        audit["trace_prefix"].append(update)
    usage = audit["resources"]
    usage["detector_tokens"] += detector_tokens
    usage["step_wall_seconds"] += wall
    usage["process_cpu_seconds"] += cpu
    usage["allocated_cpu_core_hours"] = usage["step_wall_seconds"] * allocated_cpus / 3600
    usage["peak_process_rss_kib"] = max(usage["peak_process_rss_kib"], rss)


def checked_audit(audit: dict, *, updates: int) -> dict:
    """Validate the committed audit envelope before resume or paired analysis."""
    try:
        controls = audit["controls"]
        prefix = audit["trace_prefix"]
        resources = audit["resources"]
        if (
            audit["schema_version"] != 1 or audit["updates"] != updates
            or len(prefix) != min(updates, TRACE_PREFIX_UPDATES)
            or sha256(canonical_json(controls)).hexdigest() != audit["controls_sha256"]
            or not is_digest(audit["input_sha256"]) or not is_digest(controls["initial_model_sha256"])
            or not is_digest(controls["parent_sha256"])
            or any(type(value) not in (int, float) or not isfinite(value) or value < 0 for value in resources.values())
        ):
            raise ArtifactError("invalid continuation audit or control identity")
        digest = EMPTY_INPUT_SHA256
        batch = controls["trainer"]["budget"]["batch_size"]
        for index, row in enumerate(prefix, 1):
            if row["update"] != index or len(row["trajectories"]) != batch or len(row["cores"]) != batch:
                raise ArtifactError("continuation audit contains a missing or invalid minibatch")
            digest = next_input_digest(digest, row)
        if updates <= TRACE_PREFIX_UPDATES and digest != audit["input_sha256"]:
            raise ArtifactError("continuation audit input digest disagrees with its trajectory/core trace")
        validation = audit["validation"]
        if validation is not None:
            planned = controls["validation"]
            shots, failures = validation["shots"], validation["failures"]
            if (
                planned is None or validation["phase_update"] != updates or updates != planned["phase_update"]
                or type(shots) is not int or shots != planned["shots"] or not shots
                or type(failures) is not int or not 0 <= failures <= shots
                or validation["ler"] != failures / shots
            ):
                raise ArtifactError("continuation validation is inconsistent with the terminal schedule")
    except ArtifactError:
        raise
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise ArtifactError("incomplete continuation audit") from exc
    return audit
