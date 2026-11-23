"""CPU process startup for workflows that perform numerical work."""

import os
from contextlib import contextmanager
from pathlib import Path

from .config import ConfigError, DataConfig, RuntimeConfig, validate_cpu_budget


NATIVE_THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)


def reject_excessive_native_threads(threads: int, operation: str) -> None:
    for name in NATIVE_THREAD_VARIABLES:
        inherited = os.environ.get(name)
        if inherited is not None and (
            not inherited.isascii() or not inherited.isdecimal() or not 1 <= int(inherited) <= threads
        ):
            raise ConfigError(f"inherited {name}={inherited!r} exceeds or conflicts with {operation} threads={threads}")


@contextmanager
def native_thread_limits(threads: int):
    """Set the environment inherited by spawned processes."""
    prior = {name: os.environ.get(name) for name in NATIVE_THREAD_VARIABLES}
    try:
        for name in prior:
            os.environ[name] = str(threads)
        yield
    finally:
        for name, value in prior.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def runtime_observation(data: DataConfig) -> dict:
    import torch

    return {
        "torch_intraop": torch.get_num_threads(), "torch_interop": torch.get_num_interop_threads(),
        "data_loader_workers": data.num_workers, "sampling_processes": data.sampling_processes,
    }


def configure_cpu_runtime(runtime: RuntimeConfig, data: DataConfig) -> None:
    """Apply thread limits before numerical work in an application process."""
    required = validate_cpu_budget(runtime, data)
    if runtime.allocated_cpus is None:
        raise ConfigError("runtime.allocated_cpus is required for numerical execution")
    if hasattr(os, "sched_getaffinity") and required > len(os.sched_getaffinity(0)):
        raise ConfigError(f"runtime requires {required} CPUs, exceeding process CPU affinity")

    # Set native-library limits before PyTorch or its numerical dependencies load.
    for variable in NATIVE_THREAD_VARIABLES:
        os.environ[variable] = str(runtime.intraop_threads)

    import torch

    if torch.get_num_interop_threads() != runtime.interop_threads:
        try:
            torch.set_num_interop_threads(runtime.interop_threads)
        except RuntimeError as exc:
            if "cannot set number of interop threads" not in str(exc):
                raise
            raise ConfigError(
                f"runtime.interop_threads={runtime.interop_threads} cannot be applied after "
                f"PyTorch inter-op initialization at {torch.get_num_interop_threads()}; "
                "use a fresh process for different inter-op settings"
            ) from exc
    if torch.get_num_threads() != runtime.intraop_threads:
        torch.set_num_threads(runtime.intraop_threads)


def _cpu_quota() -> float | None:
    """Return the tightest visible cgroup-v2 CPU bandwidth quota, in CPUs."""
    mount = Path("/sys/fs/cgroup")
    try:
        name = next(
            line.removeprefix("0::").lstrip("/")
            for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::")
        )
    except (OSError, StopIteration):
        return None
    quotas = []
    path = mount / name
    for directory in (path, *path.parents):
        if not directory.is_relative_to(mount):
            break
        try:
            limit, period = (directory / "cpu.max").read_text().split()
        except (OSError, ValueError):
            continue
        if limit.isdecimal() and period.isdecimal() and int(period) > 0:
            quotas.append(int(limit) / int(period))
    return min(quotas, default=None)
