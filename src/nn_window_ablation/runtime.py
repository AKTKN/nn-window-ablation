"""CPU process startup for workflows that perform numerical work."""

import os

from .config import ConfigError, DataConfig, RuntimeConfig, validate_cpu_budget


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
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
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
