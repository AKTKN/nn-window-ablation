"""Launch and supervise a single-node Linux CPU/Gloo training group."""

from dataclasses import replace
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import monotonic, perf_counter
import traceback

from .artifacts import ArtifactError
from .config import ConfigError, ResolvedConfig, validate_cpu_budget
from .evaluation_pool import bind_to_parent, cpu_affinity, stop_on_signal
from .runtime import configure_cpu_runtime, native_thread_limits, reject_excessive_native_threads


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


def training_placement(config: ResolvedConfig) -> tuple[tuple[int, ...], ...]:
    if not sys.platform.startswith("linux") or not hasattr(os, "sched_getaffinity"):
        raise ConfigError("CPU DDP requires Linux process affinity and parent-death signals")
    required = validate_cpu_budget(config.runtime, config.data)
    affinity = sorted(os.sched_getaffinity(0))
    allocated = config.runtime.allocated_cpus
    if allocated is None or required > allocated or allocated > len(affinity):
        raise ConfigError(f"DDP allocation of {allocated} CPUs does not fit effective process affinity ({len(affinity)})")
    quota = _cpu_quota()
    if quota is not None and required > quota:
        raise ConfigError(f"DDP requires {required} CPUs, exceeding visible cgroup CPU quota {quota:g}")
    threads = config.runtime.intraop_threads
    return tuple(tuple(affinity[i:i + threads]) for i in range(0, required, threads))


def _rank_train(config, group, args, started, startup):
    from .execution import _run_training

    _run_training(config, started=started, group=group, startup=startup, **args)


def _rank_entry(config, rank, placement, rendezvous, peer_receipt, parent, launched, target, args):
    try:
        _run_rank(config, rank, placement, rendezvous, peer_receipt, parent, launched, target, args)
    except BaseException:
        traceback.print_exc()
        code = 1
    else:
        code = 0
    # Workflows have already closed their readers, locks, and Gloo group.
    # Avoid waiting on third-party native shutdown hooks in the interpreter.
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(code)


def _run_rank(config, rank, placement, rendezvous, peer_receipt, parent, launched, target, args):
    bind_to_parent(parent)
    started = perf_counter()
    cpus = placement[rank]
    if not set(cpus) <= os.sched_getaffinity(0):
        raise ConfigError(f"DDP rank {rank} CPUs are outside inherited affinity")
    os.sched_setaffinity(0, cpus)
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    runtime = replace(config.runtime, processes=1, allocated_cpus=len(cpus))
    configure_cpu_runtime(runtime, config.data)

    import torch

    from .distributed_training import TrainingGroup

    torch.use_deterministic_algorithms(True)
    ready = perf_counter()
    with TrainingGroup(rank, placement, rendezvous, peer_receipt, config.training.ddp_timeout_seconds, parent, launched) as group:
        target(config, group, args, started, {"runtime": ready - started, "rendezvous": perf_counter() - ready})


def _stop_ranks(processes) -> None:
    for process in processes:
        if process.exitcode is None:
            process.terminate()
    for signal_kill in (False, True):
        deadline = monotonic() + 5
        for process in processes:
            if signal_kill and process.exitcode is None:
                process.kill()
            process.join(timeout=max(0, deadline - monotonic()))
        if all(process.exitcode is not None for process in processes):
            return
    raise ArtifactError("DDP ranks failed to exit after termination and kill")


def launch_training(config: ResolvedConfig, **args) -> None:
    """Supervise rank failures and reap processes before this command exits."""
    placement = training_placement(config)
    reject_excessive_native_threads(config.runtime.intraop_threads, "DDP rank")
    context = mp.get_context("spawn")
    started = perf_counter()
    processes = {}
    with TemporaryDirectory(prefix="nnwa-gloo-") as scratch, stop_on_signal("training") as check_stop:
        store = (Path(scratch) / "rendezvous").as_uri()
        peer_receipt = Path(scratch) / "peers-exited"
        try:
            with cpu_affinity(set().union(*placement)), native_thread_limits(config.runtime.intraop_threads):
                for rank in range(len(placement)):
                    check_stop()
                    process = context.Process(
                        name=f"nnwa-rank-{rank}", target=_rank_entry,
                        args=(config, rank, placement, store, peer_receipt, os.getpid(), started, _rank_train, args),
                    )
                    process.start()
                    processes[process.sentinel] = (rank, process)
            pending = set(processes)
            peers = {sentinel for sentinel, (rank, _) in processes.items() if rank != 0}
            if not peers:
                peer_receipt.touch()
            while pending:
                check_stop()
                for sentinel in wait(pending, timeout=0.2):
                    rank, process = processes[sentinel]
                    process.join(timeout=0.1)
                    if process.exitcode is None:
                        continue
                    if process.exitcode != 0:
                        raise ArtifactError(f"CPU DDP rank {rank} (PID {process.pid}) exited with code {process.exitcode}")
                    pending.remove(sentinel)
                    if sentinel in peers:
                        peers.remove(sentinel)
                        if not peers:
                            peer_receipt.touch()
        finally:
            _stop_ranks([process for _, process in processes.values()])
