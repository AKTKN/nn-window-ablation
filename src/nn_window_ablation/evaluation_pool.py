"""Linux spawn supervision and bounded scheduling for frozen evaluation."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
import ctypes
import multiprocessing as mp
from multiprocessing.util import Finalize
import os
from pathlib import Path
import resource
import signal
from threading import current_thread, main_thread
from time import monotonic, perf_counter

from .artifacts import ArtifactError, canonical_json, open_writer_lease
from .runtime import native_thread_limits


def append_event(path: Path, kind: str, **record) -> None:
    with path.open("ab") as file:
        file.write(canonical_json({"event": kind, "time": perf_counter(), "pid": os.getpid(), **record}) + b"\n")


def observe_process() -> dict:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    try:
        fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
        start_ticks, native_threads = int(fields[19]), int(fields[17])
        rss_bytes = int(fields[21]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        start_ticks = native_threads = rss_bytes = None
    try:
        io = {key: int(value) for key, value in (
            line.split(":") for line in Path("/proc/self/io").read_text().splitlines()
        )}
    except (OSError, ValueError):
        io = None
    try:
        fd_count = len(os.listdir("/proc/self/fd"))
    except OSError:
        fd_count = None
    return {
        "pid": os.getpid(), "start_ticks": start_ticks, "rss_bytes": rss_bytes, "native_threads": native_threads,
        "cpu_user_seconds": usage.ru_utime, "cpu_system_seconds": usage.ru_stime,
        "proc_io": io, "fd_count": fd_count,
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    }


@contextmanager
def stop_on_signal(operation: str = "evaluation"):
    """Defer termination to transaction/scheduling boundaries in the main thread."""
    requested = None

    def note(signum, _frame):
        nonlocal requested
        requested = signum

    def check():
        if requested is not None:
            raise ArtifactError(f"{operation} interrupted by {signal.Signals(requested).name}")

    old = {}
    try:
        if current_thread() is main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                old[signum] = signal.signal(signum, note)
        yield check
    finally:
        for signum, handler in old.items():
            signal.signal(signum, handler)


@contextmanager
def cpu_affinity(cpus):
    original = os.sched_getaffinity(0)
    os.sched_setaffinity(0, cpus)
    try:
        yield
    finally:
        os.sched_setaffinity(0, original)


_writer_lease = None


def bind_to_parent(parent_pid: int) -> None:
    # PR_SET_PDEATHSIG. Checking PPID afterwards covers death before prctl.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0):
        raise OSError(ctypes.get_errno(), "setting worker parent-death signal")
    if os.getppid() != parent_pid:
        os._exit(1)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def _initialize(parent_pid, lease, log_dir, slots, next_slot, initialize, initializer_args):
    global _writer_lease
    start = perf_counter()
    bind_to_parent(parent_pid)
    _writer_lease = open_writer_lease(lease, log_dir.name)

    with next_slot.get_lock():
        affinity = slots[next_slot.value]
        next_slot.value += 1
    if not set(affinity) <= os.sched_getaffinity(0):
        raise ArtifactError("evaluation worker CPUs are outside inherited process affinity")
    os.sched_setaffinity(0, affinity)

    log = log_dir / f"worker-{os.getpid()}.jsonl"
    Finalize(None, _record_worker_exit, (log,), exitpriority=0)
    append_event(log, "worker_enter", process=observe_process())
    try:
        details = initialize(*initializer_args)
    except Exception as exc:
        append_event(log, "worker_initialization_failed", error=str(exc))
        raise
    append_event(log, "worker_ready", wall_seconds=perf_counter() - start, process=observe_process(), **details)


def _record_worker_exit(log: Path):
    append_event(log, "worker_exit", process=observe_process())


def _finish_pool(pool: ProcessPoolExecutor, *, abort: bool) -> None:
    # Python 3.11 exposes neither the manager's join timeout nor worker
    # termination on the public executor API. Retain handles before shutdown.
    manager = pool._executor_manager_thread
    processes = tuple(pool._processes.values())
    pool.shutdown(wait=False, cancel_futures=True)

    if abort or manager is None:
        for process in processes:
            process.terminate()
    if manager is not None:
        manager.join(timeout=5)
        if not manager.is_alive():
            return

    deadline = monotonic() + 5
    for process in processes:
        if process.exitcode is None:
            process.kill()
    for process in processes:
        process.join(timeout=max(0, deadline - monotonic()))
    if manager is not None:
        manager.join(timeout=max(0, deadline - monotonic()))
    if any(process.exitcode is None for process in processes) or (manager is not None and manager.is_alive()):
        raise ArtifactError("evaluation pool could not stop within 10 seconds; worker leases protect unfinished staging")
    if not abort:
        raise ArtifactError("evaluation pool did not shut down cleanly")


def run_pool(
    tasks, *, slots, threads, limit, lease, log_dir, initialize, initializer_args, work,
    check_stop, on_submit, on_result, operation="evaluation",
) -> None:
    """Own at most limit futures through coordinator verification and commit."""
    context = mp.get_context("spawn")
    pool = ProcessPoolExecutor(
        max_workers=len(slots), mp_context=context, initializer=_initialize,
        initargs=(os.getpid(), lease, log_dir, slots, context.Value("i", 0), initialize, initializer_args),
    )
    pending = {}
    source = iter(tasks)

    def submit_next():
        check_stop()
        task = next(source, None)
        if task is None:
            return False
        submitted = perf_counter()
        try:
            future = pool.submit(work, task)
        except Exception as exc:
            raise ArtifactError(f"could not submit {operation} task {task}: {exc}") from exc
        pending[future] = (task, submitted)
        on_submit(task, len(pending))
        return True

    abort = True
    try:
        with native_thread_limits(threads):
            for _ in range(limit):
                if not submit_next():
                    break
            while pending:
                check_stop()
                ready = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED).done
                while ready:
                    check_stop()
                    future = ready.pop()
                    task, submitted = pending.pop(future)
                    # Check a completed result only. A broken pool can make the
                    # outcomes of several outstanding tasks unknown.
                    try:
                        result = future.result()
                    except BrokenProcessPool as exc:
                        outstanding = sorted([task.index, *(item.index for item, _ in pending.values())])
                        raise ArtifactError(
                            f"{operation} pool failed with {task} and outstanding shard indices {outstanding}: {exc}"
                        ) from exc
                    except Exception as exc:
                        raise ArtifactError(f"{operation} task {task} failed: {exc}") from exc
                    on_result(task, result, submitted, len(pending) + 1)
                    del result, future
                    submit_next()
        abort = False
    finally:
        started = perf_counter()
        _finish_pool(pool, abort=abort)
        append_event(
            log_dir / "coordinator.jsonl", "pool_shutdown", aborted=abort, wall_seconds=perf_counter() - started,
        )
