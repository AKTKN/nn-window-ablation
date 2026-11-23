"""Linux process-group measurement for the local multicore audit."""

import ctypes
import json
import os
from pathlib import Path
import resource
import signal
import stat
import subprocess
import sys
from time import monotonic, sleep


def _subreaper():
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), "set audit child subreaper")


def _io(pid):
    try:
        return {key: int(value) for key, value in (
            row.split(":") for row in Path(f"/proc/{pid}/io").read_text().splitlines()
        )}
    except (OSError, ValueError):
        return None


def _cpu_usage(who):
    usage = resource.getrusage(who)
    return {"user": usage.ru_utime, "system": usage.ru_stime}


def _record(path, record):
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def _stage_wrapper(receipt, argv):
    # The stage root reaps the production CLI and any resource tracker it
    # adopts, then captures its inclusive I/O counter exactly once.
    _subreaper()
    process = subprocess.Popen(argv)
    exit_code = process.wait()
    while True:
        try:
            os.waitpid(-1, 0)
        except ChildProcessError:
            break
    totals = {
        "exit_code": exit_code, "root_io_inclusive": _io("self"),
        "root_self_cpu": _cpu_usage(resource.RUSAGE_SELF), "waited_children_cpu": _cpu_usage(resource.RUSAGE_CHILDREN),
    }
    _record(receipt, totals)
    return exit_code


def _members(group):
    page_size = os.sysconf("SC_PAGE_SIZE")
    processes = []
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) != group:
                continue
            pid = int(entry.name)
            processes.append({
                "pid": pid, "ppid": int(fields[1]), "start_ticks": int(fields[19]), "state": fields[0],
                "rss_bytes": int(fields[21]) * page_size, "native_threads": int(fields[17]),
                "self_user_ticks": int(fields[11]), "self_system_ticks": int(fields[12]),
                "affinity": sorted(os.sched_getaffinity(pid)), "fds": len(os.listdir(entry / "fd")),
            })
        except (OSError, IndexError, ValueError):
            continue
    return processes


def _files(roots):
    files, size = 0, 0
    for root in roots:
        # Workers atomically rename staging directories while this scan runs.
        # os.walk tolerates directories disappearing between list and descent.
        paths = (Path(directory) / name for directory, _, names in os.walk(root) for name in names) if root.is_dir() else (root,)
        for file in paths:
            try:
                metadata = file.stat(follow_symlinks=False)
            except OSError:
                continue
            if stat.S_ISREG(metadata.st_mode):
                size += metadata.st_size
                files += 1
    return {"files": files, "bytes": size}


def _anonymous_files(processes, roots):
    # SpooledTemporaryFile can use unnamed files inside the run directory.
    found = {}
    marker = tuple(f"{path}/" for path in roots)
    for member in processes:
        for fd in Path(f"/proc/{member['pid']}/fd").glob("*"):
            try:
                target = os.readlink(fd)
                if not target.startswith(marker) or not target.endswith(" (deleted)"):
                    continue
                report = fd.stat()
                if stat.S_ISREG(report.st_mode):
                    found[report.st_dev, report.st_ino] = report.st_size
            except OSError:
                continue
    return {"files": len(found), "bytes": sum(found.values())}


def _atomic_files(roots):
    files = (file for root in roots if root.is_dir() for pattern in (".checkpoint.pt.*", ".run.json.*")
             for file in root.glob(pattern))
    return _files(files)


def _stop_group(process):
    for signum, seconds in ((signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            break
        deadline = monotonic() + seconds
        while monotonic() < deadline:
            process.poll()
            _reap_group(process.pid)
            if not _members(process.pid):
                return
            sleep(0.1)
    process.wait(timeout=2)
    _reap_group(process.pid)


def _reap_group(group):
    try:
        while os.waitpid(-group, os.WNOHANG)[0]:
            pass
    except ChildProcessError:
        pass


def measure(argv, directory, cpus, roots=(), temporary=(), *, interval=0.2, timeout=180,
            memory_limit=6 * 2**30, disk_reserve=4 * 2**30):
    """Monitor one fresh, affinity-confined command and stream its timeline."""
    directory.mkdir(parents=True)
    roots, temporary = tuple(roots), tuple(temporary)
    _record(directory / "command.json", {"command": argv, "cpus": cpus, "timeout_seconds": timeout})
    terminal = directory / "terminal.json"
    restricted = sorted(cpus)
    previous_affinity = os.sched_getaffinity(0)
    own_start = _cpu_usage(resource.RUSAGE_SELF)
    start = monotonic()
    peaks = {name: 0 for name in ("rss_bytes", "files", "bytes", "temp_files", "temp_bytes", "processes", "fds")}
    reason = None
    os.sched_setaffinity(0, restricted)
    try:
        with (directory / "stdout.log").open("w") as stdout, (directory / "timeline.jsonl").open("w") as log:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--stage", str(terminal), *argv],
                stdout=stdout, stderr=subprocess.STDOUT, start_new_session=True,
            )
            try:
                while process.poll() is None:
                    now = monotonic() - start
                    members = _members(process.pid)
                    rss = sum(p["rss_bytes"] for p in members)
                    output = _files(roots)
                    stage = _files(temporary)
                    anon = _anonymous_files(members, (*roots, *temporary))
                    atomic = _atomic_files(roots)
                    row = {
                        "rss_bytes": rss, "files": output["files"], "bytes": output["bytes"],
                        "temp_files": stage["files"] + anon["files"] + atomic["files"],
                        "temp_bytes": stage["bytes"] + anon["bytes"] + atomic["bytes"],
                        "processes": len(members), "fds": sum(p["fds"] for p in members),
                    }
                    peaks = {name: max(previous, row[name]) for name, previous in peaks.items()}
                    print(json.dumps(row | {"seconds": now, "members": members}), file=log)
                    usage = os.statvfs(directory)
                    if now > timeout or rss > memory_limit or usage.f_bavail * usage.f_frsize < disk_reserve:
                        reason = f"stage limit exceeded at {now:.2f}s, RSS {rss}, free bytes {usage.f_bavail * usage.f_frsize}"
                        break
                    sleep(interval)
            except OSError as exc:
                reason = f"monitor sampling failed: {exc}"
            finally:
                if process.poll() is None or _members(process.pid):
                    _stop_group(process)
    finally:
        os.sched_setaffinity(0, previous_affinity)
    own_end = _cpu_usage(resource.RUSAGE_SELF)
    stats = json.loads(terminal.read_text()) if terminal.is_file() else None
    result = {
        "exit_code": process.returncode, "wall_seconds": monotonic() - start, "cpu_ids": restricted,
        "sample_interval_seconds": interval, "sampled_peaks": peaks, "remaining": _members(process.pid),
        "monitor_cpu": {key: own_end[key] - own_start[key] for key in own_start},
        "terminal": stats, "failure": reason, "retained": _files(roots),
    }
    _record(directory / "result.json", result)
    return result


if __name__ == "__main__" and sys.argv[1] == "--stage":
    raise SystemExit(_stage_wrapper(Path(sys.argv[2]), sys.argv[3:]))
