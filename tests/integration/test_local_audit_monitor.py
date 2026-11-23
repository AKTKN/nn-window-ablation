import json
import os
from pathlib import Path
import sys

import pytest

from scripts.audit_monitor import _io, measure


@pytest.mark.resource_process
def test_monitor_observes_reaped_and_live_children_without_double_counting(tmp_path):
    child = """
import os, sys, time
allocation = bytearray(28 * 2**20)
allocation[::4096] = b'x' * (len(allocation) // 4096)
time.sleep(0.35)
with open(sys.argv[1], 'wb') as out:
    out.write(b'y' * 3_000_000)
    os.fsync(out.fileno())
start = time.process_time()
while time.process_time() - start < 0.15:
    pass
"""
    parent = """
import subprocess, sys, time
subprocess.run([sys.executable, '-c', sys.argv[1], sys.argv[2]], check=True)
time.sleep(0.3)
"""
    receipt = measure(
        [sys.executable, "-c", parent, child, str(tmp_path / "payload")], tmp_path / "measurement",
        sorted(os.sched_getaffinity(0))[:1], (tmp_path / "payload",), timeout=15, interval=0.05,
    )
    assert receipt["exit_code"] == 0 and not receipt["remaining"]
    assert receipt["retained"] == {"files": 1, "bytes": 3_000_000}
    terminal = receipt["terminal"]
    assert 3_000_000 <= terminal["root_io_inclusive"]["wchar"] < 3_065_536
    assert 0.15 <= sum(terminal["waited_children_cpu"].values()) < 2
    timeline = Path(tmp_path / "measurement/timeline.jsonl")
    rows = [json.loads(line) for line in timeline.read_text().splitlines()]
    assert max(row["processes"] for row in rows) >= 3
    assert any(row["processes"] <= 2 for row in rows[3:])
    assert max(row["rss_bytes"] for row in rows) - min(row["rss_bytes"] for row in rows[3:]) >= 20 * 2**20
    assert all(proc["start_ticks"] > 0 for row in rows for proc in row["members"])
    assert _io(1_000_000_000) is None


@pytest.mark.resource_process
def test_monitor_tolerates_concurrent_staging_renames(tmp_path):
    staging = tmp_path / "active"
    staging.mkdir()
    child = """
import os, sys
from pathlib import Path
root = Path(sys.argv[1])
for i in range(500):
    pending = root / '.pending'
    pending.mkdir()
    (pending / 'array').write_bytes(b'0' * 1024)
    published = root / 'shard'
    os.rename(pending, published)
    (published / 'array').unlink()
    published.rmdir()
"""
    result = measure(
        [sys.executable, "-c", child, str(staging)], tmp_path / "measurement",
        sorted(os.sched_getaffinity(0))[:1], temporary=(staging,), timeout=10, interval=0.01,
    )
    assert result["exit_code"] == 0 and result["failure"] is None and not result["remaining"]


@pytest.mark.resource_process
def test_monitor_observes_and_releases_unnamed_stage_spool(tmp_path):
    child = """
from pathlib import Path
from tempfile import TemporaryFile, gettempdir
from time import sleep

Path(gettempdir(), 'runtime-cache').write_bytes(b'cache')
with TemporaryFile() as spool:
    spool.write(b'x' * 3_000_000)
    spool.flush()
    sleep(0.5)
sleep(0.25)
"""
    case = tmp_path / "spool"
    result = measure([sys.executable, "-c", child], case, sorted(os.sched_getaffinity(0))[:1],
                     timeout=15, interval=0.05)
    assert result["exit_code"] == 0 and not result["remaining"]
    assert result["sampled_peaks"]["scratch_bytes"] == 3_000_005
    assert result["sampled_peaks"]["scratch_files"] == 2
    assert result["scratch_after_command"] == {"bytes": 5, "files": 1}
    assert result["scratch_retained"] == {"bytes": 0, "files": 0}
    scratch = [row["scratch_bytes"] for row in map(json.loads, (case / "timeline.jsonl").read_text().splitlines())]
    assert 5 in scratch[scratch.index(3_000_005) + 1:]
