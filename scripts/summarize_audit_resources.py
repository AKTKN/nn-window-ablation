"""Verify recorded process lifecycles and summarize production audit spans."""

import argparse
import json
import os
from pathlib import Path
from statistics import median

from scripts.audit_monitor import _record, _subreaper
from scripts.local_multicore_audit import Audit, THREAD_ENV, load, sha


def events(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def _series(values):
    return {"count": len(values), "median": median(values), "range": [min(values), max(values)]} if values else None


def _one(rows, name):
    event, = [row for row in rows if row["event"] == name]
    return event


def _spans(rows):
    totals = {}
    for row in rows:
        for span, seconds in row.get("timings", {}).items():
            totals[span] = totals.get(span, 0.0) + seconds
    return totals


def _identity(snapshot):
    return snapshot["pid"], snapshot["start_ticks"]


def _process(ready, end, cpus):
    start, stop = ready["process"], end["process"]
    assert _identity(start) == _identity(stop)
    assert set(start["affinity"]) <= set(cpus) and set(stop["affinity"]) <= set(cpus)
    return {
        "ready": start, "terminal": stop, "initialization_seconds": ready.get("wall_seconds"),
        "seconds_after_ready": end["time"] - ready["time"], "initialization_spans": ready.get("timings"),
    }


def _attempt(base, filename):
    path, = list(base.glob(f"*/{filename}"))
    return path


def _pool(base, expected_tasks, cpus, *, evaluation):
    path = _attempt(base, "coordinator.jsonl")
    coordinator = events(path)
    start = _one(coordinator, "attempt_start")
    end = _one(coordinator, "attempt_end")
    assert end["status"] == "complete"
    commits = [row for row in coordinator if row["event"] == ("task_committed" if evaluation else "shard_verified")]
    assert sorted(row["index"] for row in commits) == list(range(expected_tasks))
    assert commits[-1]["pending"] == 0
    pending = max(row["pending"] for row in coordinator if row["event"] == "task_submitted")
    options = start if evaluation else start["runtime"]
    assert pending <= options["max_in_flight"]
    count = options["workers" if evaluation else "sampling_processes"]
    assert count == len(cpus) and options["worker_threads" if evaluation else "sampling_threads"] == 1
    logs = sorted(path.parent.glob("worker-*.jsonl"))
    assert len(logs) == (min(count, expected_tasks) if count > 1 else 0)
    tasks, workers = [], []
    for worker in logs:
        rows = events(worker)
        ready, stop = _one(rows, "worker_ready"), _one(rows, "worker_exit")
        assert all(ready["native_env"].get(key) == "1" for key in THREAD_ENV)
        if evaluation:
            assert ready["model_loads"] == ready["runtime"]["torch_intraop"] == ready["runtime"]["torch_interop"] == 1
        else:
            assert ready["circuit_builds"] == ready["native_threads"] == 1
        completed = [row for row in rows if row["event"] == "task_done"]
        tasks.extend(completed)
        samples = [row["after"] if evaluation else row["process"] for row in completed]
        workers.append(_process(ready, stop, cpus) | {
            "tasks": len(samples), "task_seconds": _series([row["wall_seconds"] for row in completed]),
            "post_task_fd_samples": [row["fd_count"] for row in samples],
            "post_task_rss_bytes": [row["rss_bytes"] for row in samples],
            "native_env": ready["native_env"], "runtime": ready.get("runtime"),
        })
    if not logs:
        tasks = [row for row in coordinator if row["event"] == "task_done"]
        if evaluation:
            assert _one(coordinator, "inline_model_ready")["model_loads"] == 1
    else:
        affinities = [tuple(worker["ready"]["affinity"]) for worker in workers]
        assert len(set(affinities)) == len(logs) and all(len(mask) == 1 for mask in affinities)
        assert not _one(coordinator, "pool_shutdown")["aborted"]
    assert sorted(row["index"] for row in tasks) == list(range(expected_tasks))
    max_maps = max(row["active_mappings"] for row in (tasks if evaluation else commits))
    assert max_maps <= 4
    if evaluation:
        assert all(row["dataset_active_after"] is False and row["before"]["fd_count"] == row["after"]["fd_count"]
                   for row in tasks)
    return {
        "attempt_log": str(path), "coordinator": _process(start, end, cpus), "workers": workers,
        "runtime": options if not evaluation else {key: options[key] for key in (
            "workers", "worker_threads", "worker_affinity", "coordinator_commit_cpu", "max_in_flight",
        )},
        "pending_max": pending, "active_mappings_max_per_process": max_maps, "committed_tasks": expected_tasks,
        "first_to_last_commit_seconds": commits[-1]["time"] - commits[0]["time"],
        "task_seconds": _series([row["wall_seconds"] for row in tasks]),
        "summed_task_spans": _spans(tasks), "coordinator_spans": _spans(coordinator),
        "commit_verification_seconds": _series([row["verification_seconds"] for row in commits]),
        "queue_seconds": _series([row["queue_seconds"] for row in commits]),
        "publication_seconds": _one(coordinator, "manifest_published")["wall_seconds"],
        "shutdown_seconds": _one(coordinator, "pool_shutdown")["wall_seconds"] if logs else None,
        "scientific_shard_input_bytes": sum(row["source_bytes"] for row in tasks) if evaluation else None,
        "scientific_shard_output_bytes": sum(row["output_bytes"] for row in (tasks if evaluation else commits)),
    }


def _train(path, updates, phase, interval, cpus, distributed, threads):
    log = _attempt(path.parent / "train-attempts", "rank-0.jsonl")
    runtime = load(path.with_name("run.json"))["run"]["runtime"]
    assert runtime["torch_interop"] == 1 and runtime["torch_intraop"] == threads
    world = len(cpus) if distributed else 1
    assert runtime.get("world_size", 1) == world
    saves = [0, *range(interval, updates, interval), updates]
    if phase is not None:
        saves.append(updates)  # The validated terminal snapshot is published after the group closes.
    ranks, by_update = [], [[] for _ in range(updates)]
    for rank in range(world):
        rows = events(log.with_name(f"rank-{rank}.jsonl"))
        start, stop = _one(rows, "training_start"), _one(rows, "training_end")
        assert start["world_size"] == world and start["rank"] == stop["rank"] == rank
        if distributed:
            assert len(start["process"]["affinity"]) == 1
            _one(rows, "group_closed")
        steps = [row for row in rows if row["event"] == "update"]
        assert [row["update"] for row in steps] == list(range(1, updates + 1))
        for index, row in enumerate(steps):
            by_update[index].append(row["wall_seconds"])
        checkpoints = [row for row in rows if row["event"] == ("checkpoint_peer" if rank else "checkpoint")]
        assert [row["phase_updates"] for row in checkpoints] == (saves[:-1] if rank and phase else saves)
        if rank == 0:
            assert checkpoints[-1]["bytes"] == path.stat().st_size
            if distributed:
                _one(rows, "peer_exits_verified")
            publications = checkpoints
        ranks.append(_process(start, stop, cpus) | {
            "spawn_seconds": start["spawn_seconds"], "summed_update_spans": _spans(steps),
            "summed_checkpoint_spans": _spans(checkpoints), "ready_spans": _one(rows, "training_ready")["timings"],
            "later_update_seconds": _series([row["wall_seconds"] for row in steps[2:]]),
            "update_cpu_seconds": _series([row["cpu_seconds"] for row in steps]),
        })
    assert len({_identity(rank["ready"]) for rank in ranks}) == world
    assert len({tuple(rank["ready"]["affinity"]) for rank in ranks}) == world
    return {
        "rank_0_log": str(log), "ranks": ranks, "runtime": runtime, "updates": updates,
        "checkpoint_updates": saves, "checkpoint_bytes": [row["bytes"] for row in publications],
        "checkpoint_wall_seconds": [row["wall_seconds"] for row in publications],
        "later_update_max_rank_seconds": _series([max(step) for step in by_update[2:]]),
        "later_update_rank_spread_seconds": _series([max(step) - min(step) for step in by_update[2:]]),
    }


def summarize(root):
    recipe, index = load(root / "workload.json"), load(root / "inventory.json")
    for receipt in load(root / "canonical.json").values():
        assert sha(Path(receipt["path"])) == receipt["sha256"]
    stages = {row["name"]: row for row in events(root / "stages.jsonl")}
    evidence = {}
    for name, manifest, _ in index["datasets"]:
        path = Path(manifest)
        base = path.parent.parent / ".generation-attempts" / path.parent.name
        evidence[name] = _pool(base, len(load(path)["shards"]), stages[name]["cpu_ids"], evaluation=False)
    for name, _, distributed, checkpoint, dataset, updates, phase in index["training"]:
        row = stages[name]
        threads = 1 if distributed or name.endswith("one-thread-control") else row["cpus"]
        evidence[name] = _train(Path(checkpoint), updates, phase, recipe["checkpoint_interval"], row["cpu_ids"],
                                distributed, threads)
        evidence[name]["source_manifest"] = dataset
    predictions = [chain[-1] for chain in index["chains"]] + [case[-1] for case in index["frozen"]]
    predictions.extend(case[1] for case in index["continuations"])
    for manifest in predictions:
        path = Path(manifest)
        name = path.parent.name
        evidence[name] = _pool(path.parent / "attempts", len(load(path)["shards"]), stages[name]["cpu_ids"], evaluation=True)
    for name, row in stages.items():
        assert row["exit_code"] == 0 and not row["remaining"] and row["failure"] is None
        scratch = row.get("scratch_retained")
        assert scratch is None or scratch == {"files": 0, "bytes": 0}
        peaks = row["sampled_peaks"]
        terminal = row["terminal"]
        cpu = sum(sum(terminal[key].values()) for key in ("root_self_cpu", "waited_children_cpu"))
        monitor = sum(row["monitor_cpu"].values())
        evidence.setdefault(name, {}).update({
            "monitor_cpu_seconds": monitor, "production_and_launcher_cpu_seconds": cpu,
            "average_cpus_including_monitor": (cpu + monitor) / row["wall_seconds"],
            "sampled_peaks": peaks, "scratch_retained": scratch,
        })
    _record(root / "resource_checks.json", {"summary_source_sha256": sha(Path(__file__)), "stages": evidence})


def sustained(root):
    from scripts.verify_multicore_audit import _prediction
    from nn_window_ablation.data import open_dataset

    _subreaper()
    os.environ.update({key: "1" for key in THREAD_ENV})
    os.environ.update(PYTHONHASHSEED="0", PYTHONUNBUFFERED="1")
    recipe = load(root / "workload.json") | {"seed": 102}
    cpus = load(root / "environment.json")["chosen_cpus"][:4]
    canonical = load(root / "canonical.json")["checkpoint"]
    checkpoint = Path(canonical["path"])
    assert sha(checkpoint) == canonical["sha256"]
    output = root / "sustained"
    output.mkdir()
    _record(output / "workload.json", {"recipe": recipe, "held_shots": 257, "held_shard_rows": 4, "checkpoint": canonical})
    audit = Audit(output, recipe, cpus)
    held = audit.generate("sustained-generation", 4, output / "data", "held", "test", 257, 4, family="diagnostic")
    open_dataset(held.parent, expected_split="test").verify_payloads()
    predictions = audit.evaluate("sustained-evaluation", 4, output / "runs", held, checkpoint, family="diagnostic")
    _record(output / "resource_checks.json", {
        "generation": _pool(output / "data/.generation-attempts/held", 65, cpus, evaluation=False),
        "evaluation": _pool(predictions.parent / "attempts", 65, cpus, evaluation=True),
        "scientific_checks": _prediction(held, checkpoint, predictions),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sustained", action="store_true", help="run the separate four-CPU 65-shard diagnostic")
    args = parser.parse_args()
    action = sustained if args.sustained else summarize
    action(args.output.resolve())
