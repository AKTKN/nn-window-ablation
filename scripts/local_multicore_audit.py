"""Run and verify the fixed Linux 2/4/8-CPU saved-data experiment."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys

from scripts.audit_monitor import _record, _subreaper, measure


SCRIPT_DIR = Path(__file__).resolve().parent
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text())


class Audit:
    def __init__(self, root, recipe, cpus):
        self.root, self.recipe, self.cpus = root, recipe, cpus
        self.nnwa = str(Path(sys.executable).with_name("nnwa"))
        self.rows = []
        self.datasets = []
        self.chains = []
        self.trains = []
        self.frozen = {}
        self.continuations = []

    def run(self, name, stage, count, args, *, roots=(), temporary=(), family=None, repetition=None):
        case = self.root / "measurements" / name
        result = measure(
            [self.nnwa, stage, *map(str, args)], case, self.cpus[:count], roots, temporary,
            interval=self.recipe["sample_interval_seconds"], timeout=self.recipe["timeout_seconds"],
            memory_limit=self.recipe["memory_limit_mib"] * 2**20, disk_reserve=self.recipe["disk_reserve_gib"] * 2**30,
        )
        row = {
            "name": name, "stage": stage, "case": family, "repetition": repetition, "cpus": count,
            "command_record": str(case / "command.json"), "result_record": str(case / "result.json"), **result,
        }
        self.rows.append(row)
        with (self.root / "stages.jsonl").open("a") as file:
            print(json.dumps(row), file=file)
        print(name, f"{result['wall_seconds']:.2f}s", f"exit={result['exit_code']}", flush=True)
        if result["exit_code"] or result["remaining"] or result["failure"]:
            raise RuntimeError(f"stage failed: {name}. See {case / 'stdout.log'} and result.json")
        if not result["terminal"] or result["scratch_retained"]["files"]:
            raise RuntimeError(f"incomplete terminal accounting or retained scratch files: {case / 'result.json'}")

    def common(self, count, data_root, run_root=None):
        if run_root is not None:
            run_root.mkdir(parents=True, exist_ok=True)
        return [
            "runtime=local_cpu", f"runtime.allocated_cpus={count}", f"seed={self.recipe['seed']}",
            f"paths.data_root={data_root}", *([f"paths.run_root={run_root}"] if run_root else []),
        ]

    def generate(self, name, count, root, data_id, split, shots, shard_rows, *, workers=None, phase=None, **tags):
        workers = count if workers is None else workers
        destination = root / data_id
        self.run(name, "generate", count, [
            *self.common(count, root), f"data.id={data_id}", f"data.split={split}", f"data.shots={shots}",
            f"data.shard_shots={shard_rows}", f"data.sampling_processes={workers}",
            *([f"training={phase}"] if phase else []),
        ], roots=(destination, root / ".generation-attempts" / data_id),
            temporary=(root / f".{data_id}.generation",), **tags)
        manifest = destination / "manifest.json"
        self.datasets.append((name, manifest, split))
        return manifest

    def train(self, name, count, run_root, dataset, *, distributed=True, threads=None, parent=None, phase=None,
              validation=None, **tags):
        r = self.recipe
        directory = run_root / name
        updates = r["continuation_updates"] if phase else r["base_updates"]
        self.run(name, "train", count, [
            *self.common(count, dataset.parent.parent, run_root),
            f"run_name={name}", f"data.id={dataset.parent.name}", "data.train_source=stored_shards",
            "training.log_every_updates=1", f"training.max_steps={updates}", f"training.batch_size={r['global_batch']}",
            f"training.microbatch_size={r['local_microbatch']}", "training.optimizer=adam",
            f"training.learning_rate={r['learning_rate']}", f"training.scheduler={r['scheduler']}",
            f"training.scheduler_step_size={r['scheduler_step_size']}", f"training.scheduler_gamma={r['scheduler_gamma']}",
            f"checkpoint.save_every_updates={r['checkpoint_interval']}",
            *(["training.execution=ddp", f"runtime.processes={count}"] if distributed
              else [f"runtime.intraop_threads={threads if threads is not None else count}"]),
            *([f"training={phase}", f"checkpoint.init_from={parent}"] if phase else []),
            *([f"training.validation_data_id={validation.parent.name}"] if validation else []),
        ], roots=(directory,), **tags)
        checkpoint = directory / "checkpoint.pt"
        self.trains.append((name, count, distributed, checkpoint, dataset, updates, phase))
        return checkpoint

    def evaluate(self, name, count, run_root, dataset, checkpoint, *, mode="paired", **tags):
        directory = run_root / name
        self.run(name, "evaluate", count, [
            *self.common(count, dataset.parent.parent, run_root), f"run_name={name}", f"data.id={dataset.parent.name}",
            f"evaluation={mode}", f"evaluation.workers={count}", f"evaluation.batch_size={self.recipe['inference_batch']}",
            f"checkpoint.path={checkpoint}",
        ], roots=(directory,), temporary=(directory / ".evaluation-staging",), **tags)
        return directory / "manifest.json"

    def execute(self):
        r = self.recipe
        canonical = None
        for repeat, order in enumerate(r["cpu_orders"], 1):
            for count in order:
                key = f"chain-r{repeat}-c{count}"
                tags = {"family": "chain", "repetition": repeat}
                base = self.root / "artifacts" / key
                train_data = self.generate(f"{key}-train-data", count, base / "data", "training", "train",
                                           r["base_shots"], r["base_shard_rows"], **tags)
                checkpoint = self.train(f"{key}-train", count, base / "runs", train_data, **tags)
                held = self.generate(f"{key}-held", count, base / "data", "held", "test",
                                     r["held_shots"], r["held_shard_rows"], **tags)
                prediction = self.evaluate(f"{key}-evaluate", count, base / "runs", held, checkpoint, **tags)
                self.chains.append((train_data, checkpoint, held, prediction))
                if canonical is None:
                    canonical = train_data, held, checkpoint
        training, held, reference = canonical
        _record(self.root / "canonical.json", {
            key: {"path": str(path), "sha256": sha(path)}
            for key, path in zip(("training", "held", "checkpoint"), canonical, strict=True)
        })

        controlled = self.root / "artifacts" / "controlled"
        for repeat, order in enumerate(r["cpu_orders"], 1):
            tags = {"family": "controlled", "repetition": repeat}
            for count in order:
                for distributed in ((False, True) if repeat == 1 else (True, False)):
                    label = "ddp" if distributed else "serial"
                    self.train(f"train-r{repeat}-c{count}-{label}", count, controlled / "runs", training,
                               distributed=distributed, **tags)
                self.frozen[count, repeat] = self.evaluate(
                    f"frozen-r{repeat}-c{count}", count, controlled / "runs", held, reference, **tags,
                )
            serial_data = controlled / f"serial-r{repeat}"
            self.generate(f"serial-r{repeat}-train-data", 1, serial_data, "training", "train",
                          r["base_shots"], r["base_shard_rows"], **tags)
            self.generate(f"serial-r{repeat}-held", 1, serial_data, "held", "test",
                          r["held_shots"], r["held_shard_rows"], **tags)
        for count in r["skip_cross_thread_weight_cpu_budgets"]:
            self.train(f"train-c{count}-one-thread-control", count, controlled / "runs", training,
                       distributed=False, threads=1, family="correctness")
        self.frozen[1, 1] = self.evaluate("frozen-serial-reference", 1, controlled / "runs", held, reference,
                                            family="correctness")
        latest = self.frozen[4, 1]
        report_dir = controlled / "runs" / "analysis"
        report = report_dir / "report.json"
        self.run("paired-analysis", "analyze", 1, [
            *self.common(1, held.parent.parent, report_dir.parent),
            "run_name=analysis", f"data.id={held.parent.name}", f"analysis.predictions=[{latest}]",
            "analysis.bootstrap_seed=21", "analysis.bootstrap_replicates=99", "analysis.sparse_rule=1",
        ], roots=(report_dir,), family="compatibility")
        compact = controlled / "compact.tar.gz"
        portable = controlled / "portable"
        self.run("compact-export", "export", 1, ["--dataset", held.parent, "--report", report, "--output", compact],
                 roots=(compact,), family="compatibility")
        self.run("compact-extract", "extract", 1, [compact, portable], roots=(portable,), family="compatibility")
        replay = controlled / "reanalysis"
        self.run("compact-reanalysis", "reanalyze", 1, ["--bundle", portable, "--output", replay],
                 roots=(replay,), family="compatibility")
        self.report, self.replay = report, replay / "report.json"

        continuation_root = self.root / "artifacts" / "continuation"
        saved = self.generate("continuation-data", 4, continuation_root / "data", "adapt", "train",
                              r["continuation_shots"], r["base_shard_rows"], phase="continued_full", family="continuation")
        validation = self.generate("validation-data", 4, saved.parent.parent, "validation", "validation",
                                   4, 2, family="continuation")
        for phase in ("continued_full", "fixed_window", "dynamic_window"):
            state = self.train(phase, 4, continuation_root / "runs", saved, parent=reference, phase=phase,
                               validation=validation, family="continuation")
            prediction = self.evaluate(f"eval-{phase}", 4, continuation_root / "runs", held, state, mode="S",
                                       family="continuation")
            self.continuations.append((state, prediction, held, reference))
        _record(self.root / "inventory.json", {
            "datasets": [(name, str(path), split) for name, path, split in self.datasets],
            "chains": [[str(path) for path in chain] for chain in self.chains],
            "training": [(name, count, distributed, str(path), str(dataset), updates, phase)
                         for name, count, distributed, path, dataset, updates, phase in self.trains],
            "frozen": [(count, repeat, str(path)) for (count, repeat), path in self.frozen.items()],
            "continuations": [[str(path) for path in chain] for chain in self.continuations],
            "report": str(self.report), "replay": str(self.replay),
        })

    def restore_inventory(self):
        index = load(self.root / "inventory.json")
        self.rows = [json.loads(line) for line in (self.root / "stages.jsonl").read_text().splitlines()]
        self.datasets = [(name, Path(path), split) for name, path, split in index["datasets"]]
        self.chains = [tuple(map(Path, chain)) for chain in index["chains"]]
        self.trains = [(name, count, distributed, Path(path), Path(dataset), updates, phase)
                       for name, count, distributed, path, dataset, updates, phase in index["training"]]
        self.frozen = {(count, repeat): Path(path) for count, repeat, path in index["frozen"]}
        self.continuations = [tuple(map(Path, chain)) for chain in index["continuations"]]
        self.report, self.replay = Path(index["report"]), Path(index["replay"])

    def summarize(self):
        from scripts.verify_multicore_audit import verify

        _record(self.root / "scientific_checks.json", verify(self))
        flattened = []
        for row in self.rows:
            end = row["terminal"]
            cpu = sum(end[key][mode] for key in ("root_self_cpu", "waited_children_cpu") for mode in ("user", "system"))
            io = end["root_io_inclusive"] or {}
            flattened.append({
                **{key: row[key] for key in ("name", "stage", "case", "repetition", "cpus", "wall_seconds")},
                "cpu_seconds": cpu, "average_cpus": cpu / row["wall_seconds"],
                "peak_rss_mib": row["sampled_peaks"]["rss_bytes"] / 2**20,
                "peak_artifact_bytes": row["sampled_peaks"]["bytes"],
                "peak_temp_bytes": row["sampled_peaks"]["temp_bytes"], "peak_temp_files": row["sampled_peaks"]["temp_files"],
                "retained_files": row["retained"]["files"], "retained_bytes": row["retained"]["bytes"],
                **{key: io.get(key) for key in ("read_bytes", "write_bytes", "rchar", "wchar", "syscr", "syscw")},
            })
        with (self.root / "stages.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(flattened[0]))
            writer.writeheader()
            writer.writerows(flattened)
        _trends(self.root, flattened, self.recipe)


def _trends(root, rows, recipe):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = {
        "Training data generation": lambda row: row["case"] == "chain" and row["name"].endswith("-train-data"),
        "Held-out generation": lambda row: row["case"] == "chain" and row["name"].endswith("-held"),
        "Inline training": lambda row: row["case"] == "controlled" and row["name"].endswith("-serial"),
        "DDP training": lambda row: row["case"] == "controlled" and row["name"].endswith("-ddp"),
        "Frozen evaluation": lambda row: row["case"] == "controlled" and row["stage"] == "evaluate",
    }
    trends = {}
    for label, choose in series.items():
        grouped = {}
        for row in rows:
            if choose(row):
                grouped.setdefault(row["cpus"], []).append(row)
        entries = {}
        for cpus, data in sorted(grouped.items()):
            entries[cpus] = {
                key: {"values": values, "median": median(values), "range": [min(values), max(values)]}
                for key in ("wall_seconds", "cpu_seconds", "average_cpus", "peak_rss_mib", "read_bytes", "write_bytes",
                            "retained_files", "retained_bytes", "peak_temp_bytes")
                if (values := [row[key] for row in data if row[key] is not None])
            }
            if 2 in entries:
                speed = entries[2]["wall_seconds"]["median"] / entries[cpus]["wall_seconds"]["median"]
                entries[cpus]["speedup_from_2"] = speed
                entries[cpus]["efficiency_from_2"] = speed / (cpus / 2)
            work = recipe["held_shots"] if label in ("Frozen evaluation", "Held-out generation") else recipe["base_shots"]
            entries[cpus]["rows_per_second"] = work / entries[cpus]["wall_seconds"]["median"]
        trends[label] = entries
    _record(root / "trends.json", trends)
    metrics = (
        ("wall_seconds", "Startup-inclusive wall time (s)", "wall_time.png"),
        ("rows_per_second", "Trajectories per second", "throughput.png"),
        ("peak_rss_mib", "Sampled process-group RSS (MiB)", "memory.png"),
        ("write_bytes", "OS-attributed storage writes (bytes)", "io.png"),
    )
    for field, unit, filename in metrics:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for label, values in trends.items():
            plotted = [(cpu, metrics[field]) for cpu, metrics in values.items() if field in metrics]
            ax.plot([cpu for cpu, _ in plotted], [x["median"] if isinstance(x, dict) else x for _, x in plotted],
                    marker="o", label=label)
        ax.set(xlabel="Allocated logical CPUs", ylabel=unit, xticks=[2, 4, 8])
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(root / filename, dpi=170)
        plt.close(fig)


def _environment(root, recipe, cpus):
    from nn_window_ablation.execution import _package_source_sha256
    from nn_window_ablation.runtime import _cpu_quota

    def command(*args):
        return subprocess.check_output(args, text=True).strip()

    _record(root / "workload.json", recipe)
    _record(root / "environment.json", {
        "git_commit": command("git", "rev-parse", "HEAD"), "git_changes": command("git", "status", "--short"),
        "package_sha256": _package_source_sha256(), "python": sys.version, "platform": platform.platform(),
        "driver_sha256": {path.name: sha(path) for path in sorted(SCRIPT_DIR.glob("*audit*.py"))},
        "packages": {name: importlib.metadata.version(name) for name in (
            "nn-window-ablation", "torch", "torch-geometric", "stim", "numpy", "scipy", "matplotlib",
        )},
        "effective_affinity": sorted(os.sched_getaffinity(0)), "chosen_cpus": cpus,
        "cpu_model_and_topology": command("lscpu"), "effective_cpu_quota": _cpu_quota(),
        "topology": command("lscpu", "-e=CPU,CORE,NODE,SOCKET,ONLINE"),
        "meminfo": Path("/proc/meminfo").read_text(), "cgroup": Path("/proc/self/cgroup").read_text(),
        "cpu_max": (Path("/sys/fs/cgroup/cpu.max").read_text() if Path("/sys/fs/cgroup/cpu.max").is_file() else None),
        "memory_max": (Path("/sys/fs/cgroup/memory.max").read_text() if Path("/sys/fs/cgroup/memory.max").is_file() else None),
        "filesystem": command("findmnt", "-T", str(root), "-no", "FSTYPE"),
        "limits": Path("/proc/self/limits").read_text(), "cache_policy": "No cache flush or prewarming.",
    })
    (root / "pip-freeze.txt").write_text(command(sys.executable, "-m", "pip", "freeze") + "\n")


def main():
    from nn_window_ablation.runtime import _cpu_quota

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory outside the tracked checkout")
    parser.add_argument("--recipe", type=Path, default=SCRIPT_DIR / "local_multicore_audit.json")
    parser.add_argument("--summarize-only", action="store_true", help="verify and summarize an existing measured inventory")
    args = parser.parse_args()
    available = sorted(os.sched_getaffinity(0))
    _subreaper()
    os.environ.update({key: "1" for key in THREAD_ENV})
    os.environ.update(PYTHONHASHSEED="0", PYTHONUNBUFFERED="1")
    root = args.output.resolve()
    recipe = load(root / "workload.json" if args.summarize_only else args.recipe)
    audit = Audit(root, recipe, available[:8])
    if args.summarize_only:
        audit.restore_inventory()
    else:
        root.mkdir(parents=True)
        _environment(root, recipe, available[:8])
        quota = _cpu_quota()
        if len(available) < 8 or (quota is not None and quota < 8):
            raise SystemExit(f"matrix blocked: eight effective CPUs required. See {root / 'environment.json'}")
        audit.execute()
    audit.summarize()


if __name__ == "__main__":
    main()
