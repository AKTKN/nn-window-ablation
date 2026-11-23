# Run nn-window-ablation on a PBS cluster

The commands below prepare a fixed checkout, install the CPU environment,
render and submit small jobs, and transfer a verified results archive. Run
installation and job planning on the login node where site policy permits.
Run simulation, profiling, and training in a compute allocation.

The supplied site record lists PBS commands under `/opt/pbs/bin` and advertises
a node with 96 PBS CPU resources and 350 GB. Ask the site for the actual queue,
account, limits, and shared paths. The small acceptance examples request
one CPU and 4 GB. The repository's cluster acceptance remains pending.
[Verification status](verification/P7.md#allocated-node-acceptance-pending)

## 1. Connect and check out a fixed revision

In your local Linux or WSL shell, set the SSH destination supplied by the site
and connect.

```bash
ssh "$CLUSTER_LOGIN"
```

On the login node, set `NNWA_WORK` to an absolute project or scratch path
approved for both login and compute nodes. Set `NNWA_REVISION` to the full
Git commit selected for the run. Use your existing GitHub access for the
private repository.

```bash
: "${NNWA_WORK:?Set an approved shared project directory}"
: "${NNWA_REVISION:?Set the full Git commit to run}"
export NNWA_WORK NNWA_REVISION
export NNWA_REPO="$NNWA_WORK/source"
export NNWA_DATA_ROOT="$NNWA_WORK/data"
export NNWA_RUN_ROOT="$NNWA_WORK/runs"
export NNWA_JOBS="$NNWA_WORK/jobs"
export NNWA_EXPORTS="$NNWA_WORK/exports"

mkdir -p "$NNWA_WORK" "$NNWA_DATA_ROOT" "$NNWA_RUN_ROOT" "$NNWA_JOBS" "$NNWA_EXPORTS"
git clone https://github.com/AKTKN/nn-window-ablation.git "$NNWA_REPO"
git -C "$NNWA_REPO" checkout --detach "$NNWA_REVISION"
git -C "$NNWA_REPO" rev-parse HEAD
git -C "$NNWA_REPO" status --short
```

Keep the checkout clean while jobs are queued or running. Write plans,
datasets, job logs, and exports to the separate roots above.

## 2. Install Conda dependencies and capture the environment

Use the site's Conda installation or load its documented shell module.
Set the absolute executable path if Conda is outside `PATH`. Use a prefix
visible on compute nodes. The environment below uses Python 3.11 and the
PyTorch 2.7.0 CPU wheel.

```bash
export NNWA_CONDA_EXE="$(type -P conda)"
export NNWA_CONDA_PREFIX="$NNWA_WORK/envs/nn_window_decoding"
cd "$NNWA_REPO"

"$NNWA_CONDA_EXE" env create --prefix "$NNWA_CONDA_PREFIX" --file environment.yml
export NNWA_PYTORCH_INDEX=https://download.pytorch.org/whl/cpu
runenv() { "$NNWA_CONDA_EXE" run --no-capture-output -p "$NNWA_CONDA_PREFIX" "$@"; }
runenv python -m pip install torch==2.7.0 --index-url "$NNWA_PYTORCH_INDEX"
runenv python -m pip install -e '.[experiment,dev]'
runenv python -m pip check
runenv nnwa --help

"$NNWA_CONDA_EXE" list -p "$NNWA_CONDA_PREFIX" --explicit > "$NNWA_JOBS/conda.explicit.txt"
runenv python -m pip freeze > "$NNWA_JOBS/pip.freeze.txt"
```

If network access is restricted, use the site's mirror or pre-staged CPU
packages. Reuse the prefix only for the selected checkout. P7 checks the
installed package files against the checkout and runs Conda by absolute
executable and prefix in each batch script.
[Conda run](https://docs.conda.io/projects/conda/en/stable/commands/run.html) ·
[PyTorch CPU wheels](https://pytorch.org/get-started/previous-versions/)

## 3. Prepare bounded job specifications

This shell block writes six explicit JSON plans outside the Git checkout.
Set `NNWA_QUEUE` and `NNWA_ACCOUNT` only if the site requires them. Adjust
the one-node request to site policy. Use one application process, one PyTorch
inter-operation thread, and zero loader workers for the initial checks.

```bash
runenv python - <<'PY'
from pathlib import Path
import json
import os

jobs = Path(os.environ["NNWA_JOBS"])
shared = {
    "repository": os.environ["NNWA_REPO"],
    "source_revision": os.environ["NNWA_REVISION"],
    "conda_executable": os.environ["NNWA_CONDA_EXE"],
    "conda_prefix": os.environ["NNWA_CONDA_PREFIX"],
    "data_root": os.environ["NNWA_DATA_ROOT"],
    "run_root": os.environ["NNWA_RUN_ROOT"],
    "resources": {
        "ncpus": 1, "memory": "4gb", "walltime": "00:10:00",
        "queue": os.environ.get("NNWA_QUEUE") or None,
        "account": os.environ.get("NNWA_ACCOUNT") or None,
    },
    "threads": 1,
}
train = [
    "training.max_steps=4", "training.batch_size=2", "training.microbatch_size=1",
    "training.learning_rate=0.001", "training.optimizer=adam",
    "training.scheduler=none", "data.training_chunk_shots=3",
]
base = str(Path(os.environ["NNWA_RUN_ROOT"]) / "base/checkpoint.pt")
stages = {
    "probe": ("probe", "probe", []),
    "generate": ("generate", "generation", [
        "data.split=test", "data.id=held", "data.shots=4", "data.shard_shots=2",
    ]),
    "profile": ("profile", "profile", []),
    "train": ("train", "base", train),
    "resume": ("resume", "base", [*train, "checkpoint.resume_from=" + json.dumps(base)]),
    "evaluate": ("evaluate", "paired", [
        "evaluation=paired", "data.id=held", "checkpoint.path=" + json.dumps(base),
    ]),
}
for name, (stage, run_name, overrides) in stages.items():
    value = {
        **shared, "stage": stage, "job_name": "nnwa-" + name, "run_name": run_name,
        "receipt_dir": str(jobs / f"{name}-attempt1"), "overrides": overrides,
    }
    (jobs / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
PY

for stage in probe generate profile train resume evaluate; do
  runenv nnwa pbs render --spec "$NNWA_JOBS/$stage.json" --output "$NNWA_JOBS/$stage.pbs"
  bash -n "$NNWA_JOBS/$stage.pbs"
done
```

Review the scripts before submission. Rendering resolves the Hydra settings,
checks explicit budgets, and quotes paths and overrides as argument tokens.
A different submission attempt needs a new `receipt_dir`. Train and resume
share the same scientific run name and settings. Profiling has its own
`profile` settings for `batch_size`, `warmup`, and `steps`, defaulting to
2, 1, and 2. [OpenPBS qsub](https://github.com/openpbs/openpbs/blob/master/doc/man1/qsub.1B)

For CPU DDP, use the same `train` or `resume` plan with `threads` set
to the requested threads per rank. Include `training.execution=ddp`,
`runtime.processes=P`, `data.train_source=stored_shards`, and an explicit
saved `data.id` in the overrides. Request a single node with at least
`P * threads` CPUs. The rendered script invokes one `nnwa train` supervisor.
The global batch must be divisible by `P`; the microbatch limit applies
inside a rank. Use the same rank count and numerical settings for resume.
The receipt verifies the completed checkpoint's topology. Compute-node
Gloo and filesystem checks are pending. [CPU DDP execution](verification/cpu_ddp_training.md)

For parallel generation, set `threads` to one and include
`data.sampling_processes=N` in the `generate` overrides. Request at least
`N` CPUs and keep `runtime.processes=1`. Leave the shard size fixed when
comparing worker layouts. PBS verifies the completed dataset and the
sampler layout in its provenance. [Parallel generation](verification/parallel_generation.md)

## 4. Validate and run on an allocated node

Set the installed `qsub` and `qstat` paths and submit the probe first.

```bash
qsub_bin=/opt/pbs/bin/qsub
qstat_bin=/opt/pbs/bin/qstat

probe_id="$("$qsub_bin" "$NNWA_JOBS/probe.pbs")"
"$qstat_bin" -f "$probe_id"
```

After it exits, inspect the receipt and import output.

```bash
cat "$NNWA_JOBS/probe-attempt1/job.json"
cat "$NNWA_JOBS/probe-attempt1/stdout.log"
runenv nnwa pbs status "$probe_id" --qstat "$qstat_bin" \
  --receipt "$NNWA_JOBS/probe-attempt1/job.json"
```

The probe checks the fixed revision, environment, root visibility, CPU
affinity, and dependency imports. Verify writable data and run roots by
running generation and training. For extra path checks in an interactive
allocation, use `findmnt -T "$NNWA_DATA_ROOT"`, `df -T "$NNWA_RUN_ROOT"`,
and a small file whose checksum can be compared from the login node.
PBS nodefile entries are recorded as slots. Use scheduler accounting and
CPU affinity to check the observed allocation. The receipt keeps requested
resources and observed evidence in separate fields. Depending on site job
history settings, finished jobs may be absent from `qstat`.
[OpenPBS qstat](https://github.com/openpbs/openpbs/blob/master/doc/man1/qstat.1B)

Submit the generation, profile, and base training jobs. These three stages
can run independently. Check their receipts before launching dependent work.

```bash
generate_id="$("$qsub_bin" "$NNWA_JOBS/generate.pbs")"
profile_id="$("$qsub_bin" "$NNWA_JOBS/profile.pbs")"
train_id="$("$qsub_bin" "$NNWA_JOBS/train.pbs")"
```

Once generation and training report `complete`, submit evaluation.

```bash
evaluate_id="$("$qsub_bin" "$NNWA_JOBS/evaluate.pbs")"
```

For each stage, inspect `jobs/<stage>-attempt1/job.json` and `stdout.log`.
Completed checkpoints live under `runs/base/`, and the paired prediction
manifest lives under `runs/paired/`. The profiling JSON has the observed
thread counts, CPU timings, and peak RSS. Use a fresh job and output path for
each thread count. Configure the request for the measured workload.
[CPU threading](https://docs.pytorch.org/docs/2.7/notes/multiprocessing.html)

### Check interruption and resume

For a naturally interrupted `base` run with a saved checkpoint, submit
`resume.pbs` in a new receipt directory. It uses the original declared total
budget and checkpoint cadence, and restores the optimizer, scheduler, and
stream cursor. Completed updates after the last durable save can be replayed.

For a controlled acceptance check, use a fresh `base` run with the same
training settings, then pause via the Python workflow after update one. Run
the following inside an approved interactive allocation before submitting
`train.pbs`, with the same repository, prefix, roots, and CPU settings
exported inside the allocation.

Request an interactive allocation through the site before running that check.

```bash
"$qsub_bin" -I -l select=1:ncpus=1:mem=4gb -l walltime=00:10:00 \
  -v NNWA_REPO,NNWA_CONDA_EXE,NNWA_CONDA_PREFIX,NNWA_DATA_ROOT,NNWA_RUN_ROOT
```

```bash
cd "$NNWA_REPO"
"$NNWA_CONDA_EXE" run --no-capture-output -p "$NNWA_CONDA_PREFIX" python - <<'PY'
from pathlib import Path
from hydra import compose, initialize_config_module
from nn_window_ablation.config import resolve_config
from nn_window_ablation.execution import train_workflow

overrides = [
    "runtime=cpu", "runtime.allocated_cpus=1", "runtime.intraop_threads=1",
    "run_name=base", "data.split=train", "training.max_steps=4", "training.batch_size=2",
    "training.microbatch_size=1", "training.learning_rate=0.001",
    "training.optimizer=adam", "training.scheduler=none", "data.training_chunk_shots=3",
]
with initialize_config_module(config_module="nn_window_ablation.conf", version_base="1.3"):
    config = resolve_config(compose(config_name="config", overrides=overrides), invocation_dir=Path.cwd())
train_workflow(config, stop_after=1)
PY
```

Then submit the resume script from the login node:

```bash
resume_id="$("$qsub_bin" "$NNWA_JOBS/resume.pbs")"
```

Check `phase_updates=4` in `runs/base/run.json` and that `resume-attempt1/job.json`
records successful exit and the expected checkpoint hash. Use compatible
CPU allocation, unchanged source, and the same packages and Conda prefix.
For S+, W, and D, create three train job specs with different run names and
receipt directories, the same base `checkpoint.init_from`, and the same
data and optimizer budget. Select the respective training preset in each.
Run their paired evaluations with one held-out dataset and identical deployment
windows. [Matched continuation](verification/P6.md)

For parallel evaluation, set the job's `threads` to 1 and include
`evaluation.workers=N`, `evaluation.worker_threads=K`, and optionally
`evaluation.max_in_flight=L` in its overrides. Request at least `N * K` CPUs
and allow memory for a separate model in each active worker. The local
verification covers PBS configuration and artifact checks. Allocation,
filesystem, and performance checks on a compute node are pending.

To resume an interrupted evaluation, render another `evaluate` plan with the
same run name and input content, a fresh `receipt_dir`, and
`evaluation.resume=true` in its overrides. Worker count and outstanding-task
limit may change at fixed numerical settings and batch size. The runner
verifies existing commits and fills only missing shards. A changed or corrupt
committed shard requires investigation or a new evaluation run.
[Parallel evaluation and artifact verification](verification/parallel_evaluation.md)

## 5. Analyze, transfer, and verify locally

Run analysis where site policy permits. Large analyses can use an approved
interactive allocation. The settings below are for the tiny acceptance
fixture.

```bash
cd "$NNWA_WORK"
runenv nnwa analyze runtime=local_cpu runtime.allocated_cpus=1 run_name=summary data.id=held \
  "analysis.predictions=[\"$NNWA_RUN_ROOT/paired/manifest.json\"]" \
  analysis.sparse_rule=1 analysis.bootstrap_replicates=99 analysis.bootstrap_seed=17
runenv nnwa export --dataset "$NNWA_DATA_ROOT/held" \
  --report "$NNWA_RUN_ROOT/summary/report.json" --output "$NNWA_EXPORTS/held-results.tar.gz" \
  --log "$NNWA_JOBS/evaluate-attempt1/job.json" --log "$NNWA_JOBS/evaluate-attempt1/stdout.log" \
  --log "$NNWA_JOBS/conda.explicit.txt" --log "$NNWA_JOBS/pip.freeze.txt"
(cd "$NNWA_EXPORTS" && sha256sum held-results.tar.gz > held-results.tar.gz.sha256)
```

Add `--include-detectors --include-checkpoints` if the recipient needs to
re-decode the held-out rows. Add `--training-dataset PATH` only when a selected
stored training dataset is needed. The default archive supports statistics
from its retained rows. For S+/W/D, pass the three prediction manifests to
`analysis=adaptation` before exporting the resulting report.
[Bundle capabilities](verification/P7.md#portable-bundles)

On the workstation, set `REMOTE_EXPORT_DIR` to the approved remote export
directory used above and transfer the archive and checksum over your
authenticated connection. Use the NumPy and SciPy versions recorded in the
report. The copied package inventories can help recreate them.

```bash
scp "$CLUSTER_LOGIN:$REMOTE_EXPORT_DIR/held-results.tar.gz" .
scp "$CLUSTER_LOGIN:$REMOTE_EXPORT_DIR/held-results.tar.gz.sha256" .
sha256sum -c held-results.tar.gz.sha256
conda run -n nn_window_decoding nnwa extract held-results.tar.gz incoming
conda run -n nn_window_decoding nnwa reanalyze --bundle incoming --output replay
diff -u incoming/analysis/rates.csv replay/rates.csv
diff -u incoming/analysis/contrasts.csv replay/contrasts.csv
```

The manifest independently verifies every extracted file. Save the completed
job receipts and the archive digest with the compute-node acceptance record.
