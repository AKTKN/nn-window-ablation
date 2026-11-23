# Getting started with nn-window-ablation

This walkthrough runs a tiny CPU simulation of a rotated surface-code memory
at distance 5, with 60 extraction rounds, 12 cores, and circuit noise
probability 0.003. Each complete Stim trajectory carries raw detector bits,
one global logical label, and 12 core labels whose XOR is the global label.
The model can predict either the global label or a selected core label.
The few trajectories and updates below are for checking the workflow.
[Scientific protocol](02_ablation_protocol.md)

## 1. Install the CPU environment

From the repository root on Linux, with Conda available:

```bash
conda env create -f environment.yml
conda run -n nn_window_decoding python -m pip install torch==2.7.0 \
  --index-url https://download.pytorch.org/whl/cpu
conda run -n nn_window_decoding python -m pip install -e '.[experiment,dev]'
conda run -n nn_window_decoding python -m pip check
```

Run the creation step once. The editable installation exposes the `nnwa`
command from any directory. The environment notes also provide inventories for
recreating the locally verified Linux environment. [Environment setup](../environments/README.md)

## 2. Generate held-out data and train

Start from a fresh working directory so artifact names can be used as written.
By default the data and run roots are `./data` and `./runs` relative to this
directory. Every command below uses one numerical CPU.

```bash
workdir="$(mktemp -d)"
cd "$workdir"
nnwa() { conda run --no-capture-output -n nn_window_decoding nnwa "$@"; }
cpu=(runtime=local_cpu runtime.allocated_cpus=1)

nnwa validate "${cpu[@]}" run_name=configuration
nnwa generate "${cpu[@]}" data.split=test data.id=held \
  data.shots=4 data.shard_shots=2
nnwa train "${cpu[@]}" run_name=base \
  training.max_steps=2 training.batch_size=2 training.microbatch_size=1 \
  training.optimizer=adam training.learning_rate=0.001 training.scheduler=none \
  data.training_chunk_shots=3
```

`validate` writes the resolved settings. `generate` seals packed held-out
detectors and labels in `data/held/manifest.json` and its shards. Training
generates its own stream, with different identities from the held-out set.
Here a batch of two trajectories contributes four queries per update. The
checkpoint at `runs/base/checkpoint.pt` stores model, optimizer, sampling,
and update state. `runs/base/run.json` records completion and counters.

To resume an interruption, repeat the training command with the original
run name, roots, and settings, and add
`checkpoint.resume_from=runs/base/checkpoint.pt`. The terminal checkpoint can
be opened again with zero additional updates. New continuation arms instead
start with `checkpoint.init_from` and a fresh optimizer.
The default save cadence is 100 updates, with mandatory saves at initialization,
an explicit pause, and the terminal update. Reopening the latest durable
checkpoint may replay work after that save.
[Training configuration](../src/nn_window_ablation/conf/README.md)

To generate a finite training set before starting optimization, use a fresh
training run and this alternative to the on-demand training command.

```bash
nnwa generate "${cpu[@]}" data.split=train data.id=training \
  data.shots=5 data.shard_shots=3
nnwa train "${cpu[@]}" run_name=stored \
  data.train_source=stored_shards data.id=training \
  training.max_steps=2 training.batch_size=2 training.microbatch_size=1 \
  training.optimizer=adam training.learning_rate=0.001 training.scheduler=none \
  checkpoint.save_every_updates=2
```

Four of the five saved trajectories are consumed once. Pass a budget within
the stored capacity and retain the generator's seed, family, replicate, and
training phase. The evaluation example below uses the earlier `base`
checkpoint. Substitute `runs/stored/checkpoint.pt` to evaluate the stored run.

## 3. Evaluate one checkpoint three ways

```bash
nnwa evaluate "${cpu[@]}" run_name=paired evaluation=paired \
  data.id=held checkpoint.path=runs/base/checkpoint.pt
nnwa analyze "${cpu[@]}" run_name=summary data.id=held \
  'analysis.predictions=[runs/paired/manifest.json]' \
  analysis.sparse_rule=1 analysis.bootstrap_replicates=99 analysis.bootstrap_seed=17
```

G predicts the global bit from all observations. F predicts the 12 core bits
with full context and XORs them. S uses the same core targets and model with
deployment windows. All three see the same held-out trajectories. The
manifest binds the checkpoint hash, shot IDs, and query settings. Add
`evaluation.matching=true` to include matching with the original detector
error model.

On Linux, use a persistent pool when multiple CPUs are allocated. This example
uses up to four one-thread workers and at most eight outstanding shard tasks.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
nnwa evaluate runtime=local_cpu runtime.allocated_cpus=4 run_name=paired-parallel \
  evaluation=paired evaluation.workers=4 evaluation.worker_threads=1 \
  evaluation.max_in_flight=8 data.id=held checkpoint.path=runs/base/checkpoint.pt
```

If evaluation stops, repeat the command with the same run name and add
`evaluation.resume=true`. You can change the worker count, CPU allocation and
in-flight limit while keeping the per-worker threads and batch size fixed.
Resume verifies the committed shards and fills the gaps. Analysis can use the
final manifest after exact coverage has been committed.
[Parallel evaluation and resume](verification/parallel_evaluation.md)

Read `runs/summary/rates.csv` for shot counts, failures, logical error rates,
and Wilson intervals. Read `contrasts.csv` for signed paired differences and
discordances. For A−B, `n10` counts A failures when B succeeds and `n01`
counts the reverse. The difference is `(n10 - n01) / shots`. A positive
value means A failed more often in these shots. Use `report.json` for
per-core diagnostics and the saved analysis policy. Select production
bootstrap and sparse-interval settings from the pilot.
[Paired inference and statistics](verification/P5.md)

## 4. Export and replay the statistics

```bash
nnwa export --dataset data/held --report runs/summary/report.json \
  --output held-results.tar.gz
nnwa extract held-results.tar.gz moved-results
nnwa reanalyze --bundle moved-results --output replay
diff -u moved-results/analysis/rates.csv replay/rates.csv
diff -u moved-results/analysis/contrasts.csv replay/contrasts.csv
```

The default archive contains the packed labels, shot IDs, predictions, and
analysis records. Reanalysis takes its settings and RNG seed from the saved
report. Add both `--include-detectors` and `--include-checkpoints` when you
need to run inference again on a different machine. Extract into a new
directory and use the bundled dataset and checkpoint in `nnwa evaluate`.
See `manifest.json` for their relative paths.
[Export schema and checks](verification/P7.md)

An included terminal base checkpoint can also initialize a new continuation.
Exact training resume uses the original resolved paths, code, and runtime
configuration.

For S+, W, and D, use the matched continuation command from the README.
The three arms share a base checkpoint, new trajectory/core draws, and update
budget. Evaluate all arms on the same held-out set and analyze them together
with `analysis=adaptation`. [Matched continuation example](../README.md#run-a-matched-continuation-check)

For larger runs, use the [PBS cluster guide](04_cluster_workflow.md).
