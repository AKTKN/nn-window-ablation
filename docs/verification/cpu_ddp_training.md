# Phase 3 CPU DDP training

## Global update contract

Use `training.execution=ddp` and `data.train_source=stored_shards` in the
installed `nnwa train` command. `runtime.processes` is the world size and
`runtime.intraop_threads` is the per-rank CPU thread limit. Before reserving
a run, the command requires a positive equal local batch, a common local
microbatch schedule, one inter-op thread, and no nested loader or sampler
pools. The source must contain the full fixed-update budget. Inline training
is the default and continues to support generated-on-demand input.

The global batch is `B` trajectories. Each of `P` ranks reads one contiguous
segment of `B/P` from the ordered global plan. Each rank derives the same
plan and query draws from the existing PCG64 streams and shared cursor. The
planner previews physical row identities without opening a payload. It
retains at most two shard permutations. Each rank verifies the payloads it
opens, holds at most one mapped source shard at a time, and unpacks its
local rows. The same rank reads through a shard boundary when its segment
crosses one. The global cursor advances by `B`, including when the row
partition differs from the source shard partition.

Every trajectory contributes its global and sampled-core queries. All
ranks advance the same independent logical core and buffer streams. D draws
one pair of buffer lengths per global update. Each local mean over `m`
trajectories receives weight `m / (B/P)` before backward. The Gloo reducer
averages rank gradients. Both forward and backward run under DDP's
`no_sync()` on earlier microbatches. The last microbatch uses the DDP wrapper
with synchronization. Each rank checks finite loss and gradients and steps
the optimizer and scheduler once. A scalar reduction after the optimizer
provides the global loss and prevents the checkpoint path from proceeding
after a peer fails its update.
[PyTorch 2.7 DistributedDataParallel](https://docs.pytorch.org/docs/2.7/generated/torch.nn.parallel.DistributedDataParallel.html)

The CPU reducer uses `find_unused_parameters=True` and `static_graph=False`.
The pinned reducer tracks autograd hooks across the accumulation cycle.
Tests on the installed build verify that an earlier nonempty microbatch
contributes its convolution gradients even when all final microbatches
are empty. They also verify that when all ranks' rows are empty throughout
an update, the convolution gradients remain absent and Adam leaves their
weights, moments and step counters unchanged.
[PyTorch 2.7 reducer source](https://raw.githubusercontent.com/pytorch/pytorch/v2.7.0/torch/csrc/distributed/c10d/reducer.cpp) · [Adam behavior with absent gradients](https://docs.pytorch.org/docs/2.7/generated/torch.optim.Optimizer.zero_grad.html)

## Ownership, resources and recovery

The CLI starts one Linux spawn supervisor and uses a unique local file
rendezvous with loopback Gloo. The supervisor checks the effective affinity
and any visible cgroup-v2 bandwidth quota. It gives each rank a disjoint
slice of `runtime.intraop_threads` CPUs. Native-library limits are applied
before spawning. An inherited thread limit above the requested per-rank
count is a configuration error. The supervisor and Gloo's communication
threads share the selected CPUs. The monitored command includes their
startup, processing and cleanup costs.

Rank 0 reserves the run, holds its training lock and publishes checkpoints
and run receipts. Every checkpoint boundary gathers rank-local Torch/Python
states and checks rank agreement on model hash, update and input digest.
The synchronized copy of the unwrapped model, optimizer and scheduler is
written once using the existing atomic checkpoint replacement. Cadence,
phase counters, cursor, and global query RNGs are saved. The initialization,
periodic, controlled-pause and terminal saves remain available. Each save
records one global input digest over ordered trajectory identities, core
choices and dynamic buffers, with a bounded prefix of up to 16 updates.
Continuation audits also retain the existing cross-arm comparison digest
over trajectories and core choices.

The existing checkpoint envelope marks distributed-state schema 1 when
present. That record carries the world size, equal partition, local batch
and microbatch, effective numerical settings and rank RNG list. Distributed
resume requires the original topology, settings, source and environment.
A prior serial terminal checkpoint remains usable for evaluation and as
a continuation parent. A DDP terminal checkpoint uses the same unwrapped
decoder schema and can be loaded for evaluation or continuation without
Gloo. Exact replay of a checkpoint from an older source revision remains
bound to the original source and environment.

Ranks close their readers, release the DDP reducers, synchronize, and
destroy Gloo before rank 0 performs optional terminal validation. Before
publication, the supervisor confirms that every peer exited successfully.
The isolated child reports its exit code after workflow cleanup and
flushing Python streams, avoiding interpreter-exit native hooks. A rank
failure, signal or collective timeout causes the supervisor to stop the
group. It sends termination,
waits up to five seconds, then allows five more seconds after SIGKILL.
Linux parent-death signals stop ranks if the supervisor exits abruptly.
The prior durable checkpoint is available for exact replay. A crash after
atomic checkpoint replacement can leave the newer completed update.
PBS requires a complete run receipt whose checkpoint hash and topology
agree with the requested job.
[PyTorch Gloo and collective timeouts](https://docs.pytorch.org/docs/2.7/distributed.html)

## Production measurements and records

Each attempt appends one bounded JSONL record per rank and update under
`train-attempts/<attempt>/rank-<rank>.jsonl`. Startup records runtime,
rendezvous, metadata and initialization costs. Update spans cover source
verification and read/unpack, feature construction, forward/loss, backward,
optimizer/scheduler, and the scalar commit collective. The last DDP
backward includes compute and gradient-reducer time, and may overlap
communication. The scalar collective and checkpoint barriers report their
own elapsed spans. Checkpoint records split model/optimizer serialization
from ZIP write, fsync and atomic publication. Every exposure of process
CPU, FD, RSS or Linux I/O counters is labeled per process. End-to-end
launcher wall time and simultaneous process-tree RSS use an external
sampled timeline. Missing counters are distinguishable from zero.

The canonical saved workload used distance 5, 60 rounds, 12 five-round
cores, five-round deployment buffers and physical error probability 0.003.
It kept the 256-wide, four-layer GRU and the default seed and initialization.
It used 96 saved trajectories in 12 shards of 8, batch 8, microbatch 1,
12 updates, Adam with learning rate 0.001 and StepLR with step size 4 and
gamma 0.5. Checkpoint saves were due at 0, 4, 8 and 12. The first three
baseline runs ran through the installed command after adding timing spans
and before modifying the training algorithm. The controlled comparison uses
fresh serial and DDP runs on the final implementation and reuses the same
manifest.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
nnwa generate runtime=local_cpu runtime.allocated_cpus=1 data.split=train \
  data.id=canonical data.shots=96 data.shard_shots=8
common=(runtime=local_cpu data.train_source=stored_shards data.id=canonical
  training.max_steps=12 training.batch_size=8 training.microbatch_size=1
  training.optimizer=adam training.learning_rate=0.001 training.scheduler=step
  training.scheduler_step_size=4 training.scheduler_gamma=0.5
  checkpoint.save_every_updates=4)
for C in 2 4 8; do
  taskset -c "0-$((C-1))" nnwa train "${common[@]}" runtime.allocated_cpus="$C" \
    runtime.intraop_threads="$C" run_name="serial-$C"
done
for C in 2 4 8; do
  taskset -c "0-$((C-1))" nnwa train "${common[@]}" runtime.allocated_cpus="$C" \
    training.execution=ddp runtime.processes="$C" runtime.intraop_threads=1 run_name="ddp-$C"
done
```

The local driver launches the installed `nnwa` script directly in
`nn_window_decoding`. Its elapsed wall time begins before launching
`taskset` and includes Python startup, filesystem work and process exits,
plus polling resolution. The monitor targets a 100 ms sample interval.
The largest sum of simultaneous live process RSS can count shared memory
more than once. Linux `RUSAGE_CHILDREN` deltas after reaping cover stage CPU
time. The supervisor/monitor can adopt an orphan resource tracker to reap
its time when needed. Rank terminal CPU snapshots remain separate. Linux
`/proc/self/io` fields are retained per rank. Reported rank-0 and peer I/O
originate from sibling ranks for DDP. The monitor records its own sampling
overhead separately by excluding itself from reported stage CPU and RSS.
The stages share CPUs 0–7 with the lightweight monitor. The measured
affinity sets contain SMT sibling pairs on the local i7-13700H under WSL2.

### Measured pre-DDP baseline

The instrumented baseline used Python 3.11.18, PyTorch 2.7.0+cpu, PyG
2.6.1, Stim 1.15.0 and NumPy 2.2.6. The initial checkout was
`b5be4f6d2dfc24cd8941684b164e25542346124c`; the baseline package source
digest after adding timing spans was
`bdbd5e62c707267ebef390cc97cc2cca80682fbc64b3fc5035f42885da038536`.
The canonical training manifest SHA-256 was
`171926f604f7538558d95c209f4bbc349a164e4cfbb9399b196559f97f790d01`.

| Baseline threads | Wall (s) | Reaped CPU (s) | Sampled RSS (MiB) | First update (s) | Later update median (s) | Four checkpoint saves (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 16.170 | 24.557 | 413.5 | 0.974 | 0.858 | 1.154 |
| 4 | 14.746 | 46.651 | 413.5 | 1.104 | 0.870 | 0.528 |
| 8 | 16.465 | 98.952 | 411.4 | 1.107 | 1.000 | 0.556 |

The first baseline recorded 98,557,952 OS-attributed bytes read, versus
32,768 and 12,288 in the later baselines. Each recorded about 148 MB
written, including spool writes and repeated checkpoint replacements.
Page cache conditions varied with run order. The first-run startup cost
and I/O remain included in the table.

### Numerical criteria

The production tolerances were fixed before the distributed experiments.
For weights they are relative `1e-4` and absolute `3e-5`; for Adam first
and second moments the relative tolerance is `2e-4` and the absolute
tolerances are `2e-7` and `2e-9`, respectively. The genuine distance-5
gradient tests use relative `2e-4` and absolute `1e-7`. Different layouts
change the order of float32 operations. The absolute terms cover values
near zero, where a relative-only criterion would magnify noise. Parameter
presence, Adam step counts, scheduler state, budgets, order and sampled
query choices are checked exactly. Same-topology deterministic replay is
tested for exact tensor, RNG and cursor equality.

### Correctness and recovery checks

Use one xdist worker for process tests and eight for the independent suite,
with the numerical environment shown above. The invocations run sequentially.

```bash
taskset -c 0-7 conda run --no-capture-output -n nn_window_decoding \
  python -m pytest -n 1 --dist=load --strict-markers --strict-config -m resource_process tests
taskset -c 0-7 conda run --no-capture-output -n nn_window_decoding \
  python -m pytest -n 8 --dist=load --strict-markers --strict-config \
  -m 'not resource_process and not performance and not cluster' tests
```

The scalar oracle checks the gradient of the exact global mean for two
ranks, local batch three and microbatches 2/1. It also checks the single
optimizer and scheduler transition. The distance-5 case uses a shard
boundary inside the global batch and exercises zero-event ranks and
updates. It compares serial, DDP world size one and DDP world size two,
including gradients, exact rank agreement, weights, Adam moments and step
counters. The same fixture evaluates the terminal distributed checkpoint
through inline and parallel G/F/S and validates its PBS artifact.

The replay case restores distinct per-rank Torch and Python draws, and
demonstrates exact tensor and input replay after a controlled pause and
after rank 1 is killed beyond the last durable save. Separate barriers
exercise exceptions before backward, a bounded collective timeout,
source checksum failure, and rank failure after successful Gloo teardown.
The launcher reports failure and the run receipt never records successful
completion for those cases. The late-failure case retains its fully saved
checkpoint. A finite source is rejected before run reservation if it is
too short for the requested budget.

The continuation case uses the same completed distance-5 DDP base
checkpoint for S+/W/D, with a fixed global budget and fresh Adam in every
arm. It checks the exact ordered trajectory/core audit, the dynamic
draw against the PCG64 stream, and rank-0-only terminal validation after
Gloo has closed. The same ordered dynamic inputs are checked against a
serial continuation. PBS render tests require one supervisor and reject
invalid allocations.

A developmental process run exposed a pause-time exit stall after Gloo
had been destroyed and the peer had written its final log record. The
launcher returned failure at the configured deadline and preserved the
checkpoint. The implementation now releases the reducer before group
destruction and exits the isolated rank only after the workflow has
closed resources. Explicit peer exit codes gate run publication.

### Controlled production comparison

The final package source SHA-256 was
`f966c40cfd66e13cb135d4295b60b9f1adfe82adc6a525a5e19ee24eb3fb4576`.
All six fresh cases completed 12 updates, consumed the 96 unique
trajectories in identical global order and made 192 query examples.
Their input audit, including the sampled cores and buffer fields, was
`e495c6aec840b44c89cc28d69283196f1c97fdc5c92fcb14dd514fbd1bd9c609`.
The optimizer step counters and scheduler matched exactly.

| CPUs | Execution | Wall (s) | Reaped CPU (s) | Sampled tree RSS (MiB) | First update, rank 0 (s) | Later median, rank 0 (s) |
|---:|---|---:|---:|---:|---:|---:|
| 2 | One process, two threads | 14.317 | 24.341 | 411.7 | 0.904 | 0.868 |
| 2 | Two one-thread ranks | 11.432 | 20.268 | 905.4 | 0.591 | 0.579 |
| 4 | One process, four threads | 14.844 | 46.171 | 409.7 | 1.020 | 0.907 |
| 4 | Four one-thread ranks | 7.704 | 25.986 | 1712.0 | 0.339 | 0.305 |
| 8 | One process, eight threads | 16.414 | 99.575 | 415.0 | 1.221 | 1.025 |
| 8 | Eight one-thread ranks | 8.620 | 53.391 | 3303.4 | 0.343 | 0.300 |

At four CPUs, median later feature/forward-loss times were 0.180/0.180 s
in the single process and 0.039/0.059 s on DDP rank 0. Median serial
backward was 0.481 s; DDP rank 0 recorded 0.079 s in earlier backwards
and 0.105 s in the final backward including synchronization. Its scalar
commit collective took 0.002 s at the median. These spans have different
boundaries and the DDP timing includes overlap. The largest observed
per-update rank wall-time spread was 0.019 s with four ranks and 0.025 s
with eight. Rank-0 runtime setup was 2.084 s for four ranks and 2.866 s
for eight. Eight ranks had slightly shorter later updates and longer
total wall time than four. Additional rank startup and memory were
visible in this finite workload.

| CPUs | Execution | Storage reads (bytes) | Storage writes (bytes) | Retained run files | Retained bytes |
|---:|---|---:|---:|---:|---:|
| 2 | Single | 15417344 | 148348928 | 4 | 23430262 |
| 2 | DDP | 4096 | 148533248 | 5 | 23463621 |
| 4 | Single | 0 | 148348928 | 4 | 23430232 |
| 4 | DDP | 0 | 148766720 | 7 | 23507477 |
| 8 | Single | 7667712 | 148348928 | 4 | 23430503 |
| 8 | DDP | 0 | 149209088 | 11 | 23592086 |

These are OS-attributed rank-side counters. They include spool and
replacement work and are distinct from retained bytes and logical read/write
counters. At four CPUs the terminal checkpoint was 23,410,568 bytes for
single-process training and 23,447,325 bytes for the four-rank run.
Every timed command left an empty process group.

The DDP pairs at two and four CPUs passed the predeclared weight and
Adam moment tolerances. The eight-CPU pair passed the moment tolerances
and had one weight beyond tolerance out of 1,945,537. The maximum
absolute weight differences for the 2/4/8 pairs were
`9.8832e-6`, `4.8727e-6` and `6.6537e-5`. The eight-CPU outlier was
`rnn.weight_ih_l2[630,89]` and its applicable tolerance was about
`3.543e-5`. The same coefficient differed by `7.6387e-5` between the
two-thread and eight-thread serial baselines before introducing DDP, with
the same pattern in the fresh serial controls. All three one-thread-rank
DDP checkpoints met the predeclared criteria against the two-thread
reference. The measured eight-CPU pair remains outside the stated
elementwise weight criterion.

The coefficient diagnostic used fresh single-process runs at one, two,
and eight intra-op threads on the same input. The coefficient started
at the same float32 value. Its first-step gradients were approximately
`1.536e-9`, `1.404e-9`, and `2.493e-9`, respectively, while Adam's
epsilon was `1e-8`. Independent scalar reconstruction matched each
observed Adam update within `8e-9`. The first Adam updates for two and
eight threads already differed by approximately `7.64e-5`, accounting
for the terminal discrepancy. The eight-rank checkpoint compared with
the one-thread diagnostic control passed the unchanged thresholds for
every weight and Adam moment. Its maximum weight difference was
`6.743e-7`. This isolates the observed sensitivity to changing
intra-op thread count on this workload. The primary timing row and its
weight mismatch are retained.

### Executed suites and retained evidence

The focused ordinary suite passed 40 tests, and the first focused DDP
suite passed four. Following the teardown correction, the cumulative
resource suite passed 23 tests in 225.33 s, and the final ordinary suite
passed 304 tests in 69.97 s. The environment emitted Matplotlib/PyParsing
deprecation warnings. `pip check` reported no broken requirements.

The local workspace sibling `../nnwa-local-evidence/phase3-ddp/` retains
`train_matrix.py`, `workload.json`, exact installed-command argv and exit
status, per-command process timelines, canonical dataset and checkpoints,
rank JSONL logs, inventory and test logs, and `matrix-comparison.json`.
The diagnostic traces and scripts are in the same directory. From the
repository, the drivers were invoked as follows, with the numerical
environment above. The final suite and measurements also used
`PYTHONUNBUFFERED=1`.

```bash
matrix=../nnwa-local-evidence/phase3-ddp
taskset -c 0-7 conda run --no-capture-output -n nn_window_decoding python "$matrix/train_matrix.py" baseline
taskset -c 0-7 conda run --no-capture-output -n nn_window_decoding python "$matrix/train_matrix.py" serial
taskset -c 0-7 conda run --no-capture-output -n nn_window_decoding python "$matrix/train_matrix.py" ddp
taskset -c 0-7 conda run --no-capture-output -n nn_window_decoding python "$matrix/compare.py"
taskset -c 0-7 conda run --no-capture-output -n nn_window_decoding python "$matrix/numeric_trace.py"
taskset -c 0-7 conda run --no-capture-output -n nn_window_decoding python "$matrix/check_diagnostic.py"
```

Each layout was measured once for this phase. Allocated-node Gloo and
PBS execution, shared-filesystem behavior, 96-CPU performance, and the
final multi-stage 2/4/8-CPU audit after parallel generation remain pending.
