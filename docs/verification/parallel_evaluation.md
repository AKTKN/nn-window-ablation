# Phase 2 parallel frozen evaluation

## Execution and resource bounds

`nnwa evaluate` consumes a sealed held-out manifest and a terminal checkpoint.
The same shard kernel runs inline with `evaluation.workers=1` and in a Linux
Python 3.11 spawn pool above one. Every task owns a complete shard, including
the last uneven shard. It reads fixed batches of detector rows and runs the
selected modes on the same IDs. Window features are built after cropping.
Recurrent state starts fresh for every query. Models run in inference mode
and their parameter and buffer digests are checked after each task. Optional
PyMatching uses the original global-only DEM.

The coordinator verifies dataset metadata, checkpoint and query provenance,
and the intended numerical settings before accepting any work. Each worker
loads one checkpoint/model, its query layout and optional MWPM decoder. It
also loads its own copy of the lightweight dataset descriptors and physical
metadata once. A worker opens and verifies the arrays of its assigned
source shard only. The coordinator opens labels and IDs during verification
of a new result. Resume and PBS verification also read and verify the full
source payload of every committed shard.

Let `W` be the worker count, `K` the per-worker intra-op threads, and `L`
the in-flight limit. The pool has at most `min(W,L,missing shards)` live
model owners. Each owns at most four source payload mappings, one input
batch plus its feature workspace, and one shard's prediction arrays. The
coordinator verifies one shard at a time, with at most three source label/ID
mappings during a new commit and one prediction file mapped at a time. The
memory needed for metadata is proportional to the number of shard descriptors
per process and to the compact inventory needed for the final manifest.
There are at most `L` submitted tasks awaiting commit and at most `L` staged
shard outputs, with one atomic file write in flight per active worker. The
bounded IPC descriptor is the index, attempt ID and numerical contract
digest. The result reports a staged receipt checksum, coverage, output size,
and worker identity and timing. Worker stage/resource details go directly
to files. The coordinator discards each future and result after validating
and publishing it.

Set `runtime.processes=1`, `data.num_workers=0`, and
`data.sampling_processes=1`. Inline evaluation uses the configured runtime
threads. In the pool, set coordinator intra-op and inter-op counts to one
and allocate at least `W * K` CPUs from the observed affinity. Workers get
disjoint sets of `K` CPUs, validate against their own sets, and run one
inter-op thread. The coordinator runs one numerical thread and pins
verification/publication to the first spare CPU when available. With
`W * K` CPUs it shares the first worker's CPU for those commits. Python
also uses queue-management threads and a resource tracker for process
bookkeeping. Inherited native-library thread values above `K` are rejected
before checkpoint or dataset work. The coordinator sets native limits for
itself and supplies the worker limits before Python spawn. Use an allocation
or a facility such as `taskset` to constrain the entire launched command.
[Python 3.11 process pools](https://docs.python.org/3.11/library/concurrent.futures.html)

## Resume and archive schemas

New runs write a schema-2 `contract.json` for a schema-3 prediction manifest.
The semantic digest binds the held-out manifest, label and feature schemas,
checkpoint bytes and model state, query/window and threshold policies, fixed
batch structure, device and dtype, effective worker threads, Torch numerical
flags, and source and dependency identity. CPU model and platform remain in
the numerical environment. Allocation, CPU affinity, worker count, maximum
in-flight count and order of completion are execution provenance. Changing
the batch size, precision, thread settings or numerical environment requires
a new run. Bitwise comparisons apply to the exercised fixed settings.

The coordinator holds the advisory run lock. Each attempt gets a UUID and a
private `.evaluation-staging/<attempt>/<index>/` for each submitted shard.
Workers stage the payload and receipt there. The coordinator validates the
returned identity and on-disk receipt, source record, checksums, array
shapes, ID order, finite logits, threshold, padding and core XOR. It then
promotes the directory by a same-filesystem rename and syncs both parent
directories. Commits may appear in any order. The coordinator writes
`manifest.json` after exact coverage, listing shards in index order. A
matching completed-run resume checks the existing coverage and payloads
without loading a model or submitting inference work. Schema-1 and schema-2
complete predictions remain readable by analysis and export. Use a fresh
evaluation name to produce a schema-3 run from an old run's saved inputs.
Compact archives carry the sealed prediction manifest and checksummed arrays
needed by reanalysis.

On a task error or SIGINT/SIGTERM, the coordinator stops replenishing the
queue. Already published shards remain committed, including a rename that
succeeded before a later error. The CPython 3.11 pool cancels queued work and,
on failure, terminates active workers. Shutdown allows five seconds for the
manager to finish, then uses SIGKILL and a further five seconds to join when
needed. Linux parent-death signals stop workers after a coordinator SIGKILL,
including a PPID check to cover a death during worker startup. Workers hold
shared leases while they can write. A resumed coordinator must obtain an
exclusive lease before removing stale attempt files and fails explicitly
if a writer still holds a lease. A normal CLI exit releases its worker pool
and the multiprocessing resource tracker exits when its inherited pipe
closes. The coordinator never marks incomplete output as complete.
[CPython 3.11 executor lifecycle](https://github.com/python/cpython/blob/3.11/Lib/concurrent/futures/process.py) · [Linux parent-death signals](https://man7.org/linux/man-pages/man2/PR_SET_PDEATHSIG.2const.html)

## Instrumentation

Each attempt writes `attempts/<attempt>/coordinator.jsonl` and one
`worker-<pid>.jsonl` per spawned worker. The inline path records task events in
the coordinator file. These logs carry monotonic timestamps, CPU allocation
and affinity, workers and thread settings, pending count, loaded model
count, input rows and payload bytes/files. Task timings cover source
verification/open, input read/unpack, features and forward by mode,
serialization, and model-state verification. Coordinator timings include
queue wait, staged verification, publication, shutdown and total workflow
time. Normal worker exit records provide terminal process counters. After a
forced exit, the latest available worker event is the last observation.

Process snapshots provide PID plus start ticks, RSS, descriptors, native
threads, self user/system CPU and Linux `/proc/<pid>/io` counters when
available. The coordinator's terminal I/O can include reaped children.
Keep its inclusive I/O totals and worker breakdowns separate to avoid
double-counting. Treat per-process RSS snapshots independently or sample
simultaneous process-tree RSS for an aggregate. Missing counters are JSON
null. The logs are diagnostic records outside the scientific digest and
outside the required archive contents. They can be passed as optional export
logs.

## Executed local checks

The checkout started at `ee9c82060030e70a479fb22110c0ef58aa888740`. Tests
ran in Conda `nn_window_decoding` using Python 3.11.18 and the CPU Torch
build. The installed editable source was used. All six native thread
environment values were one and `PYTHONHASHSEED=0`. The invocations ran
sequentially within CPUs 0–7.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0

taskset -c 0-7 conda run --no-capture-output -n nn_window_decoding \
  python -m pytest -n 1 --dist=load --strict-markers --strict-config \
  -m resource_process tests/integration/test_parallel_evaluation.py \
  -k 'serial_two or replenishes or coordinator-kill'
taskset -c 0-7 conda run --no-capture-output -n nn_window_decoding \
  python -m pytest -n 8 --dist=load --strict-markers --strict-config \
  -m 'not resource_process and not performance and not cluster' tests
taskset -c 0-7 conda run --no-capture-output -n nn_window_decoding \
  python -m pytest -n 1 --dist=load --strict-markers --strict-config \
  -m resource_process tests
```

The focused spawned checks passed 3 tests in 45.57 s. The cumulative suites
passed 291 ordinary tests in 87.43 s and 17 resource tests in 145.32 s.
Matplotlib emitted PyParsing deprecation warnings.

- The genuine distance-5 fixture used the 256-wide, four-layer GRU, five
  shards, nine held-out trajectories, a zero-event row, an uneven final
  shard, G/F/S and MWPM. One, two and four workers produced byte-identical
  prediction files. Independent whole-shot failures and paired discordances
  matched analysis. Worker startup records showed one actual model load
  per worker. Task records showed four active mappings and the same FD
  count immediately before and after completed tasks.
- A partial-write barrier held shard 0 while shard 1 committed. Shard 2
  was submitted and reached its own barrier before shard 0 was released.
  The pending count stayed at or below two and the sealed inventory
  was canonical. IPC checks covered JSON-compatible descriptors and
  compact receipts. Failed writes and model mutations closed the mappings
  without publishing the task.
- Worker SIGKILL, coordinator SIGTERM and coordinator SIGKILL after shard
  1's commit preserved that noncontiguous subset. Each resumed from two to
  three workers and submitted only the missing indices. No owned process
  group remained after the test harness reaped the abruptly orphaned
  processes. An active lease and a concurrent coordinator both blocked
  recovery from removing active staging. Faults immediately before and
  after promotion confirmed when the on-disk transaction could be reused.
- Tests rejected corrupted receipts, prediction payloads, task identities,
  and committed outputs. PBS artifact verification and analysis rejected
  incomplete runs. Analysis, compact export and reanalysis checked both
  completed parallel and resumed output. A completed-run resume submitted
  zero tasks and preserved the manifest bytes.

## Installed-command smoke on 2026-11-23

The smoke used Python 3.11.18, Torch 2.7.0+cpu, PyG 2.6.1, Stim 1.15.0,
PyMatching 2.2.2, NumPy 2.2.6 and SciPy 1.17.1 on an i7-13700H under WSL2.
`pip check` reported no broken requirements. The working tree's package
source SHA-256 was
`dae7c16d834d5ac77114309902c96c828713ee380236009743f2e95ce7cb22b0`.

The installed commands generated two saved training trajectories, ran one
SGD update with global batch two and microbatch one, and generated 17 held-out
trajectories in shards of 4/4/4/4/1. Independent fresh evaluation runs used
one, two and four CPUs with one thread per worker and batch size two.
All three runs evaluated the same held-out manifest
`19dcfa3ace47240d7410d743242c7fcd6fe80bcfec9663a36761eaa8c2ee85da`
and terminal checkpoint
`90bdfdd8a6ab5e69aa617513fe948415288eb40ce72729b2fe2694ce6a4ac2a4`.
The input and prediction IDs, logits, decisions and XOR matched exactly,
including MWPM decisions. Independent counts gave six failures each for
G/F/S, zero for MWPM, and zero discordances for the three G/F/S contrasts.
The workload checks coverage and reproducibility at a fixed short budget.

| Installed command | Allocated CPUs | Wall (s) | Reaped tree CPU (s) | Sampled aggregate RSS (MiB) |
|---|---:|---:|---:|---:|
| Generate training data | 1 | 2.056 | 1.867 | 224.1 |
| Train terminal checkpoint | 1 | 3.681 | 3.539 | 380.6 |
| Generate held-out data | 1 | 2.359 | 1.939 | 223.8 |
| Inline G/F/S and MWPM | 1 | 10.246 | 9.502 | 425.5 |
| Parallel G/F/S and MWPM | 2 | 11.016 | 17.143 | 1140.0 |
| Parallel G/F/S and MWPM | 4 | 9.407 | 24.233 | 1891.2 |
| Analyze four-worker output | 1 | 4.213 | 4.088 | 379.2 |

Wall time starts before launch and includes imports, verification, publication,
exit and up to roughly one poll interval. RSS is the largest sampled sum across
live processes in the owned process group, with a 100 ms target interval and
shared pages counted in each process's RSS. The monitor ran on CPU 7 outside
the measured stage CPU sets. CPU time uses Linux `RUSAGE_CHILDREN` deltas
after reaping, including any adopted resource tracker and without adding
worker self times. Every stage exited zero with an empty owned process group.

The coordinator's inclusive storage counters showed reads of 0/0/4096 bytes
and writes of 344064/393216/462848 bytes for one/two/four workers. These are
OS-attributed counters under ordinary page-cache conditions. Raw worker I/O
snapshots are retained separately. The runs retained 59/61/63 files totaling
79915/83417/86697 bytes, including 12104/15606/18886 bytes of attempt logs.
The observed outstanding-task peaks were 1/4/5. The two-worker run took
longer than inline while using more CPU time and RSS. The four-worker run
finished earlier while using the most CPU time and RSS. Startup, repeated
checkpoint loading across workers, and a small shard count are visible in
the raw records. This single pass does not estimate sustained throughput.

Raw evidence is in the local workspace sibling
`../nnwa-local-evidence/phase2-2026-11-23/`. It contains the runnable
`run_smoke.py`, per-command argv and exit/RSS/CPU JSON, sampled process
timelines, environment inventory, inputs, checkpoints, prediction files,
production JSONL, and `comparison.json`. The smoke driver was run from the
repository with the thread exports above.

```bash
taskset -c 0-7 conda run --no-capture-output -n nn_window_decoding \
  python ../nnwa-local-evidence/phase2-2026-11-23/run_smoke.py
```

Allocated-node PBS execution, shared-filesystem durability, 96-CPU throughput,
and the final 2/4/8-CPU multicore audit after training and generation
parallelism remain unverified.
