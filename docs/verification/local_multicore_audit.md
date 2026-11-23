# Local multicore smoke and resource audit

## Fixed workload

The saved-data matrix uses the distance-5, 60-round rotated surface-code
Z-memory circuit with 12 five-round cores, buffers 5/5 and circuit noise
0.003. It keeps the 256-dimensional, four-layer GRU in CPU float32.
Base training processes 96 trajectories in 12 optimizer updates with
global batch 8, local microbatch 1 and two queries per trajectory.
Adam starts at 0.001 with the Phase 3 StepLR schedule, step size 4 and
gamma 0.5. Frozen G/F/S uses batch 1 and 129 held-out trajectories in 33
shards, including a short final shard. Matching is disabled. S+/W/D
uses a separate 32-trajectory dataset, four additional updates per arm
and four distinct validation shots.

The checked-in recipe fixes seed 101 and the shard sizes. The first
repetition runs 2, 4 and 8 CPUs, the second 8, 4 and 2. The controlled
training order changes from inline then DDP to DDP then inline. The
first two-CPU chain supplies the canonical training manifest, held-out
manifest and terminal checkpoint. Generation timings reuse the
identically configured chain stages and include fresh serial
references. Both repetitions use fresh output directories with no
cache flush or explicit prewarming. Each production command is run in
a fresh subprocess and receives the intended persisted artifact path.

The earlier eight-thread serial float32 discrepancy remains a numerical
exception. This audit measures both eight-thread training timings,
compares their exact input and optimizer controls, and uses a fresh
one-thread serial run on the same eight-CPU mask for the eight-rank
numerical comparison. The direct eight-thread versus eight-rank
floating-point tensor check is skipped under the previously stated
instruction. The published tolerances still apply to the numerical
comparisons that run. [Phase 3 diagnostic](cpu_ddp_training.md)

## Measurement and execution

The local run used WSL2 on an Intel Core i7-13700H and ext4, with a
7.67 GiB-visible host and one NUMA node. Linux exposed logical CPUs
0–19. The selected 0–7 CPUs were four SMT sibling pairs, and the
visible cgroup had no CPU bandwidth or memory limit. The
`nn_window_decoding` Conda environment used Python 3.11.18, PyTorch
2.7.0+cpu, PyG 2.6.1, Stim 1.15.0 (SSE2), NumPy 2.2.6 and SciPy
1.17.1. The checkout started from `ca7e3dcdb9913d63beedf1d41a8774f2ec8bb2a6`
with the audit changes in the working tree. `environment.json` records
the dirty status, script hashes, topology, limits and package source
SHA-256 `2cc972821283e2a2ce6fd7c2286f9bfd84d7356b521e587061f0c9233cb87e66`.

The monitor pins itself and the command tree to CPUs selected from the
effective Linux affinity and checks the visible cgroup CPU quota.
Generation and frozen evaluation workers use one numerical thread
each. Each DDP rank uses one intra-op and one inter-op thread, with one
CPU in its affinity mask. Inline training uses the allocated number of
intra-op threads and one inter-op thread. The driver sets the six
common native-library thread variables to one before launch. The inline
training runtime sets them to its configured intra-op count. Application logs
retain actual worker/rank settings and affinity, and process timelines
retain PID plus creation ticks, native thread counts and FD counts.

Elapsed time spans launch, imports, verification, work, durable
publication, and ownership cleanup. It includes the sampling/polling
overhead. The monitor samples about every 200 ms. Peak tree RSS is
the largest sampled simultaneous sum over live processes and can
count shared pages multiple times. Short memory, FD and file peaks
can fall between samples. Application spans use monotonic wall
clocks. Worker spans can overlap each other, and backward spans can
include Gloo time. The elapsed stage time measures the critical path.
The raw spans preserve
initialization, reads and unpacking, feature construction, forward,
backward and synchronization, optimizer work, checkpoint
serialization and publication, Stim sampling, packing and writes,
queue waits, commits and shutdown.

The launcher reaps the production CLI and any adopted descendants.
Its self CPU and waited-child CPU are counted once. Monitor CPU is
recorded separately, and the resource summary includes approximate
average CPUs including the monitor. The launcher's terminal inclusive
Linux I/O is counted once. Per-process terminal I/O is available for
breakdowns. Storage counters, logical character and syscall counts,
scientific payload bytes, retained artifacts and sampled staging files
are reported independently. Memory mapping and cached reads may
bypass these I/O counters. Storage counters describe OS attribution
on this machine. [Linux process I/O](https://man7.org/linux/man-pages/man5/proc_pid_io.5.html)

Stage scratch contains live read spools and disposable runtime caches.
The monitor tracks named files and open unlinked files, deduplicating
the latter by device and inode. It records named scratch at command
exit and clears the stage-owned scratch only after successful exit and
group reap. Training's atomic replacements and write spools in the
output tree are also sampled. Filesystem and proc collection is sampled.
Missing terminal I/O and missing process FD/affinity observations are
represented as unavailable. Each stage has a 180-second timeout, a
5,632 MiB sampled RSS budget and a 4 GiB minimum disk-free budget.

The first Phase 5 attempt stopped after a successful two-rank training
command because the scratch gate found eight generated runtime-cache
files totaling 50,546 bytes. The stage had a terminal receipt and an
empty process group. The monitor now records these files and then
removes the owned scratch directory. The failed audit attempt remains
under `../nnwa-local-evidence/phase5-audit/`. The fresh matrix starts
from the beginning with the corrected monitor. The scientific package
source is unchanged.

## Complete chains

Both repetitions completed all four stages at every CPU budget. Each
entry uses its own persisted training and held-out data and its own
terminal checkpoint. Sums add the independently measured wall times.

| CPUs | Rep | Generate train (s) | DDP train (s) | Generate held-out (s) | Evaluate (s) | Sum (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 2.33 | 11.06 | 3.09 | 33.49 | 49.97 |
| 2 | 2 | 1.87 | 11.07 | 3.11 | 30.71 | 46.76 |
| 4 | 1 | 1.89 | 9.56 | 2.71 | 21.15 | 35.31 |
| 4 | 2 | 2.08 | 8.62 | 2.48 | 20.29 | 33.48 |
| 8 | 1 | 1.89 | 10.44 | 2.31 | 18.67 | 33.31 |
| 8 | 2 | 1.91 | 12.33 | 2.59 | 18.61 | 35.43 |

Median total stage time fell from 48.37 s at two CPUs to 34.39 s at
four CPUs and 34.37 s at eight CPUs.

The canonical SHA-256 receipts record the exact three fixed inputs.

- Checkpoint — `41494ac5fb30267f77c7fe0e16128754c0e79191817753d2a8efee0f31b5a209`
- Held — `0a505a95ae5236a40a6c752a2061d5e89d1c554e1dcba05ed491fbf295765cf2`
- Training — `9514b3df11059f7d92138f90877fdb9f65c5a35e7134fb9cad236e72e0119ab6`


## Controlled timing

The two observations give the range and its midpoint median. Throughput
includes startup and shutdown. Speedup is the two-CPU median divided by
the tested median. Efficiency divides speedup by the ratio of the two
CPU budgets.

| Stage | CPUs | Rep 1 / 2 (s) | Median (s) | Trajectories/s | Updates/s | Speedup | Efficiency |
|---|---:|---:|---:|---:|---:|---:|---:|
| DDP training | 2 | 10.66 / 11.10 | 10.88 | 8.83 | 1.10 | 1.00 | 1.00 |
| DDP training | 4 | 8.21 / 9.12 | 8.66 | 11.08 | 1.39 | 1.26 | 0.63 |
| DDP training | 8 | 10.27 / 9.91 | 10.09 | 9.52 | 1.19 | 1.08 | 0.27 |
| Frozen evaluation | 2 | 30.33 / 31.19 | 30.76 | 4.19 | — | 1.00 | 1.00 |
| Frozen evaluation | 4 | 19.88 / 21.34 | 20.61 | 6.26 | — | 1.49 | 0.75 |
| Frozen evaluation | 8 | 18.19 / 17.97 | 18.08 | 7.14 | — | 1.70 | 0.43 |
| Held-out generation | 2 | 3.09 / 3.11 | 3.10 | 41.61 | — | 1.00 | 1.00 |
| Held-out generation | 4 | 2.71 / 2.48 | 2.60 | 49.69 | — | 1.19 | 0.60 |
| Held-out generation | 8 | 2.31 / 2.59 | 2.45 | 52.65 | — | 1.27 | 0.32 |
| Inline training | 2 | 15.13 / 14.68 | 14.90 | 6.44 | 0.81 | 1.00 | 1.00 |
| Inline training | 4 | 16.32 / 18.26 | 17.29 | 5.55 | 0.69 | 0.86 | 0.43 |
| Inline training | 8 | 18.02 / 19.01 | 18.52 | 5.18 | 0.65 | 0.80 | 0.20 |
| Training data generation | 2 | 2.33 / 1.87 | 2.10 | 45.80 | — | 1.00 | 1.00 |
| Training data generation | 4 | 1.89 / 2.08 | 1.99 | 48.32 | — | 1.06 | 0.53 |
| Training data generation | 8 | 1.89 / 1.91 | 1.90 | 50.55 | — | 1.10 | 0.28 |

The serial generation references took 2.27 and 1.84 s for training
data (median 2.06 s) and 4.99 and 3.30 s for held-out data (median
4.14 s). Their scientific outputs matched the parallel layouts.

## Controlled resource medians

CPU seconds count the launcher and its children. Average CPUs also
include the monitor. File and byte peaks are independent per-field
maxima across 200 ms samples. MB is decimal. Read scratch covers
checkpoint-load spools and runtime caches.

| Stage | CPUs | CPU (s) / avg CPUs | Peak RSS (MiB) | Storage read / write (MB) | Logical read / write (MB) | Retained files / MB | Peak temp files / MB | Read scratch peak (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Training data generation | 2 | 1.88 / 0.93 | 187.6 | 0.00 / 0.99 | 28.25 / 0.77 | 68 / 0.741 | 66.5 / 0.712 | 0.008 |
| Training data generation | 4 | 2.81 / 1.45 | 292.2 | 0.00 / 1.00 | 44.52 / 0.78 | 70 / 0.744 | 66 / 0.708 | 0.004 |
| Training data generation | 8 | 5.64 / 3.01 | 501.9 | 0.04 / 1.00 | 77.07 / 0.80 | 74 / 0.749 | 66 / 0.708 | 0.008 |
| Held-out generation | 2 | 1.77 / 0.60 | 187.8 | 0.00 / 1.46 | 28.37 / 0.86 | 173 / 0.818 | 171.5 / 0.751 | 0.008 |
| Held-out generation | 4 | 2.93 / 1.16 | 291.4 | 0.06 / 1.47 | 44.64 / 0.87 | 175 / 0.821 | 170.5 / 0.746 | 0.004 |
| Held-out generation | 8 | 6.12 / 2.54 | 503.1 | 0.07 / 1.48 | 77.19 / 0.89 | 179 / 0.827 | 171 / 0.746 | 0.000 |
| Inline training | 2 | 25.44 / 1.72 | 428.4 | 0.00 / 148.36 | 228.34 / 148.32 | 4 / 23.431 | 7 / 40.907 | 0.025 |
| Inline training | 4 | 53.93 / 3.14 | 430.2 | 0.00 / 148.36 | 228.34 / 148.32 | 4 / 23.431 | 7 / 46.847 | 0.025 |
| Inline training | 8 | 111.44 / 6.04 | 424.3 | 0.00 / 148.36 | 228.34 / 148.32 | 4 / 23.431 | 7 / 46.843 | 0.025 |
| DDP training | 2 | 19.83 / 1.84 | 901.9 | 0.00 / 148.57 | 298.31 / 343.17 | 5 / 23.465 | 11.5 / 31.356 | 0.051 |
| DDP training | 4 | 28.14 / 3.27 | 1719.5 | 0.00 / 148.79 | 424.32 / 732.80 | 7 / 23.509 | 20 / 46.997 | 0.102 |
| DDP training | 8 | 62.21 / 6.20 | 3314.5 | 0.00 / 149.27 | 676.35 / 1512.61 | 11 / 23.593 | 36 / 47.173 | 0.204 |
| Frozen evaluation | 2 | 53.87 / 1.78 | 1128.9 | 0.00 / 71.93 | 486.74 / 70.69 | 336 / 0.291 | 30 / 20.661 | 20.657 |
| Frozen evaluation | 4 | 64.66 / 3.16 | 1849.4 | 0.05 / 118.87 | 800.55 / 117.62 | 338 / 0.295 | 56 / 11.807 | 11.799 |
| Frozen evaluation | 8 | 106.28 / 5.92 | 3281.2 | 0.00 / 212.76 | 1428.18 / 211.48 | 342 / 0.302 | 108.5 / 44.280 | 44.280 |

Across all 60 measured commands, the largest sampled aggregate RSS
was 3,316.9 MiB during eight-rank DDP. Every command exited zero with an
empty owned process group. Successful commands left zero scratch
files after monitor cleanup, and the artifact trees had no generation
or evaluation staging directories. Before cleanup, generated caches reached
32 files and 202,280 bytes in one stage. The matrix process timelines
totaled 3,937,588 bytes, and application JSONL totaled 2,827,146 bytes.
The monitor used 17.75 CPU seconds across the matrix, with at most
1.13 CPU seconds for one stage. Its largest per-stage CPU-to-wall ratio
was 0.052 CPUs.

### Training

All base runs wrote checkpoints at updates 0, 4, 8 and 12. Inline runs
serialized 78.03 MB across four writes. The DDP totals for 2/4/8 ranks
were 78.11, 78.18 and 78.33 MB. Each run retained a terminal checkpoint
of about 23.4 MB, plus small run records and rank logs. About 148–149 MB
of OS-attributed writes per base command reflects spool files and
durable publication. Logical writes also include Gloo traffic through
file descriptors, which explains why DDP logical writes scale much
more rapidly than storage writes.

Each row below summarizes updates 3–12. A DDP update duration uses
the maximum elapsed time among its ranks. The first two updates
remain part of the identical measured work and update budget.

| Layout | CPUs | Rep 1 median and range (s) | Rep 2 median and range (s) |
|---|---:|---:|---:|
| Inline | 2 | 0.852 [0.830, 0.901] | 0.879 [0.826, 0.910] |
| Inline | 4 | 0.961 [0.799, 1.600] | 1.067 [0.742, 1.377] |
| Inline | 8 | 1.088 [1.021, 1.321] | 1.182 [1.048, 1.524] |
| DDP | 2 | 0.565 [0.548, 0.598] | 0.592 [0.556, 0.607] |
| DDP | 4 | 0.305 [0.290, 0.328] | 0.331 [0.310, 0.408] |
| DDP | 8 | 0.329 [0.275, 0.399] | 0.315 [0.271, 0.388] |

Within-update elapsed spread after the first two updates reached
0.4, 0.6 and 3.4 ms at 2/4/8 ranks. Those timings include the update
synchronization boundary. Per-rank CPU and span records are retained
for imbalances within the synchronization interval. In the first
repetition, the summed rank-0 backward span that included gradient
synchronization grew from 1.24 s at four ranks to 2.27 s at eight.
Rank-0 runtime initialization rose from 2.28 s to 3.65 s. This is
consistent with startup and synchronization limiting eight-rank DDP
on a fixed global batch of eight. Eight ranks used about twice the
sampled RAM and CPU seconds of four ranks. Inline later updates and
end-to-end times rose with thread count while the single process's
resident footprint stayed near 425 MiB. Both are bounded performance
limitations for this workload. The measurements leave the exact
division between runtime, SMT and synchronization effects unresolved.

### Generation and frozen inference

In the first two-CPU held-out generation run, the summed worker spans
were 3.15 s for shard payload and manifest writes and 0.012 s for
Stim sampling. Those sums overlap across workers. Together with the
measured stage wall times they put the small-shard cost primarily
around startup, verification and publication. The modest improvement
with more workers is a bounded performance result.

All frozen jobs read 74,035 bytes of scientific shard payload and
shard manifests and produced 118,775 bytes of committed prediction
files and receipts. Workers loaded their model once on startup and
closed the current shard before accepting the next. The coordinator
bound the queue to two tasks per worker. In repetition 1 the elapsed
time from first to last commit fell from 20.61 to 9.45 to 6.04 s at
2/4/8 workers. Wall time outside that interval rose to 12.15 s at
eight workers. It includes imports, model and checkpoint setup, early
work, final publication and shutdown. The first-repetition median
task time rose from 1.41 s to 2.16 s between two and eight workers.

The full training-state archive is spilled to a scratch file when it
exceeds the 16 MiB memory threshold. The coordinator loads it once
for verification, then each inference worker loads it once. Median
storage writes rose from 71.93 MB to 212.76 MB even though the retained
prediction tree stayed near 0.30 MB. In repetition 1 at eight workers,
the observed live scratch peak was 65.10 MB and the OS recorded
210.99 MB of cancelled writes. The five- and nine-load layouts' scratch
peaks also varied across repetitions, since short-lived files can
fall entirely between samples. The cumulative counters and per-worker
ready records confirm bounded startup write amplification. The
recorded one-time loads and task traces show no per-shard checkpoint
reloads.

## Scientific and compatibility checks

- **Generation and saved-data boundaries.** For each split and training
  phase, serial and parallel generation produced the same sampling
  recipes, dataset identities, physical file hashes and shard hashes.
  Full manifests retain their own execution provenance. Readback checked
  saved coverage, row identities and shot-level label XOR. Every base
  checkpoint points to its intended source manifest and records the
  same ordered 96 trajectories, core draws and 192 query examples.
- **Training controls.** Both repetitions at 2 and 4 CPUs passed exact
  input, cursor, scheduler and Adam step comparisons and the Phase 3
  tensor tolerances. Maximum absolute weight differences were
  `9.88e-6` and `4.87e-6`. At eight CPUs the timed layouts matched the
  exact controls. Both eight-rank checkpoints passed the tensor
  comparison against the new one-thread serial control, with maximum
  weight difference `6.74e-7`. The weight tolerances were
  `rtol=1e-4, atol=3e-5`. Adam moment tolerances were `rtol=2e-4` with
  absolute tolerances `2e-7` and `2e-9` for first and second moments.
- **Frozen evaluation and archive.** The seven frozen layouts had
  identical IDs, logits, thresholded predictions, shard receipts and
  global XOR. The held-out readback counted 74, 59 and 59 failures for
  G, F and S over 129 trajectories. The independently computed
  `(n10,n01)` discordances were `(41,56)` for F−G and S−G, and
  `(0,0)` for S−F. These counts check equality on the fixture. Analysis,
  compact export (62,034 bytes), extraction and replay preserved
  the rates, contrasts and primary decomposition.
- **Matched S+/W/D.** The four-rank continuation arms used the same
  parent checkpoint and input/core audit. Each reset Adam, ran four
  updates, and published a validated terminal checkpoint. Each recorded
  saves at 0 and 4, followed by the validated terminal replacement.
  Dynamic buffers were `(5,5), (5,5), (5,5), (5,3)`. All three terminal
  checkpoints reloaded for S-mode evaluation against the canonical
  129-shot held-out input.

## Sustained resource check

A separate four-worker run used seed 102 and 257 new held-out
trajectories across 65 shards. The generator and evaluator completed
all 65 tasks (3.36 s and 31.62 s). Each pool allocated 16 or 17 tasks
per worker. Each coordinator stayed at or below eight pending tasks.
Generation verification and evaluation each peaked at four active
source mappings per task. Evaluation reported release of those
mappings after every task. Both stage process groups exited empty.

After the first two tasks per worker, completion samples held
15 descriptors in both pools. Generation workers stayed between
52.36 and 52.61 MiB RSS across workers, and evaluation between
357.34 and 360.06 MiB. The largest within-worker RSS range for inference
was 2.72 MiB. These observations are consistent with persistent
runtime and model allocation followed by bounded shard ownership over
the exercised sequence. Longer run resource behavior remains outside
this test.

## Interruption and regression checks

The Conda environment passed 322 ordinary tests with eight one-thread
pytest workers in 103.87 s and 34 process-resource tests with one pytest
worker in 265.24 s. Both suites used the same scientific package source
as the audit and ran separately from it.

- **Evaluation.** The real distance-5 test interrupted a two-worker run
  after shard 1 committed out of order. Three workers resumed by
  submitting only shards 0, 2, 3 and 4. The existing receipt stayed
  byte-identical, coverage was exact, and the final prediction
  receipts and paired analysis matched the uninterrupted reference.
- **DDP.** A ready-file barrier selected rank loss during update 3,
  after the durable update-2 save. The checkpoint remained intact.
  Same-world-size replay matched model, optimizer, scheduler,
  cursor, counters, consumed inputs and RNG state exactly against
  the uninterrupted reference. The distance-5 gradient fixture
  also passed inline/equivalent DDP gradient and Adam checks, including
  an update with globally unused parameters.
- **Generation.** Stopping a worker or the coordinator at the
  partial-write barrier left no complete dataset. A fresh four-worker
  retry reproduced the serial scientific receipts and cleaned staging.
  The bounded queue fixtures checked completion-driven submission and
  small task and receipt messages for generation and evaluation.
- **Monitor.** The tests verified inclusive I/O and waited-child CPU
  accounting, sampled allocation and release, missing PID handling,
  and concurrent staging renames. A known 3,000,000-byte anonymous
  spool was sampled while open, disappeared on close, and the only
  named cache file (five bytes) was removed after reaping.

The exercised workflow and fault cases left no unresolved
correctness or resource-lifecycle blocker. The eight-thread
numerical exception remains scoped to the direct tensor comparison
described above. The startup, synchronization and I/O costs are
bounded limitations of the measured workload.

## Retained evidence and reproduction

The successful records are in the workspace sibling
`../nnwa-local-evidence/phase5-audit-final/`. `environment.json`,
`pip-freeze.txt` and `workload.json` capture the execution environment and
recipe. `canonical.json` and `inventory.json` bind input and output artifacts.
`stages.jsonl`, `stages.csv` and `trends.json` hold individual measurements and
the two-observation medians and ranges. `scientific_checks.json` contains
the readback and `resource_checks.json` contains span and lifecycle checks.
Each measured command has its arguments, terminal receipt, stdout and
process timeline under `measurements/`. The per-task and rank logs are
retained alongside their artifacts. The separate validation directory
contains pytest logs, exact validation commands and fault fixtures.

From the repository, use a fresh evidence directory for a new run.

```bash
audit=../nnwa-local-evidence/phase5-audit-new
conda run --no-capture-output -n nn_window_decoding \
  python -m scripts.local_multicore_audit "$audit"
conda run --no-capture-output -n nn_window_decoding \
  python -m scripts.summarize_audit_resources "$audit"
conda run --no-capture-output -n nn_window_decoding \
  python -m scripts.summarize_audit_resources "$audit" --sustained
```

The evidence includes `wall_time.png`, `throughput.png`, `memory.png`
and `io.png`. The local Phase 5 result covers the exercised 2/4/8 logical
CPU configurations. Allocated-node PBS, cluster I/O and 96-CPU behavior
remain unverified.
