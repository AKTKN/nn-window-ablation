# Local 2/4/8-CPU saved-data audit

## Workload and execution order

The measured workload uses the distance-5, 60-round rotated-memory
circuit with 12 five-round cores, buffers 5/5, equal circuit-noise rate
0.003 and the 256-dimensional, four-layer GRU. The base dataset has
96 trajectories, in shards of 8. Training uses 12 global updates with
batch 8 and local microbatch 1. Adam uses learning rate 0.001 with the
Phase 3 StepLR schedule, step size 4 and gamma 0.5. Checkpoint writes are
due at updates 0, 4, 8 and 12. The frozen primary evaluation uses
G/F/S, inference batch 1 and 129 held-out trajectories in 33 shards,
including one partial shard. A continuation run uses one shared,
separate set of 32 trajectories, with four additional updates for each
of S+, W and D, plus four distinct validation shots.

The driver in `scripts/local_multicore_audit.py` reads the checked-in
`scripts/local_multicore_audit.json` recipe. It launches all four
installed commands for each complete chain. The first repetition uses
CPU order 2, 4, 8 and the second uses 8, 4, 2. It reuses the generation
measurements from the complete chains and runs two fresh serial
generation controls. All controlled training runs share the first
complete chain's saved training manifest. All controlled frozen
inference runs share that chain's exact held-out manifest and its
terminal checkpoint. Controlled training uses serial then DDP in the
first repetition and DDP then serial in the second. No cache flush or
prewarming is applied. The driver gives every measured stage its own
output name and executes them sequentially.

At the user's request, the two eight-thread serial training benchmark
rows and their direct eight-thread/eight-rank numerical checks are
omitted. Phase 3 already reproduced the single GRU coefficient that
crossed the agreed float32 tolerance when changing the serial thread
count. The eight-rank chain, eight-rank controlled training and
same-thread-count rank comparisons are included.
[Phase 3 diagnostic](cpu_ddp_training.md)

## Measurement boundaries

The driver uses the installed `nnwa` from the `nn_window_decoding` Conda
environment. It selects CPUs from effective Linux affinity and enforces
the selected set in the command tree, including its launcher and
monitor. Stage time starts before launching the wrapper and ends after
reaping. It includes application imports, startup, input verification,
writes, synchronization, publication and cleanup. Per-task and per-rank
JSONL records decompose application stages using monotonic time. A
backward span that includes Gloo may overlap gradient communication.
Per-process or per-task durations are kept separate from elapsed stage
time.

The monitor samples every 200 ms. The aggregate memory field is the
largest observed simultaneous sum of process RSS and can count shared
pages more than once. PID plus start ticks identifies individual
processes. Production workers record end-of-life CPU and I/O
observations when they exit normally. The stage wrapper adopts and reaps
descendants and emits its own CPU time plus CPU time from waited
children, and its final inclusive `/proc/self/io`. The monitor records
its own overhead separately. Rank and worker I/O observations are kept
as breakdowns to avoid adding children to their inclusive stage total.
Linux storage counters, logical I/O counters, and retained artifact size
measure separate quantities. Snapshot peaks include sampled named and
unnamed checkpoint staging files when observable. Memory and output
peaks between polls may be missed.
[Linux process I/O semantics](https://man7.org/linux/man-pages/man5/proc_pid_io.5.html)

The configured per-stage wall limit is 180 s. The monitor caps sampled
aggregate RSS at 5,632 MiB and keeps at least 4 GiB of disk free. It
terminates only the owned process group at a limit and retains its
command, timeline and failure record. The monitor test uses a short
waited-for CPU child, a sampled allocation and release, known logical
writes, and a nonexistent PID query. It checks inclusive totals without
adding the child's counters again.

## Retained monitor failure and correction

The first audit attempt passed the early stages and then the monitor
encountered a `FileNotFoundError` while traversing a sampler's temporary
shard directory during its atomic rename. The monitor stopped the owned
stage with SIGTERM. The eight-worker held-out command had no published
manifest and its generation staging directory was removed. The original
command, Python traceback, process timeline and application logs are
retained under `../nnwa-local-evidence/local-multicore-audit/` and
`../nnwa-local-evidence/phase4-generation/audit-driver.log`.

The monitor now tolerates a directory disappearing between listing and
descent, and records sampling failures with stage state. Both focused
monitor tests passed in 3.48 s after the change, including a stress case
with 500 staging renames. The final measurement uses fresh directories
and the corrected monitor for every layout. The numerical package
source was unchanged between these audit attempts.

## Complete chains

The audit used Python 3.11.18, PyTorch 2.7.0+cpu, PyG 2.6.1, Stim
1.15.0 (SSE2), NumPy 2.2.6 and SciPy 1.17.1 under the documented Conda
environment. The WSL2 filesystem was ext4. Logical CPUs 0–7 were
available and selected, with pairs of sibling CPUs in the reported
topology. The visible root cgroup bandwidth was unlimited.
The tested package source SHA-256 was
`2cc972821283e2a2ce6fd7c2286f9bfd84d7356b521e587061f0c9233cb87e66`.

Both four-command chains succeeded at each CPU budget. The sum below
adds their four independently measured stage times.

| CPUs | Repetition | Generate train (s) | DDP train (s) | Generate held-out (s) | Evaluate (s) | Sum of stages (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 1.847 | 11.242 | 3.298 | 32.578 | 48.966 |
| 2 | 2 | 2.041 | 10.825 | 3.096 | 30.977 | 46.939 |
| 4 | 1 | 1.845 | 9.017 | 2.471 | 20.530 | 33.862 |
| 4 | 2 | 1.849 | 8.629 | 2.472 | 19.983 | 32.933 |
| 8 | 1 | 1.886 | 9.205 | 3.546 | 17.167 | 31.804 |
| 8 | 2 | 1.705 | 8.985 | 2.331 | 17.737 | 30.759 |

The canonical references were the first two-CPU chain's training manifest
`283a081e6dbf1b8a51e4e133937f1aa189c71e0be63dc239ba54e5cda98a150a`,
held-out manifest
`c8a8e72076cd96d94320f48567c773b636ceb6564bb067ed4fe2974c53cc98c2`
and terminal checkpoint
`dc40cec644f7cfd4b31207269a922b7330217f109311a1cc3c7c60092065ebf1`.

## Sustained shard ownership

The first 33-shard two-CPU held-out generation apportioned 16 and 17
tasks to its workers. After their first two tasks, each held 15
descriptors and roughly 52.5 MiB RSS at task completion, with a constant
observed RSS within each worker. The coordinator reached four pending
tasks and four active payload mappings. Each verified shard produced
five files.

The first 33-shard fixed-input two-CPU evaluation also apportioned 16
and 17 tasks. After the first two tasks per worker, completion samples
held 15 descriptors with approximately 358–361 MiB RSS. Each task
opened four source mappings and closed them before reporting completion.
The coordinator reached four pending tasks. These finite observations
show stable FD counts and a small RSS range after startup on this
workload.

## Controlled measurements

The table gives both observations and their midpoint median. The CPU time,
sampled RSS and storage writes columns also show the two-run medians.
The eight-thread serial row was skipped as described above.

| Stage | CPUs | Wall observations (s) | Median wall (s) | CPU (s) | Sampled tree RSS (MiB) | Storage writes (MB) |
|---|---:|---|---:|---:|---:|---:|
| Generate train | 2 | 1.847, 2.041 | 1.944 | 1.503 | 187.3 | 0.987 |
| Generate train | 4 | 1.845, 1.849 | 1.847 | 2.485 | 291.6 | 1.004 |
| Generate train | 8 | 1.886, 1.705 | 1.796 | 5.373 | 501.2 | 1.004 |
| Generate held-out | 2 | 3.298, 3.096 | 3.197 | 1.820 | 186.8 | 1.460 |
| Generate held-out | 4 | 2.471, 2.472 | 2.471 | 2.620 | 292.2 | 1.466 |
| Generate held-out | 8 | 3.546, 2.331 | 2.939 | 5.974 | 502.1 | 1.483 |
| Inline train | 2 | 14.473, 14.905 | 14.689 | 24.689 | 425.4 | 148.361 |
| Inline train | 4 | 18.305, 18.336 | 18.321 | 56.379 | 423.1 | 148.361 |
| DDP train | 2 | 10.623, 11.232 | 10.928 | 19.194 | 900.7 | 148.566 |
| DDP train | 4 | 9.306, 9.844 | 9.575 | 29.407 | 1718.6 | 148.791 |
| DDP train | 8 | 9.467, 10.239 | 9.853 | 57.885 | 3315.1 | 149.266 |
| Frozen G/F/S | 2 | 31.932, 30.841 | 31.387 | 54.619 | 1126.8 | 71.934 |
| Frozen G/F/S | 4 | 20.761, 21.365 | 21.063 | 64.780 | 1845.6 | 118.870 |
| Frozen G/F/S | 8 | 17.044, 17.955 | 17.500 | 103.169 | 3278.8 | 212.781 |

Frozen evaluation processed approximately 4.11, 6.12 and 7.37
trajectories/s. The speedups from two CPUs were 1.49 at four and 1.79
at eight, with efficiencies of approximately 0.75 and 0.45. Controlled
DDP processed 1.10, 1.25 and 1.22 updates/s, or about 8.79, 10.03 and
9.74 trajectories/s. Its speedups from two CPUs were 1.14 and 1.11.
The corresponding DDP average CPU utilizations were about 1.76, 3.07
and 5.87 logical CPUs. Frozen evaluation averaged about 1.74,
3.08 and 5.89 CPUs. Per-stage records retain the individual CPU and
wall observations and the separate monitor overhead.

For this small generation workload, the pool's startup and the fixed
manifest and filesystem work occupy a substantial fraction of wall
time. Held-out generation was slower at eight workers than at four
in the median. Inline training was slower at four threads than at
two in this run. Eight ranks had higher DDP memory and CPU use with
slightly longer wall time than four. These are bounded performance
costs on this local workload.

The first controlled DDP repetition wrote checkpoints at updates 0, 4, 8
and 12 at every CPU budget. Their cumulative serialized sizes were about
78 MB per run, with one terminal checkpoint of about 23.4 MB retained.
For rank 0, later update medians at 2/4/8 ranks were 0.560/0.377/0.302 s.
The largest within-update rank wall-time spreads were
0.017/0.011/0.031 s. Its final backward spans including synchronization
were 0.101/0.135/0.193 s at the median. The shorter later updates at
eight ranks coexisted with a higher startup-inclusive median than four.

The sampled eight-rank training peak included two temporary files
totaling 46,954,386 bytes. Its run retained 11 files and 23,593,481
bytes. On the first eight-worker held-out generation command the
sampled staging peak was 171 files and 755,567 bytes. The published
dataset and its scoped attempt logs retained 179 files and 826,515
bytes. Frozen evaluation's staged prediction files peaked at 21/41/81
files and 7,218/14,436/28,872 bytes in the first 2/4/8-worker repetition.
Its corresponding retained prediction and attempt files were
336/338/342 and about 0.29–0.30 MB.

Frozen evaluation's storage write counters increased with worker count.
Checkpoint loading spills the full state to a temporary file once in
the coordinator and once per worker when it exceeds the 16 MiB spool
threshold. The first repetition's cancelled-write counters were about
70/117/211 MB for 2/4/8 workers, consistent with 3/5/9 loads of the
roughly 23.4 MB checkpoint. This is bounded startup and I/O amplification.
The prediction staging figures above cover the run's output tree.
Checkpoint read spools in system temporary storage are outside that
file-count scope. The root inclusive I/O measures their OS-attributed
activity. The raw traces retain read bytes, write bytes, logical bytes
and syscall counts separately.

## Scientific readback and disposition

All 57 measured commands exited zero and had empty owned process groups
on exit. Across serial generation and two, four and eight workers, the
same split and phase produced identical sampling recipes, IDs, physical
receipts and shard receipts. Each chain used its own exact generated
manifest and its own terminal checkpoint. Base training consumed the
same ordered 96 trajectories and sampled cores in 12 updates and
produced 192 query examples per run.

Both controlled repetitions met the Phase 3 weight and moment criteria
in the two- and four-CPU pairs. Maximum weight differences were
`9.8832e-6` and `4.8727e-6`, respectively. The two eight-rank checkpoints
were compared with the two-rank reference, with one intra-op thread per
rank in both layouts. Their maximum weight difference was `5.6997e-7`,
within the same criteria. The direct eight-thread serial comparison
was skipped in both repetitions.

All seven fixed-input inference layouts, including the inline reference,
had identical prediction receipts and validated payloads. Independent
whole-trajectory counts were 74, 59 and 59 failures for G, F and S over
129 rows. Signed discordance counts `(n10,n01)` were `(41,56)` for F−G
and S−G and `(0,0)` for S−F. Exact ID coverage, the finite logits and
threshold, and the core XOR were checked from saved arrays. Compact
archive replay reproduced the original rates, contrasts and primary
decomposition. These counts check equality on this short workload.

The four-rank S+/W/D jobs shared one parent and one continuation
manifest, matched the input/core audit, and had fresh Adam state with
four optimizer steps. D drew `(5,5)`, `(5,5)`, `(5,5)`, `(5,3)`
in update order. All three reloaded and evaluated S-mode on the canonical
test trajectories. The separate validation split had four shots per arm.

The current-source process suite also exercised generation worker loss
and coordinator loss, genuine distance-5 out-of-order evaluation resume
at a new worker count, and genuine distance-5 DDP rank loss after an
unsaved update. The generation retry reproduced the serial byte
receipts. Evaluation reused the already committed shard and reproduced
the reference predictions and statistics. DDP same-topology replay
matched terminal tensors and control/RNG state exactly. The monitor
rename race was corrected and the entire timing matrix rerun. The
observed startup, memory and I/O amplification are bounded on the
exercised workload.

## Retained evidence

The successful audit is in the local workspace sibling
`../nnwa-local-evidence/local-multicore-audit-final/`. Its
`environment.json`, `workload.json`, `canonical.json`, and
`inventory.json` record the source, environment, work and inputs.
`stages.csv` and `trends.json` contain measurements, `scientific_checks.json`
contains readback, and the stage timelines totaled 3,585,347 bytes.
`wall_time.png`, `throughput.png`, `memory.png` and `io.png` show the
startup-inclusive local trends. Raw per-task and rank logs are stored
with their generation data roots or run directories.

```bash
taskset -c 0-7 conda run --no-capture-output -n nn_window_decoding \
  python -m scripts.local_multicore_audit ../nnwa-local-evidence/local-multicore-audit-final
```

The local audit is complete for the exercised layouts and stated skip.
Allocated-node PBS execution, filesystem behavior on the cluster,
and 96-CPU performance remain unverified.
