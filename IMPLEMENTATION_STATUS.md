# Implementation status

Last updated 2026-11-23. P0 through P7 are implemented and verified locally.

## Delivered structure

The repository provides the installed `nnwa validate` workflow, typed
configuration, packaged Hydra presets, and CPU runtime checks. The simulator
builds the original and annotated Stim circuits and samples aligned detector
bits and logical labels. The window pipeline selects permitted raw detector
bits and batches Lange-compatible graphs with label-independent metadata.
The shared query decoder supports CPU forward/backward steps. P4 adds immutable
datasets, fixed-update training, and exact CPU checkpoint replay. P5 uses one
frozen checkpoint for paired G/F/S and reports whole-trajectory statistics.
P6 runs matched S+/W/D continuation and verifies the controls in paired analysis.
P7 renders PBS jobs, records execution evidence, and reconstructs analyses
from verified portable bundles.
The reliability update adds bounded dataset readers, periodic checkpoints,
and transactional evaluation resume.
Multicore Phase 2 adds a bounded Linux spawn pool for frozen, saved-shot
evaluation and verified resume across worker counts.
Multicore Phase 3 adds synchronous CPU/Gloo training on saved shards with
a global update budget and same-topology checkpoint replay.
Multicore Phase 4 adds deterministic, bounded Stim generation for persisted
training, validation and test shards.

## GitHub publication

On 2026-11-23, the initial scaffold and phase prompts were committed and pushed
to `main` in the private GitHub repository [AKTKN/nn-window-ablation](https://github.com/AKTKN/nn-window-ablation).
GitHub reported `main` as the default branch, and `git ls-remote` returned the
initial local commit `dbf1b742457a412b5aa5834409d14bf31f76e58b` for `main`.
Before committing, all 52 entries in the original scaffold manifest matched their
recorded sizes and SHA-256 hashes. The manifest and check log describe that
original snapshot. The publication checks also parsed the 13 Python placeholders
and package metadata, and both PBS templates passed `bash -n`.

The `prompts/` directory is now ignored by Git. All 11 local prompt files were
checked byte for byte after removing them from the Git index.

## Phase status

| Phase | Scope | Status |
|---|---|---|
| P0 | Packaging, CLI, typed configuration, validated CPU environment | Implemented and verified locally. Compute-node validation pending. |
| P1 | Physical circuit, clean cuts, exact global/core labels | Implemented and verified locally on the finite fixtures. |
| P2 | Observation windows and features without future leakage | Implemented and independently verified locally. |
| P3 | Lange model adapter, shared queries, CPU profile | Implemented and independently verified locally. Allocated-node profile pending. |
| P4 | Datasets, training, checkpoints, deterministic resume | Implemented and independently verified locally. Allocated-node checks pending. |
| P5 | Paired G/F/S evaluation and uncertainty | Implemented and independently verified locally. Main-study choices pending. |
| P6 | Matched S+/W/D continuation | Implemented and independently verified locally. Allocated-node and campaign checks pending. |
| P7 | PBS workflows, export, and local reanalysis | Implemented and independently verified locally. Allocated-node acceptance pending. |

The phase contract is [05_implementation_phases.md](docs/05_implementation_phases.md).
The scientific pilot is $d=5$; a small $d=3$ unit fixture is permitted in P1 but is
not a required training stage or the pilot.

## Validation record

### Training update log cadence and Japanese operating guides

All four training presets default to one detailed update event per rank every
100 phase updates. Preflight rejects more than 1,000 projected update events
unless `training.force_large_log=true`. Checkpoint and attempt events use their
own schedules. The local resource audit requests every update explicitly.

In the `nn_window_decoding` environment, the ordinary suite passed 329 tests
(`-n 8 -m 'not resource_process and not performance and not cluster'`), and
the process-resource suite passed 34 (`-n 1 -m resource_process`). The tests
covered both sides of the limit, the override, and per-rank cadence with
one and two Gloo ranks. A focused checkpoint test also passed after changing
the interval across a pause from two to four. Model, optimizer, RNG, counters
and continuation audit identities matched the uninterrupted reference.

The [Japanese setup walkthrough](docs/ja/experiment_setup.md) ran locally
through training on eight stored trajectories and paired evaluation and
analysis on five independent validation trajectories. The four-step run
logged updates 2 and 4. Every command in the
[96-CPU guide](docs/ja/cluster_96cpu.md) passed its configuration preflight.
The installed train CLI rejected 49,920 projected rows before creating the
run. The validate CLI accepted both the 960-row plan and the explicit
large-log override. Throughput and peak memory on a 96-CPU node remain
unmeasured. [Command and parameter reference](docs/ja/cli_parameters.md)

### Multicore Phase 4

The generation checks passed in the cumulative suites of 322 ordinary and
32 process-resource tests. The distance-5 generator produced identical
scientific payloads in the one-, two-, and four-worker layouts. Writer and
coordinator termination left no partial complete dataset, and retry produced
the reference payloads. The suites also exercised distance-5 DDP replay and
out-of-order evaluation recovery under the current package source.
[Parallel generation verification](docs/verification/parallel_generation.md)

### Multicore Phase 5

Both four-command saved-data chains completed at each of 2, 4 and 8 CPUs.
All 60 measured commands passed, including fixed-input comparisons, compact
archive replay and matched S+/W/D continuation. The separate four-worker
65-shard check showed stable descriptor counts and released its mappings.
The final Conda suites passed 322 ordinary and 34 process-resource tests,
including real distance-5 recovery after evaluation, training and generation
interruptions. Successful measured stages left no owned processes or scratch
files. Four-rank DDP had the lowest controlled training median (8.66 s), and
eight-worker evaluation was 1.70× as fast as two workers on this fixture.

The two eight-thread inline training runs are included in the timings. The
direct eight-thread/eight-rank tensor check was skipped for the known Phase 3
float32 discrepancy. Exact input and optimizer controls matched, and both
eight-rank checkpoints met the unchanged tensor tolerances against a fresh
one-thread serial control. Allocated-node and 96-CPU validation remain pending.
[Local multicore audit](docs/verification/local_multicore_audit.md)

### Multicore Phase 3

The CPU/Gloo training checks passed 304 ordinary tests and 23 resource
tests. The distance-5 checks exercised globally unused convolution
parameters, matched continuations, and evaluation of a terminal DDP
checkpoint. Same-topology pause and rank-loss replay restored model,
optimizer, scheduler, cursor, and rank RNG state exactly.

The measured 2/4/8-CPU serial and DDP runs used one canonical stored
dataset and matched all 96 trajectory identities, sampled core choices,
and 12 global updates. The paired 2- and 4-CPU runs met the predeclared
weight and Adam tolerances. The 8-CPU pair exceeded the weight tolerance
at one GRU coefficient, which also differed in the pre-DDP serial thread
comparison. A one-thread diagnostic matched the eight-rank weights and
moments within the unchanged tolerances and traced the outlier to the first
Adam update under different thread counts. The raw timing, process-tree
memory, and I/O records were retained. Compute-node validation remains pending.
[CPU DDP verification](docs/verification/cpu_ddp_training.md)

### Multicore Phase 2

The local cumulative suites passed 291 ordinary tests and 17 resource tests.
The genuine distance-5 fixture and a 17-shot installed-command smoke matched
prediction bytes for one, two and four workers, including G/F/S and MWPM.
Process interruption checks preserved out-of-order commits and resumed
with a different worker count. Completed and resumed predictions passed
whole-trajectory analysis, compact export/reanalysis, and PBS artifact checks.
The smoke retained startup-inclusive timing, sampled process-tree RSS,
I/O and attempt records. Compute-node checks remain pending.
[Parallel evaluation verification](docs/verification/parallel_evaluation.md)

### Reliability update

The cumulative ordinary suite passed 281 tests and the isolated resource
suite passed four. A Linux child read 96 shards under a 24-descriptor limit,
peaking at four payload mappings for full reads and three for labels-only
reads. A genuine distance-5 run resumed finite stored training and recovered
paired inference after process termination. The resumed outputs matched the
uninterrupted outputs and reproduced the paired statistics through export,
reanalysis, and PBS artifact verification. The shared-filesystem and
production-throughput checks remain pending.
[Reliability verification](docs/verification/reliability.md)

### P7

All 36 focused tests and all 240 independently run ordinary tests passed.
The distance-5 fixtures replayed G/F/S statistics in a fresh process from
detector-free bundles and separately reloaded a checkpoint for re-decoding.
They also reproduced archived S+/W/D contrasts. PBS rendering, job receipts,
and portable archive verification were checked locally. The allocated-node
probe, submission, interruption and resume, and remote transfer gates remain
pending. [P7 verification](docs/verification/P7.md) · [PBS cluster guide](docs/04_cluster_workflow.md)

### P6

The independent cumulative run passed all 204 ordinary tests. All three
distance-5 arms executed from the same terminal base checkpoint with matched
trajectory/core draws and update budgets. P5 evaluated the fixed deployment
windows and checked the continuation receipts. The D arm reproduced its full
three-update trajectory, buffer, model, optimizer, and scheduler state after a
fresh-process resume. The local checks include optional terminal validation
on a separate split. Production budgets and allocated-node checks remain
pending. [P6 verification](docs/verification/P6.md)

### P5

The independent cumulative run passed all 193 ordinary tests. A genuine
three-shot distance-5 fixture verified G/F/S, the original-DEM PyMatching
baseline, and paired counts and intervals from the persisted predictions.
The checks rejected reordered identities and mismatched checkpoints, and
verified nondegenerate limits for zero discordances. Main-analysis settings,
between-seed aggregation, and allocated-node checks remain pending.
[P5 verification](docs/verification/P5.md)

### P4

The independent cumulative run passed all 152 ordinary tests. The full
distance-5 decoder replayed four declared Adam updates exactly after a
two-update interruption, including consumed identities, model and optimizer
state, RNG state, and the next data position. The data checks verified paired
label round trips, retained zero-event shots, and rejection of corrupted
shards. The phase added a separate model-only terminal load for evaluation.
The compute-node environment and filesystem, main budgets, and decoder
performance remain pending. [P4 verification](docs/verification/P4.md)

### P3

All 15 focused tests, 126 independently run cumulative tests, and four
isolated CPU profile cases passed. The checks covered pinned upstream
components, full and sampled BCE gradients, one full distance-5 update,
checkpoint readback, and prediction isolation. The local profile measured
1, 2, 4, and 8 threads without changing the architecture. Main training and
shot budgets remain unresolved pending the allocated-node profile.
Exact commands, results, and the P4 handoff are in [P3.md](docs/verification/P3.md).

### P2

All 23 focused tests passed. The independent reviewer ran all 111 cumulative
tests and checked the pinned Lange graph conventions, all 108 pilot masks,
and invariance to excluded detector changes. The feature tests retain empty
graphs and separate supervision from model inputs. Exact commands, review
results, and the P3 handoff are in [P2.md](docs/verification/P2.md).

### P1

All 23 focused tests and all 88 cumulative tests passed with eight workers
in `nn_window_decoding`. The independent Pauli-frame oracle matched 2,645
elementary outcomes in the distance-3 fixture and the selected faults at the
distance-5 pilot cuts and endpoints. The pilot checks also verified the
measurement-dependent detector arrivals and retention of zero-event shots
in a separate $p=0$ fixture. The witness cases, exact commands, inventory
checksum, and C1 limits are recorded in [P1.md](docs/verification/P1.md).

### P0

The local C0 checks passed with 61 focused and 65 cumulative tests, with zero
failures or skips. The installed wheel resolved the pilot from outside the
checkout and recorded both children of a Hydra sweep. Invalid configurations
and run-name collisions were rejected without publishing new run records.
A fresh Conda environment recreated from the local inventories passed
`pip check`, dependency imports, and the wheel CLI check. The exact commands,
environment, and per-claim evidence are recorded in [P0.md](docs/verification/P0.md).

### Development environment

Created `nn_window_decoding` from the recorded local Conda builds and pip
versions and installed the project in editable mode with the `experiment` and
`dev` extras. The dependency imports and `pip check` passed with Python 3.11.18,
PyTorch 2.7.0+cpu, and Stim 1.15.0. The cumulative suite at source revision
`041d60d` passed all 65 tests in 9.67 s, with eight workers and one numerical
thread per worker. It used the thread exports below and the following command.

```bash
taskset -c 0-7 conda run --no-capture-output -n nn_window_decoding \
  python -m pytest -n 8 --dist=load --strict-markers --strict-config \
  -m 'not performance and not cluster' tests
```

### Original scaffold

The original scaffold was subject only to static checks of Python syntax and
docstring-only contents, TOML/YAML parsing, configuration-group file references,
local Markdown links, shell syntax of PBS examples, and archive integrity.
The companion `scaffold_checks.json` records that original snapshot.

### Shared test infrastructure

The test suite uses `pytest-xdist`, a thread bootstrap before worker collection,
and a packaged `runtime=test_cpu` profile. The shared tests run from the working
tree based on `0b78262`. The [assurance case](docs/verification/assurance_case.md)
tracks the claims and their prerequisites.

The local Conda environment `nn-window-ablation` was created from `environment.yml`
with Conda 24.9.2 on Linux x86-64 under WSL2. It used Python 3.11.18, pip 26.2.2,
pytest 8.4.2, pytest-xdist 3.8.0, and PyTorch 2.7.0+cpu from the official CPU wheel
index. The editable install with `experiment,dev` also resolved Hydra 1.3.2,
OmegaConf 2.3.0, NumPy 2.2.6, SciPy 1.17.1, Stim 1.15.0, PyMatching 2.2.2,
PyTorch Geometric 2.6.1, and Matplotlib 3.10.3. These imports and `pip check`
passed locally. A wheel built with `python -m pip wheel --no-deps` contained the
test CPU YAML with the same bytes as the source file.

The focused and cumulative checks ran sequentially with these commands.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
taskset -c 0-7 conda run --no-capture-output -n nn-window-ablation \
  python -m pytest -n 8 --dist=load --strict-markers --strict-config \
  -m 'not performance and not cluster' tests/unit/test_shared_cpu.py
taskset -c 0-7 conda run --no-capture-output -n nn-window-ablation \
  python -m pytest -n 8 --dist=load --strict-markers --strict-config \
  -m 'not performance and not cluster' tests
```

Both runs created eight workers within CPU affinity 0–7. The focused run had
4 passes in 6.84 seconds and the cumulative run had 4 passes in 4.66 seconds.
Both reported zero failures and skips. The test IDs are in
`tests/unit/test_shared_cpu.py`.

| Test | Expected result and observed check |
|---|---|
| `test_worker_starts_with_one_numerical_thread` | The native environment values and both active PyTorch thread counts are one. |
| `test_compatible_torch_setup_can_be_repeated` | Reapplying the bootstrap after an autograd operation preserves both PyTorch settings. |
| `test_incompatible_interop_setup_fails_in_fresh_process` | A synchronous child with inter-op threads already set to two fails explicitly when the bootstrap requests one. |
| `test_packaged_test_cpu_profile_composes_with_pilot` | Hydra composition matches the independently specified CPU settings and the pilot's distance, noise probability, model width, and GRU depth. |

These checks established the shared test setup before P0. The phase records
above provide subsequent acceptance evidence. Allocated compute-node
checks remain pending.

## Decisions retained for the pilot

Resolve the cluster queue/account policy, permitted walltime, actual CPU topology,
and compute-node visibility of repository/environment/data paths. Choose the
effective thread and batch settings from the $d=5$ profile. Determine $U_0$, $U_1$,
validation/test shot counts, and shard/chunk sizes before the main campaign.
The corresponding configuration fields remain null rather than implying a
measured budget.

## References

- PyTorch, previous versions and CPU installation. https://pytorch.org/get-started/previous-versions/
- Conda, run. https://docs.conda.io/projects/conda/en/stable/commands/run.html
- OpenPBS, qsub manual source. https://github.com/openpbs/openpbs/blob/master/doc/man1/qsub.1B
