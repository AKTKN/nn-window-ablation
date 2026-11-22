# Implementation status

Last updated 2026-11-23. P0 through P3 are implemented and verified locally.

## Delivered structure

The repository provides the installed `nnwa validate` workflow, typed
configuration, packaged Hydra presets, and CPU runtime checks. The simulator
builds the original and annotated Stim circuits and samples aligned detector
bits and logical labels. The window pipeline selects permitted raw detector
bits and batches Lange-compatible graphs with label-independent metadata.
The shared query decoder supports finite CPU forward/backward steps and
model checkpoint payloads. Persistent training, statistical evaluation,
and campaign orchestration await their implementation phases.

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
| P4 | Datasets, training, checkpoints, deterministic resume | Not implemented. |
| P5 | Paired G/F/S evaluation and uncertainty | Not implemented. |
| P6 | Matched S+/W/D continuation | Not implemented. |
| P7 | PBS campaign, export, and local re-analysis | Not implemented; script templates only. |

The phase contract is [05_implementation_phases.md](docs/05_implementation_phases.md).
The scientific pilot is $d=5$; a small $d=3$ unit fixture is permitted in P1 but is
not a required training stage or the pilot.

## Validation record

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
