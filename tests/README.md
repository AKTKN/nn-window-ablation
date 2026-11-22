# Running and extending the test suite

## Local test execution

Use the Conda/Python 3.11 environment with the CPU PyTorch build and both the
`experiment` and `dev` extras. Follow the [environment setup](../environments/README.md).
From the active environment and the repository root, run one pytest session at a
time with the following startup settings.

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0
python -m pytest -n 8 --dist=load --strict-markers --strict-config \
  -m "not performance and not cluster" tests
```

If the process is allocated fewer than eight CPUs, use the smaller explicit
worker count and record it. Use the same settings for focused tests by changing
the test path or adding `-k`. Each worker initializes Torch intra-op and inter-op
parallelism to one before collecting tests. The session controller coordinates
workers. [pytest-xdist worker behavior](https://pytest-xdist.readthedocs.io/en/stable/how-to.html)

When P0 adds the CLI, ordinary subprocess tests should call it synchronously
with `runtime=test_cpu`, `runtime.processes=1`, `data.num_workers=0`, and
`data.sampling_processes=1`. For allocations smaller than eight, also set
`runtime.allocated_cpus` to the actual allocation. The parent worker waits for
its child without starting a second pytest pool.

Run tests marked `performance` after the ordinary suite with `-n 1` and
`-m "performance and not cluster"`. When comparing thread settings, start a
fresh synchronous child for each profile and set the native and Torch limits
explicitly within the allocation, up to eight numerical CPUs in total. Keep
`cluster` tests opt-in for an allocated node. [PyTorch inter-op startup constraint](https://docs.pytorch.org/docs/2.7/generated/torch.set_num_interop_threads.html)

## Test authoring

Put phase tests in `tests/unit/test_pK_*.py` or `tests/integration/test_pK_*.py`
and keep independent reference algorithms in `tests/reference/`. Use explicit
seeds or a stable test identity to derive streams. Sort parameters so every xdist
worker collects the same cases in the same order. Use `tmp_path` to isolate files.

Derive expected results independently of the production rule being checked.
Use exact comparisons for discrete values and explain floating-point tolerances.
Add a negative control for each major claim, such as a parity-preserving core
label corruption for P1. Use explicit faults for rare events and a separate
`p=0` fixture to force zero-event shots. The tests establish correctness of the
comparison procedure, while logical error-rate ordering and training gains are
experimental outcomes. Missing required local dependencies should fail the
suite. Mock scheduler transport or external commands only when their real
execution falls outside the test's scope.

The [phase contracts](../docs/05_implementation_phases.md) and
[assurance case](../docs/verification/assurance_case.md) define the evidence each
implementation must supply.

| Phase | Required evidence |
|---|---|
| P0 | Installed configuration discovery, valid derived lengths, invalid-setting rejection, and CPU dependency checks. |
| P1 | Noiseless/single-fault labels, clean-cut ownership, unchanged physical detector process, and shotwise global/core XOR identity. |
| P2 | Future-bit perturbation invariance, terminal arrival-phase policy, zero-event windows, and cropping before graph features. |
| P3 | Genuine distance-5 forward/backward step, shared query gradients, independent window state, and immutable G/F/S model state. |
| P4 | Packed-array round trip, split/shot identity, interruption-safe writes, and interrupted/resumed stream equivalence. |
| P5 | Whole-shot pairing, known discordance-count contrasts, final parity identity, and the specified intervals. |
| P6 | Common initial checkpoint, equal trajectories/updates/core-ID streams, optimizer reset, and unchanged deployment windows. |
| P7 | Actual small PBS allocation, complete/incomplete run identification, archive checksums, and local statistical reconstruction. |

Small $d=3$ fixtures may be used for inexpensive P1 algebraic checks. The scientific
pilot, CPU profile, and model integration check use $d=5$. Static parsing of the
scaffold is not evidence that these invariants hold in a simulator.

## References

- Stim observable annotation reference.
  https://github.com/quantumlib/Stim/blob/v1.15.0/doc/gates.md#the-observable_include-instruction
- Zhang et al. https://arxiv.org/html/2601.09921v1
- PyTorch, reproducibility. https://docs.pytorch.org/docs/2.7/notes/randomness.html
- pytest, getting started. https://docs.pytest.org/en/stable/getting-started.html
