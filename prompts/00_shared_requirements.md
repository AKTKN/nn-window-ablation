# Shared implementation and verification requirements

Apply these requirements to every phase prompt. The task is to implement the
requested phase in `nn-window-ablation`, together with its tests, execute the
applicable checks, and record the evidence. Inspect the existing implementation
before editing; retain valid work from earlier phases.

## 1. Source of the specification

Read `AGENTS.md`, `IMPLEMENTATION_STATUS.md`, the relevant documents under
`docs/01_theory_and_decoder.md` through `docs/05_implementation_phases.md`, and
`prompts/test_assurance_plan.md`. Use the actual packaged configuration schema.
The current user instruction adds mandatory phase-specific tests and an
8-CPU implementation/test budget. It supersedes any earlier suggestion to use
16 or 96 CPUs for this execution session. The 96-CPU PBS profile describes a
separate future cluster allocation.

Implement the phase rather than returning only a plan. Resolve ordinary
implementation choices consistently with the scientific specification. If an
earlier implementation violates a dependency required by this phase, make the
small necessary correction and add its regression test. Record material
scientific ambiguities instead of changing the target, observable convention,
architecture, or comparison to make a test pass. Do not implement unrelated
later phases, start a production campaign, or publish the repository as part of
these prompts.

## 2. Scientific invariants

The scientific pilot is rotated surface-code $Z$ memory with
$d=5$, $T=12d=60$, $c=b_L=b_R=5$, nominal bulk width $W=15$, and $p=0.003$.
Use the four equal circuit-noise parameters specified in the protocol. There are
12 disjoint core labels. Preparation belongs to the first core and destructive
readout to the last. Under the clean-cut convention,

$$
G_0=0,\qquad G_J=Y,\qquad Y_i=G_i\oplus G_{i-1},\qquad
Y=\bigoplus_{i=1}^{J}Y_i.
$$

The label identity alone does not establish physical fault ownership; P1 must
compare the labels with an independent frame oracle. Teacher labels and auxiliary
observable columns are unavailable to all decoders. Raw observations are selected
before graph features are constructed. Arrival phases distinguish the final
ancilla round $T^{-}$ from destructive readout $T^{+}$; nonfinal committed outputs
cannot use the latter. Empty observed slots differ from missing observations.

Retain the Lange backbone's 256-dimensional representation and four GRU layers,
the appropriate spatial aggregator, additive query metadata, and one shared
scalar head. Initialize the entire distance-5 study model randomly. Small
distance-3 circuits and test-only analytic models are permitted as unit fixtures;
they do not replace the distance-5 full-architecture integration checks.

G/F/S use the same offline checkpoint and normalization state. F/S predict the
same core labels and use the same XOR composition; their observation contexts
differ. Reset recurrent state for each independent query/window. S+/W/D fork the
same checkpoint, reset optimizer/scheduler state at the fork, retain the
full-context global objective, and share new continuation trajectories and core
query IDs. Continuation data use a namespace distinct from base training and
common to all three arms. Draw one dynamic buffer pair per logical minibatch of
$B$ trajectories and reuse it across gradient-accumulation microbatches. The
continuation step budget counts new updates.

Do not encode an expected LER ordering or a desired scientific conclusion in
tests. A model's stochastic improvement, a published accuracy level, and training
convergence are measured outcomes, not unit-test assertions.

## 3. Eight-CPU test execution

P0 must add `pytest-xdist` to the development dependencies and implement a shared
test bootstrap. Resolve a compatible version with the selected Python/pytest
environment and record the installed version. Do not substitute an ordinary
serial pytest run for the requested parallel validation without explaining an
actual blocker.

Use this environment and command for the ordinary cumulative suite, from the
active Conda environment and repository root:

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

Use `-n 8` explicitly, not `-n auto`: the host can expose more CPUs than this
task's allocation. If the actual process allocation is smaller than eight, use
that smaller count and report it. Run only one pytest session at a time. The
controller performs coordination; at most eight workers execute numerical work.
The xdist distribution mode does not guarantee test order [1].

Initialize PyTorch intra-operation and inter-operation parallelism to one per
worker before tensor/autograd work. The inter-operation setting is process-wide
and can only be set once before parallel work; make startup initialization
idempotent when the requested setting is already active, rather than repeatedly
calling the setter or suppressing arbitrary runtime errors [3, 4]. Test imports
must not execute training at collection time. Set native-library thread limits
before importing numerical libraries.

CLI subprocess tests must explicitly select the test CPU profile. Use zero
data-loader workers and no concurrent sampler/process pools in the ordinary
suite. A synchronous child process may replace its idle test parent's numerical
work, but a test must not start another eight-worker pytest pool. Mock scheduler
transport and external commands when their real execution is outside the test.

Tests that measure multithreaded runtime or intentionally exercise a multiworker
loader belong to the `performance` marker. Run them separately, after the
ordinary suite, with xdist `-n 1` and a total numerical CPU budget no larger than
eight. Set the native/Torch thread limits for that isolated experiment explicitly.
Such timing measurements must not run beside eight busy xdist workers. Use a
fresh numerical subprocess for each thread-count profile when startup settings
differ, with the single xdist worker waiting synchronously. Performance
measurements are not hardware-independent correctness gates [3].

## 4. Isolation, deterministic inputs, and independent oracles

Use `tmp_path` or equivalent per-test directories for mutable artifacts. Worker
IDs can isolate paths, but must not determine scientific seeds or expected
results. Derive each test's random stream from an explicit constant and stable
test identity, or parameterize fixed seeds. Do not use Python's randomized
`hash()` for persistent seeds. Sort parameter lists; every worker must collect
the same tests in the same order [2, 5].

Unit tests must establish a scientific or software contract, not merely execute a
line or assert a mocked return value. Expected outputs come from hand-calculated
fixtures, a separate reference algorithm, exhaustive finite cases, mathematical
identities, or explicitly trusted upstream primitives. Test-reference modules
must not call the production function being checked or its critical ownership,
availability, or label helper. Sharing primitive array libraries is acceptable;
sharing the disputed rule is not.

Use exact comparisons for bits, IDs, indices, integer counts, and serialized
payload hashes. State numerical tolerances for floating-point comparisons and
their reason. Do not widen a tolerance to hide an unexplained discrepancy. Use
controlled finite examples instead of rare-event significance tests as mandatory
gates. Inject faults or choose fixtures explicitly when an event must be present.
Use a separately identified $p=0$ fixture to force zero-event trajectories; do
not require a rare zero-event shot in a bounded $p=0.003$ sample or relabel its
generating noise distribution.

Add a targeted negative control for each major claim: a deliberately corrupted
artifact, prohibited future observation, mismatched identity, or test-local
incorrect mapping must be detected by its comparator/validator. Do not leave
deliberate defects in production code. Mocks are appropriate for filesystem or
scheduler failures; they are not evidence for circuit, label, or NN correctness.

## 5. Cumulative evidence and completion

Implement phase-specific tests under `tests/unit/test_pK_*.py` and
`tests/integration/test_pK_*.py`; place small independent helpers in
`tests/reference/`. These names are an organizational convention, not a required
public API. Register `performance` and `cluster` markers. Required local tests
must fail clearly when core dependencies are absent, rather than disappear under
blanket `importorskip` or unexplained `xfail`.

During development, run the focused phase tests with the same eight-worker and
thread policy. At phase completion, run the cumulative ordinary suite once. Fix
failures and rerun the affected checks; avoid repeated large sweeps with no
remaining question. Previous-phase fixtures must be reproducible locally, rather
than depend on the order in which pytest happened to execute tests.

Create or update `docs/verification/assurance_case.md` and
`docs/verification/P0.md` through `P7.md` as phases are implemented. For each claim
record its assumptions, independent oracle, test IDs, negative control, observed
result, and remaining domain of validity. A later phase may rely only on a claim
whose prerequisites are recorded. Update `IMPLEMENTATION_STATUS.md` accurately.

The final phase response must report implemented behavior, changed files, the
exact test commands, pass/fail/skip counts, duration, CPU/thread settings, source
and environment identity, and the claim supported by the results. Distinguish
unexecuted cluster checks from passed local tests. Provide small reproduction
commands and the next phase's usable interfaces. No simulated results or test
success may be invented.

## References

1. pytest-xdist, running tests across multiple CPUs.
   https://pytest-xdist.readthedocs.io/en/stable/distribution.html
2. pytest-xdist, known limitations.
   https://pytest-xdist.readthedocs.io/en/stable/known-limitations.html
3. PyTorch, multiprocessing and CPU oversubscription.
   https://docs.pytorch.org/docs/2.7/notes/multiprocessing.html
4. PyTorch, inter-operation thread initialization.
   https://docs.pytorch.org/docs/2.7/generated/torch.set_num_interop_threads.html
5. pytest, temporary directories.
   https://docs.pytest.org/en/stable/how-to/tmp_path.html
6. PyTorch, reproducibility.
   https://docs.pytorch.org/docs/2.7/notes/randomness.html
7. Lange et al. https://arxiv.org/abs/2610.03869
8. Zhang et al. https://arxiv.org/html/2601.09921v1
