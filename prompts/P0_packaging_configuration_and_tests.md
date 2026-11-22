# P0: Implement packaging, configuration, and the verification infrastructure

Implement P0 in the existing `nn-window-ablation` repository. Read `AGENTS.md`,
`IMPLEMENTATION_STATUS.md`, `docs/03_repository_design.md`,
`docs/04_cluster_workflow.md`, `docs/05_implementation_phases.md`,
`prompts/00_shared_requirements.md`, and `prompts/test_assurance_plan.md` first.
Read the actual `pyproject.toml`, `environment.yml`, and packaged YAML files.
The source files are currently responsibility placeholders; inspect them before
assuming any behavior exists.

## 1. Implementation scope

Create an installable `src` package and a working `nnwa` console entry point.
Implement the `validate` workflow and command dispatch for the five documented
verbs. Commands belonging to unimplemented phases must return a clear nonzero
status explaining the missing phase; they must not create fabricated successful
run artifacts. Do not implement circuit or model behavior in P0.

Implement Hydra 1.3 composition through the intended entry point, including
multirun. Keep configuration/orchestration outside scientific modules. Package
all default YAML resources so the installed command works from a directory
outside the source checkout. Maintain the existing configuration field names,
global experiment packaging, and defaults ordering [1, 2].

Implement typed resolution and validation. Compute

$$
T=d\,\mathtt{rounds\_per\_distance},\quad
c=d\,\mathtt{core\_per\_distance},\quad
b_L=\lceil d\,\mathtt{left\_per\_distance}\rceil,
$$

and the corresponding $b_R$. Save the input ratios and derived integers. Validate
odd integer distance, positive integral rounds, $c=d$, divisibility, supported
buffer ratios, probability range, supported model/runtime, and path fields.
An unresolved training or analysis budget may be inspected by `validate`, but a
workflow that needs it must reject it before starting computation. Null does not
mean an unlimited run or permission for an implicit scientific default.

Resolve relative paths against the invocation directory before I/O. Generate a
UUID-based run name before Hydra creates output directories if the caller omits
one. Explicit names must not silently reuse unrelated runs. Preserve a coherent
resume policy for P4; do not implement checkpoint loading yet. Multirun children
must receive distinct directories and their resolved overrides.

Use Conda/Python 3.11 and the documented CPU dependency candidate. Add a
compatible `pytest-xdist` version to development dependencies. Validate the
installed dependency imports when available, retain exact environment records,
and report unresolved installation failures honestly. Preserve the official CPU
wheel source; no GPU dependency or CUDA device is required [3, 4].

Add `conf/runtime/test_cpu.yaml` with CPU execution, one numerical thread per
test process, and an allocation cap of eight. Ensure CLI test invocations also
set `data.num_workers=0`, `data.sampling_processes=1`, and one application process.
Implement startup validation that prevents requested execution concurrency from
exceeding a declared allocation. Do not infer permission to use all host CPUs.
The ordinary cluster profile may remain separate from this test profile.

## 2. Test bootstrap and unit tests

Set up pytest markers, xdist-safe temporary artifacts, deterministic collection,
and the thread initialization required by the shared instructions. Do not add
pass-only placeholder tests. Add the following actual behavior checks, using
these file names or an equally clear phase-organized equivalent:

- `tests/unit/test_p0_config.py`: hand-calculated defaults give
  $(d,T,c,b_L,b_R,W,J)=(5,60,5,5,5,15,12)$; a half-distance buffer at $d=5$
  gives 3 and at $d=7$ gives 4. Check experiment/group/CLI override precedence,
  equal noise probabilities, and invalid distances, ratios, probabilities,
  partial cores, and unresolved required execution fields.
- `tests/unit/test_p0_runtime.py`: explicit thread/allocation fields are honored;
  an over-budget request is rejected before numerical work; repeated compatible
  startup is idempotent. A conflicting late inter-operation-thread request must yield a specific
  error rather than suppress an arbitrary PyTorch exception. Check that one
  worker does not start a nested numerical pool.
- `tests/unit/test_p0_paths.py`: path resolution has an independently written
  expected absolute result; unsafe/colliding run identities are rejected or
  resolved according to the documented rule. Two independently created run
  identities cannot share an output directory. No source/user path is hard-coded.

Use direct expected constants and temporary directory layouts as the oracle.
Do not calculate expected round lengths by calling the production resolver.

## 3. Installed-command integration checks

Add `tests/integration/test_p0_installed_cli.py`. Build/install the package in an
isolated test environment or another documented installation fixture and invoke
`nnwa validate` from a temporary working directory outside the checkout. Avoid
an inherited repository `PYTHONPATH` that would conceal missing package data.
Check the composed pilot settings, explicit environment-derived storage roots,
an explicit test CPU profile, and the command's exit code.

Execute a small two-condition `validate` multirun. Verify each child's resolved
values and output identity; do not emulate Hydra by manually merging YAML in the
test. Run this synchronously inside one xdist worker, with each child restricted
to one numerical thread. No nested pytest session is required.

Check invalid invocation and an unimplemented scientific workflow. They must
produce a specific error and no completed scientific run. Import the installed
package in a subprocess and assert that it does not sample data, launch jobs, or
create experiment outputs. The environment/installation fixture must not be
mutated concurrently by eight workers; build once before the parallel suite or
use isolated installations without changing the active environment during tests.

## 4. Evidence and execution

P0 establishes C0: later phases receive the intended typed physical/model/runtime
configuration through the installed interface. It does not establish any circuit
or decoder property. Include a negative control in which an invalid resource or
scientific setting is rejected, and ensure changing a preset actually changes the
resolved result. Record the installation mode and the package-data path exercised.

Apply the shared environment limits, then run focused P0 tests and the ordinary
cumulative suite using:

```bash
python -m pytest -n 8 --dist=load --strict-markers --strict-config \
  -m "not performance and not cluster" tests
```

Create `docs/verification/assurance_case.md` with C0 and its tests, and
`docs/verification/P0.md` with actual commands, environment, CPU/thread settings,
results, and any blocked checks. Update `IMPLEMENTATION_STATUS.md` without marking
P1–P7 implemented. Include generated test/build outputs in appropriate ignore
rules while retaining the concise verification records in source control.

Return the implemented interfaces, test evidence, and the commands needed to
start P1. Do not submit PBS jobs or publish the repository.

## References

1. Hydra 1.3, Defaults List. https://hydra.cc/docs/1.3/advanced/defaults_list/
2. Hydra 1.3, working directory.
   https://hydra.cc/docs/1.3/tutorials/basic/running_your_app/working_directory/
3. Conda, managing environments.
   https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html
4. PyTorch, previous versions and CPU wheels.
   https://pytorch.org/get-started/previous-versions/
5. pytest-xdist, process counts and distribution.
   https://pytest-xdist.readthedocs.io/en/stable/distribution.html
6. PyPA, src layout.
   https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/
7. PyTorch, inter-operation thread initialization.
   https://docs.pytorch.org/docs/2.7/generated/torch.set_num_interop_threads.html
