# Development contracts

## Scope and status

This repository is a preimplementation scaffold. Implement the phase requested
by the user, using `docs/05_implementation_phases.md` as its scope and acceptance
contract. Update `IMPLEMENTATION_STATUS.md` with executed evidence and unresolved
issues. A proposed command, test, environment, or result must not be described as
implemented or verified before it exists and has been checked.

The user will prepare phase-specific implementation and test prompts. Keep
GitHub publication and cluster submission separate from local implementation;
perform them when requested. The `.pbs.example` files are templates.

## Scientific contracts

The theory and protocol documents define the physical circuit, clean temporal
cuts, label ownership, observation availability, common query interface, and
statistical estimands. Preserve these definitions when adapting upstream source.

- The scientific pilot is $d=5,T=60,c=b_L=b_R=5,p=0.003$, with random initialization.
- Retain the 256-dimensional representation and four GRU layers in the primary
  architecture. CPU tuning changes execution settings, not model capacity.
- The original global label equals the XOR of the 12 core labels on every shot.
- Teacher labels and auxiliary observable information never enter model features.
- Crop available raw observations before constructing graphs. Treat the last
  ancilla round and destructive readout as separate arrival phases.
- Reset recurrent state between independent windows; retain zero-event shots.
- G/F/S reuse exactly the same model and normalization state. S+/W/D start from
  that checkpoint and obey the matched data, optimizer-reset, and update policy.
- Pair complete trajectories, not individual windows, in test statistics.

## Implementation boundaries

Use one installed `src` package. Scientific modules accept explicit typed
parameters and arrays. Keep Hydra and filesystem orchestration in configuration,
CLI, and workflow modules. Do not add scheduler behavior to model or simulator
code. Store generated data and run artifacts outside tracked source.

Use the `nn_window_decoding` Conda environment for local development,
simulation, and tests. Run `conda activate nn_window_decoding` or use
`conda run -n nn_window_decoding`. Follow the documented CPU PyTorch
installation in `environments/README.md`. Validate the resolved environment
on an allocated compute node before claiming compatibility. No CUDA device
is required by the initial execution path.

Add tests for scientific invariants and behavior introduced by the requested
phase. Do not add pass-only tests or mark a static scaffold check as a scientific
test. Record upstream source commit, retained license notices, and task-specific
modifications when code is actually incorporated.

## References

- Lange et al. https://arxiv.org/abs/2610.03869
- Zhang et al. https://arxiv.org/html/2601.09921v1
- PyPA, src layout. https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/
- Hydra 1.3, Defaults List. https://hydra.cc/docs/1.3/advanced/defaults_list/
- Conda, managing environments. https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html
