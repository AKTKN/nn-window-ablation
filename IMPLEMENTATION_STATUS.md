# Implementation status

Snapshot date: 2026-11-22. State: preimplementation scaffold.

## Delivered structure

The repository contains English scientific/design documents, docstring-only
Python module placeholders, Hydra YAML examples, package/dependency declarations,
and PBS examples. `nnwa` has no registered console entry point. No upstream model
source has been vendored. No Git repository was initialized or published by this
delivery.

## Phase status

| Phase | Scope | Status |
|---|---|---|
| P0 | Packaging, CLI, typed configuration, validated CPU environment | Not implemented; metadata/configuration examples only. |
| P1 | Physical circuit, clean cuts, exact global/core labels | Not implemented. |
| P2 | Observation windows and features without future leakage | Not implemented. |
| P3 | Lange model adapter, shared queries, CPU profile | Not implemented. |
| P4 | Datasets, training, checkpoints, deterministic resume | Not implemented. |
| P5 | Paired G/F/S evaluation and uncertainty | Not implemented. |
| P6 | Matched S+/W/D continuation | Not implemented. |
| P7 | PBS campaign, export, and local re-analysis | Not implemented; script templates only. |

The phase contract is [05_implementation_phases.md](docs/05_implementation_phases.md).
The scientific pilot is $d=5$; a small $d=3$ unit fixture is permitted in P1 but is
not a required training stage or the pilot.

## Validation record

The delivered scaffold is subject only to static checks: Python syntax and
docstring-only contents, TOML/YAML parsing, configuration-group file references,
local Markdown links, shell syntax of PBS examples, and archive integrity.
The companion `scaffold_checks.json` records the checks actually completed.

These checks do not execute Hydra composition, install or resolve the candidate
scientific environment, run pytest, simulate a shot, execute a model, or submit a
PBS job. Scientific acceptance criteria remain untested until their phase is
implemented.

## Decisions retained for P0 and the pilot

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
