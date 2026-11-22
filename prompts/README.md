# Codex implementation prompt pack

This pack contains English prompts for implementing P0–P7 of
`nn-window-ablation`, including phase-specific unit tests, integration tests,
and a compositional verification argument. These are instructions for future
implementation; no simulation code or test results are supplied by this pack.

## Use

Extract the archive into the repository root so this directory is `prompts/`.
The existing repository must contain the previously supplied scientific/design
documents under `docs/`. Give Codex the full contents of one phase file, starting
with P0. Each prompt directs it to read the shared requirements and assurance
plan from the repository. Complete the phase's implementation and verification
record before proceeding to the next phase.

| File | Task |
|---|---|
| [00_shared_requirements.md](00_shared_requirements.md) | Scientific invariants, eight-CPU xdist policy, oracle independence, and evidence records. |
| [test_assurance_plan.md](test_assurance_plan.md) | C0–C7 claims and the reasoning connecting phase tests to simulation correctness. |
| [P0_packaging_configuration_and_tests.md](P0_packaging_configuration_and_tests.md) | Installed package, Hydra, configuration, CPU limits, and test infrastructure. |
| [P1_circuits_and_labels.md](P1_circuits_and_labels.md) | Physical circuits, exact labels, independent frame oracle. |
| [P2_windows_and_features.md](P2_windows_and_features.md) | Availability, window selection, and graph features. |
| [P3_model_and_query_interface.md](P3_model_and_query_interface.md) | Lange adapter, query/loss interface, and CPU execution. |
| [P4_data_training_and_resume.md](P4_data_training_and_resume.md) | Storage, training streams, optimizer steps, and resume. |
| [P5_paired_evaluation_and_statistics.md](P5_paired_evaluation_and_statistics.md) | G/F/S, paired counts, confidence intervals, and end-to-end checks. |
| [P6_matched_context_training.md](P6_matched_context_training.md) | Matched S+/W/D continuation and controlled context schedules. |
| [P7_pbs_export_and_reanalysis.md](P7_pbs_export_and_reanalysis.md) | PBS orchestration, portable export, and local reconstruction. |

Ordinary executable tests use pytest with pytest-xdist (`python -m pytest -n 8`)
and one numerical thread per worker. Timing or deliberately multithreaded tests run separately with
`-n 1` and at most eight numerical CPUs in total. This is independent of the
future 96-CPU cluster profile. The exact shell environment and exception rules
are in the shared file [1, 2].

The phase evidence accumulates in `docs/verification/assurance_case.md` and
per-phase records. It must distinguish algebraic arguments, finite exhaustive
checks, deterministic integration checks, empirical experiments, and pending
cluster validation. The test suite must not assume that streaming decoding is
worse or that context-aware training improves LER; those are study outcomes.

## References

1. pytest-xdist, running tests across multiple CPUs.
   https://pytest-xdist.readthedocs.io/en/stable/distribution.html
2. PyTorch, CPU multiprocessing.
   https://docs.pytorch.org/docs/2.7/notes/multiprocessing.html
3. Lange et al. https://arxiv.org/abs/2610.03869
4. Zhang et al. https://arxiv.org/html/2601.09921v1
