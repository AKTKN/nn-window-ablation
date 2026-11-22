# P7: PBS workflows, portable export, and independent re-analysis

Implement P7 in the existing `nn-window-ablation` repository. Read `AGENTS.md`, `IMPLEMENTATION_STATUS.md`, `docs/02_ablation_protocol.md`, `docs/03_repository_design.md`, `docs/04_cluster_workflow.md`, P7 of `docs/05_implementation_phases.md`, `prompts/00_shared_requirements.md`, and `prompts/test_assurance_plan.md`. Inspect actual runtime/YAML presets and existing `.pbs.example` files. Require P0–P6 for the complete local integration path; retain useful profiling launchers introduced earlier. The default deliverable is implemented and tested local orchestration and export. Use any separately established session authorization for bounded cluster acceptance; do not assume that cluster access is available or start a production campaign or GitHub publication.

## Implementation and reproducibility contract

Provide site-configurable PBS rendering/validation for generation, profiling, training/resume, and evaluation. Use explicit Conda executable and environment prefix, `conda run --no-capture-output`, a fixed source revision, explicit data/run roots, and portable run names. Record requested resources separately from observed allocation, scheduler identity, CPU/thread settings, and completion status. Keep scheduler behavior outside simulation/model modules. Do not guess queue/account names, scratch paths, or shared-filesystem visibility.

The documented cluster candidate is one node with up to 96 CPU resources and 350 GB, subject to actual allocation. Local automated tests have eight CPUs. Enforce separate profiles; a local fixture must never launch 96 workers or inherit the cluster's 16-thread default inside each pytest worker. Unresolved training budgets must fail launch validation with actionable field names. Rendering a script must not execute `qsub`, `ssh`, or `scp`.

Implement a finalized export manifest and portable compressed analysis bundle. Include selected run/checkpoint identities, inference and training provenance, dataset/label schema and shot identities, predictions, global/core labels, metrics, resolved analysis settings and RNG seeds, source/environment records, logs, and checksums. Include model checkpoints when requested for re-decoding or continuation. Raw training shards and raw detectors are optional for statistical re-analysis; the latter are required for re-decoding. Define these bundle capabilities explicitly.

Use manifest-relative content paths and a schema version. Preserve original source files. Publish a complete archive/manifest only after writing and verification succeed. Distinguish incomplete, failed, and completed runs so interrupted output cannot become an apparent completed comparison. Avoid relying on cluster usernames, absolute mount points, or unsafe archive member paths. Verify checksums after extraction and retain enough statistical metadata to reproduce the originally selected intervals.

Record local execution/export evidence under C7, conditional on C4–C6. Keep its local claims separate from the pending cluster-compatibility claim.

## Unit tests and independent references

- `tests/unit/test_p7_pbs_rendering.py`: render from explicit fixture configuration and compare the essential directives/argument tokens with independently written expectations. Cover paths containing spaces, missing Conda prefix, unresolved budgets, excess thread/worker allocation, missing optional queue/account, and portable run IDs. Validate shell syntax without executing the script. Place failing sentinel executables named `qsub`, `ssh`, and `scp` on the test PATH to prove rendering never calls them. Use `tmp_path`; never access a real scheduler.
- `tests/unit/test_p7_job_status.py`: verify state transitions from planned/running to completed or failed, and check that missing exit evidence or artifacts cannot imply completion. Model scheduler responses only at the transport boundary. An injected transport verifies argument construction and status parsing; it supplies no evidence of a real remote job.
- `tests/unit/test_p7_export_manifest.py`: construct small independently specified arrays and records, export them, and verify exact contents with standard JSON/NumPy readers rather than production loaders alone. Recompute file digests independently. Cover missing files, reordered shot IDs, mismatched label provenance, changed bytes, truncated arrays, unknown schema versions, and unfinished run records. Do not repair corrupt scientific content silently.
- `tests/unit/test_p7_archive_portability.py`: round-trip an archive into a differently named temporary directory and resolve all required inputs relative to its manifest. Test truncation, duplicate members, absolute paths, parent traversal, and links escaping the extraction root. Use a controlled supported extraction policy and verify that a rejected archive creates no files outside the destination. Compression must leave source artifacts intact.
- `tests/unit/test_p7_export_statistics.py`: use complete-shot truth tables with independently calculated failure/discordance counts. Reconstruct the signed differences and selected confidence limits from the exported arrays and stored analysis policy. Verify that modified predictions or an incompatible checkpoint condition are detected. An absent optional checkpoint must not block statistics-only analysis; an absent required label array must.

## Scientific integration and clean-room reconstruction

Implement `tests/integration/test_p7_d5_export_reanalysis.py` using actual $d=5,T=60,p=0.003$ circuits and joint labels, the real query-conditioned 256-wide/four-layer model, checkpoint saving/loading, and P5 evaluation. Use only a few shots and a small declared training budget. Export genuine G/F/S results; include tiny S+/W/D results where the existing P6 fixture permits. Check paired identities, core-XOR composition, condition provenance, and canonical model state after reload. Do not substitute fabricated labels, a constant decoder, or mocked graph/model layers for this path.

Extract the analysis bundle into a new working directory and invoke the installed analysis CLI in a fresh process. Hide the original circuit, dataset/training-shard roots, run directory, and fixture checkpoint from that process. A statistics-only bundle must reconstruct results without circuit sampling, model inference, or training. Verify required file access explicitly where practical, and make attempted regeneration fail. Compare exact shot/failure/discordance counts, signed point estimates, the selected CI method, and CI values under the stored algorithm/environment/seed. Compare numerical CSV/JSON content; timestamps and plot binary bytes need not match.

Separately, for a bundle declared capable of re-decoding, reload its actual checkpoint and required detector/structure records and verify inference on a tiny subset. Keep the statistical reconstruction and re-decoding acceptance claims distinct. Neither test requires low LER.

## Execution and acceptance record

Run local unit and scientific integration tests with the shared eight-CPU limits:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONHASHSEED=0 python -m pytest -n 8 --dist=load --strict-markers --strict-config -m "not performance and not cluster" tests
```

Use P0's `test_cpu` profile with PyTorch intra/inter-operation threads one, data-loader workers zero, and unique worker directories. Run isolated timing checks, if any, with `-n 1` and at most eight numerical threads; do not mix timings with concurrent correctness tests.

Keep real cluster acceptance in a separate opt-in `cluster` test/checklist: approved compute-node Conda imports, shared-path visibility, a submitted small CPU job from a fixed checkout, resource recording, short training interruption/resume, and checksum-verified transfer. Execute it when actual access and the existing session authorization cover the bounded check; do not request the same permission again. Without that access/authorization, complete all local work and record these gates as pending, with runnable commands for the user. Mocks cannot satisfy them, and pending remote checks do not invalidate demonstrated local composition.

Update `IMPLEMENTATION_STATUS.md`, `docs/verification/P7.md`, `docs/verification/assurance_case.md`, and the run/export schema. Report exact executed commands, passed and failed tests, repaired defects, evidence locations, negative controls, and the boundary between local verification and unexecuted remote acceptance. Do not claim a trained study result, production campaign, remote deployment, or cluster performance measurement from this phase's tiny fixtures.

## Primary references

- OpenPBS `qsub`: https://github.com/openpbs/openpbs/blob/master/doc/man1/qsub.1B
- Conda noninteractive execution: https://docs.conda.io/projects/conda/en/stable/commands/run.html
- Python archive handling: https://docs.python.org/3.11/library/tarfile.html
- PyTorch CPU oversubscription: https://docs.pytorch.org/docs/2.7/notes/multiprocessing.html#cpu-in-multiprocessing
- pytest-xdist process selection: https://pytest-xdist.readthedocs.io/en/stable/distribution.html
