# P4: Implement datasets, training, and exact resume

Implement P4 in `nn-window-ablation`: persistent datasets, deterministic training streams, the training loop, checkpoint/fork/resume behavior, and meaningful tests. Complete the code and execute the checks rather than returning an implementation plan. Do not run the main experiment, publish code, or submit cluster jobs.

## Required context and inherited claims

Read repository `AGENTS.md`, `IMPLEMENTATION_STATUS.md`, `docs/01_theory_and_decoder.md`, `docs/02_ablation_protocol.md`, `docs/03_repository_design.md`, P4 in `docs/05_implementation_phases.md`, and all relevant data/training/checkpoint configurations. Read this bundle's `prompts/00_shared_requirements.md` and `prompts/test_assurance_plan.md` before implementation.

P1 establishes physical labels, P2 establishes observable inputs, and P3 establishes the model/objective contract. Verify their executed evidence and retain their regression tests. P4 establishes that storage, sampling, optimization, and interruption preserve those contracts when composed. Passing serialization tests alone does not establish correct physical labels or useful decoding performance.

## Storage and identity contract

Implement packed `uint8` `.npy` shards with little-endian bit order within bytes, valid-bit counts, shapes, schema versions, circuit/feature/label provenance, and checksums. Support read-only memory mapping and minibatch unpacking. Store original global labels and the fixed 12 core labels separately from decoder inputs. [1]

Give each trajectory a stable dataset/shard/row identity. Split complete trajectories before creating queries/windows; every view of a trajectory remains in that split. Retain zero-event shots, including all-zero batches. Finalize immutable validation/test datasets before evaluation. Generated training chunks and reusable training shards must obey the same identity and label contracts.

Write shards and checkpoints to temporary files, finish their integrity records, and atomically publish completed artifacts. Publish a dataset manifest only after its listed shards are complete. Readers must reject incomplete manifests, corrupted payloads, inconsistent shapes/bit counts, or mismatched identities. An interrupted replacement must leave the previous completed checkpoint usable. State the filesystem assumptions of atomic publication.

## Random streams and training semantics

Use stable, versioned seed derivation, not Python's process-randomized `hash()`. Separate roles for initialization, circuit sampling, trajectory ordering, core-query sampling, dynamic-buffer sampling, and model stochastic operations. Include explicit dataset split, distance/replicate, training phase, and chunk/cursor identity where appropriate. Record the sampler version and sampling-call/chunk structure; a seed alone is not a portable dataset identifier. [2]

Base training and continuation use different trajectory/core-query namespaces. S+, W, and D share the same continuation trajectory sequence, minibatch order, and core-index sequence within a replicate; their arm names must not change those streams. Dynamic-buffer randomness is separate and cannot advance the physical-data or core-index streams. Its sampling unit is one logical $B$-trajectory minibatch: reuse that buffer pair across its accumulation microbatches. Implement these reusable stream/fork primitives now; the comparative adaptation campaign belongs to P6.

Each optimizer update consumes the declared effective batch of $B$ trajectories, with one global and one sampled-core query per trajectory. Use P3's normalized objective and explicit accumulation semantics. `training.max_steps` means updates in the current phase: a continuation with parent count $U_0$ and budget $U_1$ performs $U_1$ new updates. Record phase-local updates, cumulative updates, trajectories, and query examples separately. Unresolved budgets, optimizer choices, or required data settings must produce explicit execution errors.

Keep three operations distinct: evaluation loads model state; initialization/fork copies the parent model but creates fresh optimizer/scheduler and continuation streams; resume restores the same interrupted run. Resume must not apply the new-phase optimizer-reset flag. Save model parameters/buffers, optimizer moments, scheduler state, counters, all relevant RNG states, data chunk/row cursor, sampled-core state, and buffer-sampling state. Capture each NumPy generator's bit-generator state, Torch CPU RNG, and Python RNG where used; reseeding global generators is insufficient. Restore stochastic state after reconstruction operations that consume randomness and before the next training action.

## Unit tests with independent expectations

| Test file | Required evidence |
|---|---|
| `tests/unit/test_p4_shards_and_identity.py` | Compare packed arrays against hand-specified byte patterns, including non-byte-aligned bit counts. Verify immutable row identities, split disjointness, empty-event retention, and independent NumPy decoding. Inject truncation, byte corruption, wrong shapes, duplicate IDs, and swapped label artifacts; require specific rejection. |
| `tests/unit/test_p4_rng_namespaces.py` | Compare recorded sample/core-ID sequences across worker assignment and restart. Base and continuation namespaces must differ; continuation arm names must leave shared streams identical. Advancing the buffer stream must not change the next trajectory or core ID. |
| `tests/unit/test_p4_trainer_update_oracle.py` | Use a test-only scalar convex problem $L(\theta)=(\theta-a)^2/2$ and SGD without momentum. Derive expected updates independently: $\theta_{k+1}=(1-\eta)\theta_k+\eta a$. Check actual trainer updates, effective-batch accumulation, and counters against these values. Do not compute the expected result with the production optimizer/loss helper. |
| `tests/unit/test_p4_checkpoint_failures.py` | Inject write failures before publication and manifest finalization. Reject incomplete checkpoints and incompatible schemas/configurations while preserving the previous valid checkpoint. Distinguish fork from resume, including optimizer reset and phase-local versus cumulative counters. |

For label permutations, an unchanged manifest must detect modified bytes or reassigned artifact identities. Also use a deterministic P1 reference fixture with distinct known label rows to detect wrong label-to-trajectory associations even when structurally valid files are produced. Checksums alone do not prove semantic pairing. Include negative controls for replaying the base stream, arm-dependent sampling, double-normalized accumulation, and interpreting $U_1$ as an absolute cumulative stopping step. For example, a parent counter of 7 and a continuation budget of 4 must end at local counter 4 and cumulative counter 11.

## Mandatory composed integration tests

In `tests/integration/test_p4_paired_data_pipeline.py`, generate a small genuine $d=5,T=60$ dataset, persist it, reload it, construct P2 query batches, and run P3. Compare detector/label arrays and identities with the original in-memory records. Recheck the shotwise core-XOR identity and compare model outputs from the in-memory and reloaded paths. Inject a mislabeled or reordered artifact and demonstrate that the relevant integrity/identity/reference check rejects it. Preserve ordinary zero-event shots throughout.

In `tests/integration/test_p4_d5_train_resume.py`, use the full distance-5, 256-wide, four-layer model and a small batch. Compare four uninterrupted updates with two updates, checkpoint/reconstruction, and two resumed updates. Both paths declare four updates from the start; interrupt one after two without changing its declared budget. Use identical genuine trajectories, deterministic CPU settings, and one thread. Select bounded test-only Adam/scheduler settings that exercise optimizer moments and scheduler state; they are not campaign defaults. Compare consumed trajectory/core IDs, losses, parameters, buffers, optimizer/scheduler states, counters, and next-stream states. Within this fixed environment and identical operation sequence, require exact replay; do not weaken tolerances after observing failure. [3]

The pinned production GRU has no dropout by default; preserve it. Additionally use an explicitly test-only stochastic/dropout model to prove that restoring Torch RNG is necessary and effective. Removing RNG restoration must cause the replay checker to detect divergence. Similarly exercise incorrect optimizer moments and cursor restoration. If dynamic-buffer sampling is enabled in a test fixture, check its next choices after resume without changing the inherited objective or deploying P6 early.

The tiny convex/stochastic fixtures validate trainer mathematics and RNG handling; they cannot replace the full-model integration. Do not mark the full-model test as optional solely because it is slower.

## Eight-CPU execution and completion report

Use `runtime=test_cpu`, eight xdist workers, one PyTorch intra-operation and inter-operation thread per worker, zero data-loader workers, and no nested sampler pools. Apply the shared native-library limits before numerical imports. Run inherited and new ordinary tests with:

```bash
python -m pytest -n 8 --dist=load --strict-markers --strict-config \
  -m "not performance and not cluster" tests
```

Do not infer timing from concurrent correctness tests. Document the exact environment and deterministic settings: PyTorch does not promise cross-version or cross-platform bitwise replay. [3, 4]

Update `IMPLEMENTATION_STATUS.md`, `docs/verification/assurance_case.md`, and `docs/verification/P4.md`. Record C4's interfaces, assumptions, schema changes, commands, test counts, independent oracles, negative-control failures detected, full-model replay evidence, and remaining issues. Separate inherited C1–C3 evidence from the newly established C4 claim. Keep main-run budgets unresolved unless explicitly authorized measurements determine them; no decoder-performance result is implied by four fixture updates.

## References

1. NumPy bit packing and memory-mapped arrays: https://numpy.org/doc/stable/reference/generated/numpy.packbits.html and https://numpy.org/doc/stable/reference/generated/numpy.load.html
2. Stim sampling API and seed limitations: https://github.com/quantumlib/Stim/wiki/Stim-v1.13-Python-API-Reference
3. PyTorch reproducibility: https://docs.pytorch.org/docs/2.7/notes/randomness.html
4. pytest-xdist distribution modes: https://pytest-xdist.readthedocs.io/en/stable/distribution.html
