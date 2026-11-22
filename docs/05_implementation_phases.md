# Implementation phases and acceptance criteria

Status: implementation plan. The phases below are task contracts for later development and review. They do not report completed simulator code, successful tests, trained checkpoints, cluster jobs, or experimental results. They are not ready-to-send implementation prompts.

## 1. Fixed decisions and execution workflow

The repository is `nn-window-ablation`, its import package is `nn_window_ablation`, and its proposed command entry point is `nnwa`. Development and short correctness checks take place locally. Versioned source is subsequently transferred through the user's GitHub workflow, cloned on the cluster, and executed under PBS using `qsub`. Dataset generation and substantive training take place on the cluster. Selected run artifacts are compressed there and downloaded with `scp` for local inspection and analysis. Creating or publishing a remote repository is not part of this scaffold delivery.

The target cluster allocation is one CPU node with up to 96 PBS CPU resources and 350 GB of memory. Conda is the environment manager. CPU execution is the supported initial path; the package must not require a CUDA device or silently substitute a smaller model when no GPU is available. The site's actual PBS resource syntax and walltime limits belong in a cluster profile, separate from scientific configuration.

The scientific pilot starts directly at

$$
d=5,\qquad T=60,\qquad c=5,\qquad b_L=b_R=5,\qquad W=15,\qquad p=0.003.
$$

Use the query-conditioned Lange architecture with its 256-dimensional representation, four GRU layers, and the distance-5 spatial aggregation structure, including its $2\times2$ tile aggregation. Initialize all study-model parameters randomly for the first distance-5 pilot. Preserve the architecture across the ablation arms. The public Lange source is the architecture and implementation reference [R1, R2]; adopting it does not establish a performance result for the added query task.

No distance-3 training run is a prerequisite. Small $d=3$ fixtures may be used in P1 for inexpensive invariant checks, but the scientific pilot and CPU profiling are at $d=5$. A later $d=3$ warm-start policy can be studied separately. Before the main campaign, choose and record one initialization policy; use that same policy for every compared arm. Do not mix random initialization and distance-transfer initialization inside a context-training comparison.

The scientific definitions in the theory and ablation-protocol documents remain authoritative: physical circuit, clean cuts, global and core labels, observation deadlines, one shared query-conditioned head, G/F/S comparisons, and matched S+/W/D continuation budgets. The phases implement those definitions without changing them to accommodate a convenient package interface.

## 2. Phase order and review records

| Phase | Main output | Depends on |
|---|---|---|
| P0 | Installable CPU package, validated configuration, Conda environment specification | Design documents |
| P1 | Physical circuit, exact auxiliary labels, instruction-cut metadata | P0 |
| P2 | Windows, availability masks, feature construction with no future leakage | P1 |
| P3 | Lange adapter, common global/core queries, CPU profile | P2 |
| P4 | Reproducible data pipeline, training, checkpoint and resume | P3 |
| P5 | Frozen G/F/S comparison on paired complete shots | P4 |
| P6 | Matched S+/W/D continuation and buffer evaluation | P5 |
| P7 | Reproducible PBS campaign, compressed artifact export, local re-analysis | P4–P6 |

A minimal PBS launcher from P7 should be brought forward when P3 profiling or P4 training first runs on the cluster. P7 then consolidates the tested job and export workflow after the scientific paths work. The ordering therefore does not require long local training before cluster integration.

At the end of a phase, retain a short review record containing the source commit, exact environment, executed checks, observed results, and unresolved issues. A proposed acceptance criterion is not marked passed until its execution evidence exists. Review the phase's scientific invariants before adding the next layer of functionality.

## P0. Packaging, configuration, and CPU environment

### Scope

Create one installable package with a `src` layout, a small CLI, typed configuration records, packaged experiment defaults, and explicit paths for data and run artifacts. Scientific modules receive resolved parameters and arrays; they do not depend on the current working directory or Hydra's global state.

Provide a Conda environment specification for CPU execution, including compatible versions of Python, CPU PyTorch, PyTorch Geometric, Stim with Pauli-target observable support, PyMatching, NumPy, SciPy, and the selected configuration/test packages. Pin the resolved environment after compatibility checks. Keep the portable environment intent and the actual Linux environment export distinguishable [R3].

The default scientific preset is `pilot_d5`, with the parameters in Section 1. CPU thread counts, data-loader workers, sampler processes, paths, and PBS resource fields are execution settings. They must not change $d$, the network width, its GRU depth, the physical noise model, or query semantics.

### Acceptance criteria

- A fresh local CPU environment can install the package and import its declared dependencies.
- The installed CLI resolves the default configuration from a directory outside the source checkout.
- The resolved configuration records $d=5$, $T=60$, $c=b_L=b_R=5$, the four equal Stim noise parameters, `device=cpu`, and random initialization.
- Invalid combinations, such as an undefined core partition or an unavailable model backend, produce a specific configuration error before creating a run.
- Run metadata records package versions, source revision, CPU identity, allocated CPU count, and thread/worker settings.
- The CLI resolves a unique run name before Hydra output creation; independent jobs cannot silently share an unrelated output directory.
- Importing the package does not launch training, sample data, submit jobs, or create large artifacts.

The first environment check validates dependencies and configuration only. It does not claim that the scientific simulator exists yet.

## P1. Physical circuits, exact labels, and clean cuts

### Scope

Implement the rotated surface-code $Z$-memory circuit with all four specified Stim noise probabilities equal to $p$. Keep the original physical circuit and its global logical observable separate from the teacher-annotated circuit. Record exact instruction locations for completed-round cuts, physical detector availability, preparation, and destructive readout.

At each internal core cut, place the auxiliary Pauli annotation after the complete ancilla measurement/reset and before the reset or idle noise assigned to the next interval. A premeasurement fault belongs to the preceding interval; a post-cut fault belongs to the next interval. The final core owns the destructive-readout tail. Stim introduced Pauli-target `OBSERVABLE_INCLUDE` support in version 1.15 [R4, R5].

Sample physical detectors and all labels jointly. For cumulative signs $G_0=0$, $G_i$ at internal cuts, and $G_J=Y$ from the actual final observable, construct

$$
Y_i=G_i\oplus G_{i-1},\qquad
Y=\bigoplus_{i=1}^{J}Y_i.
$$

The selected physical $\overline Z$ representative must match the original final measurement support and be preserved by a complete noiseless extraction round. This is an instruction-level property, not an inference from detector coordinates alone.

### Acceptance criteria and integration checks

- Removing the auxiliary observable annotations restores the exact original physical operations and detector definitions.
- Noiseless samples have the expected zero global and cumulative flip labels.
- Every sampled shot satisfies the core-XOR identity, including preparation and final-readout faults.
- Deterministic injected noise faults, such as probability-one Pauli error channels, agree with an independent frame-propagation calculation at selected cuts. An intended deterministic Pauli gate is not substituted for a noise fault when testing flip labels.
- Selected faults immediately before and after a cut have the prescribed ownership, including faults whose detector effects span two regions.
- A small annotated-circuit batch agrees with an independently inspected Pauli-frame reference on the relevant labels.
- The classical baseline's circuit/DEM contains only the original global observable; auxiliary teacher columns cannot enter its syndrome input.

Small $d=3$ circuits are permitted only as convenient unit fixtures in this phase. Include a $d=5,T=60$ integration fixture before declaring the physical-data contract complete.

## P2. Windows, observation availability, and graph features

### Scope

Implement the fixed $c=d$ partition and buffered observation windows. Availability is defined from the constituent measurement records and their phase. Completion of the final ancilla round at $T^{-}$ is distinct from destructive readout at $T^{+}$.

At the default $b_R=d$, a nonfinal core whose deadline equals $T$ commits at $T^{-}$. It receives no terminal-readout detectors. The final core waits until $T^{+}$. Window completion never revises previously committed predictions.

Restrict raw detector observations before constructing the Lange encoder's two-round graphs or any other temporally supported features. Preserve empty observed layers as explicit temporal slots. Represent unavailable context and physical padding separately from an observed zero-event layer. Construct query metadata independently of target values.

### Acceptance criteria and integration checks

- For $d=5,T=60$, the partition has 12 nonoverlapping target cores and bulk observation windows of 15 rounds.
- Each selected detector is available by its window's declared deadline.
- The penultimate default window excludes destructive-readout detectors, while the final window includes the allowed terminal layer.
- Perturbing observations outside a window's availability set leaves that window's feature representation unchanged.
- Perturbing final readout leaves every already released nonterminal window unchanged.
- Two-round graph construction cannot import an excluded layer through a precomputed embedding.
- Empty observed slots, unavailable slots, and boundary padding retain their intended identities through graph batching.
- Packing, cropping, batching, and unbatching preserve shot IDs, core IDs, temporal order, and detector indices.

This phase establishes the observation contract. A good classification score cannot compensate for a failure of these leakage checks.

## P3. Lange adapter, common queries, and CPU profiling

### Scope

Adapt the pinned upstream GNN/spatial-aggregation/GRU implementation while retaining its license and provenance [R1, R2]. Add the agreed query and boundary metadata before one shared scalar readout. Global and core queries use the same parameter set, rather than independently trained output heads.

Every independent window starts from zero recurrent state. Full-context core queries and window core queries predict exactly the same $Y_i$ labels. The global query predicts $Y$. The direct distance-5 pilot initializes the backbone, spatial aggregator, metadata projection, and readout randomly; all are trainable.

Implement the shared objective with $\lambda=1$. During training, each trajectory supplies a global query and one uniformly sampled core query. The sampled core term estimates the average over the fixed 12 cores. At evaluation, all required core queries are executed.

### Acceptance criteria and integration checks

- One CPU forward/backward step executes on a genuine $d=5,T=60$ sampled batch with finite logits, loss, and gradients.
- Both global and core losses reach the shared parameters, including the query projection; query metadata carries no target bits.
- One saved state dictionary supports G, F, and S without parameter mutation, head replacement, or normalization-statistic updates.
- Restoring full context reproduces the same full-context core-query path. The test does not incorrectly require the XOR of core predictions to equal the independently queried global prediction.
- Evaluating a window after another window does not retain its recurrent state.
- Source provenance records the upstream commit and the task-specific modifications.

### Required CPU profile

Profile $d=5,T=60,W=15,p=0.003$ with the specified 256-wide, four-layer architecture. Measure circuit sampling and labels, graph construction, model forward pass, backward/update, and serialization separately. Distinguish training's two queries per trajectory from evaluation's complete set of core queries.

Record complete trajectories per second, query examples per second, update time, peak resident memory, and the actual process/thread configuration. Begin with one training process, a conservative thread allocation, and no data-loader workers; increase concurrency in a small controlled sequence only while it improves measured throughput. Data-loader workers, sampler workers, and BLAS/PyTorch threads share the same allocated CPU-resource budget (up to 96 in the supplied node record). PyTorch documents the performance loss caused by process/thread oversubscription [R6].

Use the profile to choose effective batch size, worker counts, $U_0$ full-context updates, $U_1$ continuation updates, and validation/test sizes. These budgets remain undecided until measured on the target CPU node. The profile must not silently shrink the model or assume that 96 training processes are efficient.

## P4. Reproducible datasets, training, and resume

### Scope

Implement compact detector/label shards with manifests, immutable shot identifiers, and explicit train/validation/test splits. Keep all windows and queries from a trajectory in one split. Retain zero-event shots. Generate and store the paired test set independently of model training.

Implement fixed-update training and resumable checkpoints. Store the model, optimizer, scheduler, normalization buffers, random-generator states, data-stream position, sampled-core stream, resolved configuration, environment, and source revision. The checkpoint at exactly $U_0$ defines the offline model used in G/F/S; later adaptation checkpoints use the prescribed $U_1$ continuation budget.

### Acceptance criteria and integration checks

- Shards round-trip without changes to detectors, labels, bit order, or row identities.
- Training, validation, and test trajectory IDs are disjoint; no window-level split can leak a trajectory between sets.
- A resumed short run consumes the same next trajectories and sampled core indices as an uninterrupted reference run.
- In a fixed deterministic CPU test environment, resumed and uninterrupted optimizer updates match under the declared numerical criterion.
- An incomplete write is not accepted as a completed checkpoint or shard; existing valid artifacts remain readable after interruption.
- Update and example counters are unambiguous: $U$ updates at effective batch size $B$ consume $UB$ trajectories and $2UB$ query examples under the chosen policy.
- The primary checkpoint is selected by the fixed update count, not by main-test performance.

The reproducibility claim is limited to the recorded environment and deterministic policy. PyTorch does not promise identical results across releases or platforms [R7]. Store test arrays and their hashes so paired evaluation does not depend on recreating a seeded sampling sequence on a different environment.

## P5. Frozen G/F/S benchmark

### Scope

Implement evaluation of one $\theta_F$ checkpoint on the same complete test trajectories:

- G: one global query using complete closed history.
- F: all core queries using complete closed history, followed by XOR.
- S: the same core queries using their allowed windows, followed by XOR.

Report whole-shot memory LER, paired differences, and uncertainty. Retain per-core disagreements and their parity to diagnose composition and endpoint effects. An optional full-history PyMatching result uses the original physical DEM and the same detector rows.

### Acceptance criteria and integration checks

- G/F/S load the same checkpoint hash and retain unchanged model state during evaluation.
- Prediction records preserve checkpoint training regime and inference mode separately; the corresponding window conditions are identifiable as S, S+, W, or D.
- All three outputs align to identical shot IDs and the same global labels. Missing, duplicated, or reordered identities are detected before statistics are combined.
- For F and S, final failure equals the XOR of core disagreement bits on every shot.
- Paired counts $n_{10}$ and $n_{01}$ reproduce the signed LER difference on small manually specified examples, including zero-discordance cases.
- Confidence intervals follow the declared whole-shot method; windows are never treated as independent bootstrap observations.
- The report separates $P_F-P_G$ from $P_S-P_F$, rather than attributing the entire $P_S-P_G$ difference to missing context.
- Failure counts and interval widths accompany small or zero estimates; the workflow does not manufacture a finite relative ratio when its denominator is unresolved.

Run the first scientific pilot at $d=5,p=0.003$. It establishes feasibility and estimates failure/discordance rates for budgeting. It does not establish convergence, scaling with distance, or the effect size expected in the main campaign. Add independent training seeds and $d=7$ only after the pilot gives a concrete resource estimate.

## P6. Matched S+/W/D adaptation

### Scope

Fork S+, W, and D from the same offline checkpoint within each training replicate. Reset optimizer and scheduler state in all three arms, including S+, using the same settings. Reuse the same new trajectory stream, minibatch order, effective batch size, sampled core IDs, and number of optimizer updates.

S+ continues full-context training. W uses the fixed deployment windows for its core-query term. D samples buffer lengths from $\{0,\lceil d/2\rceil,d\}$ while keeping core length $c=d$ and the target partition fixed. All arms retain the same full-context global term and the same $\lambda=1$. Evaluate each at identical declared deployment buffers. These are study adaptations inspired by dynamic-context training [R8], not an unchanged ASR implementation.

### Acceptance criteria and integration checks

- The initial model hash is identical across S+/W/D within a replicate, and no arm reinitializes a layer.
- `training.max_steps` counts $U_1$ new updates; parent and phase-local counters remain separate, and resume restores phase-local progress.
- All arms use the same continuation trajectory/core-query namespace, distinct from base training; chunk cursors cannot replay the original full-context training stream.
- Each arm records exactly $U_1B$ additional trajectories and $2U_1B$ query examples for $U_1$ updates.
- Trajectory order and sampled core IDs match across arms; dynamic buffer randomness has its own recorded stream.
- Buffer selection changes input availability but never target labels, core ownership, or the final-readout timing rule.
- Fixed-window evaluation uses the same windows for S+, W, and D. Dynamic training receives no extra inference context.
- Additional training uses the same validation schedule and fixed terminal-update checkpoint policy; the test set is not used for arm selection.
- CPU hours, processed detector tokens, and memory accompany the matched data/update budgets. Equal updates are not labeled equal floating-point work.

Use the paired contrast $P_{S+}-P_W$ to assess context-matched training beyond additional optimization, and $P_W-P_D$ to assess dynamic context training. Whole-shot pairing and independent training-seed variability remain separate parts of the uncertainty analysis.

## P7. PBS campaigns, artifact export, and local reconstruction

### Scope

Provide site-configurable PBS launchers for dataset generation, CPU profiling, training/resume, and evaluation. They activate the recorded Conda environment, identify a fixed source revision, use explicit data/run roots, and log scheduler job identity, actual allocated resources, and effective thread/worker settings. Long computation belongs in submitted jobs, not in an interactive login-node process.

Bring forward the minimal profiling launcher when P3 is ready. After P4–P6, finalize the repeatable campaign configuration and export workflow. Keep source transfer through Git separate from large generated datasets and checkpoints.

An export manifest identifies each included run, checkpoint, test dataset, prediction array, statistics table, resolved configuration, environment record, source revision, and file checksum. Provide a compact analysis bundle containing everything needed to reconstruct the reported comparisons; allow raw training shards to be a separate, optional archive. Compress completed artifacts on the cluster and download the archive with the user's `scp` workflow. Compression must not remove the source artifacts.

### Acceptance criteria and integration checks

- A small submitted CPU job executes from a cloned checkout without relying on local absolute paths or an interactive shell's already-active environment.
- Requested and effective resources agree with the selected cluster profile; concurrent worker/thread settings remain within the allocation.
- A short training job can be interrupted and resumed through the same checkpoint mechanism used locally.
- A failed or incomplete job is distinguishable from a completed run in its manifest and export selection.
- Archive contents match their checksums before transfer and after extraction into a clean local directory.
- Local analysis reconstructs the reported paired counts, LER estimates, and confidence intervals from the exported predictions and labels without regenerating training data.
- Data paths in a restored bundle resolve relative to its manifest or explicit configuration, rather than to the original cluster username or mount path.

This phase verifies operational reproducibility. It does not authorize automatic publication, start a production campaign, or turn a small job check into an experimental performance claim.

## References

- [R1] M. Lange et al., *Modular Neural Decoder for Surface Code Memory and Logic on Superconducting Processors*. https://arxiv.org/abs/2610.03869
- [R2] Author implementation, pinned source commit `458210ac74ed0047760b8789c169e6b9eff732b8`. https://github.com/LangeMoritz/modular-gnn-rnn-decoder/tree/458210ac74ed0047760b8789c169e6b9eff732b8
- [R3] Conda, *Managing environments*. https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
- [R4] Stim 1.15.0 release notes, including Pauli targets for `OBSERVABLE_INCLUDE`. https://github.com/quantumlib/Stim/releases/tag/v1.15.0
- [R5] Stim gate reference, `OBSERVABLE_INCLUDE`. https://github.com/quantumlib/Stim/blob/main/doc/gates.md#the-observable_include-instruction
- [R6] PyTorch, *Multiprocessing best practices*, CPU oversubscription. https://docs.pytorch.org/docs/2.7/notes/multiprocessing.html#cpu-in-multiprocessing
- [R7] PyTorch, *Reproducibility*. https://docs.pytorch.org/docs/2.7/notes/randomness.html
- [R8] X. Li et al., *Dynamic Chunk Convolution for Unified Streaming and Non-Streaming Conformer ASR*. https://arxiv.org/abs/2304.09325
- [R9] K. Zhang et al., *Learning to Decode in Parallel: Self-Coordinating Neural Network for Real-Time Quantum Error Correction*. https://arxiv.org/html/2601.09921v1
