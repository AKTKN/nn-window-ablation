# Repository design for the neural window-decoding ablation

Design version: 2026-11-22. Status: proposed specification, prior to implementation.

## 1. Scope and status

This document specifies the planned organization of the simulator, decoder adaptation, and experiment pipeline. The repository name is `nn-window-ablation`, the Python import name is `nn_window_ablation`, and the command-line entry point is `nnwa`. P0 provides configuration validation in the installed package. See the [implementation status](../IMPLEMENTATION_STATUS.md) for executed checks and pending scientific and cluster work.

The initial repository should support one scientific comparison: a model trained with complete, closed syndrome histories is evaluated on finite open windows, and its performance is compared with the same architecture trained with the intended window context. The primary model has one query-conditioned scalar head for both global and core targets. The same offline checkpoint $\theta_F$ defines G (one global query on full history), F (XOR of all core queries on full history), and S (XOR of the same core queries on their windows). An unmodified upstream Lange model may be retained as a separate reproduction reference; it is not G. The repository should record these inference choices without introducing separate primary global/core heads or a general decoder framework.

The initial execution path is Linux, Python 3.11, Conda, and CPU simulation, preprocessing, and training. Development takes place locally; versioned source is later cloned from GitHub onto the cluster, substantive computation runs under PBS, and selected results return through compressed archives and `scp`. The first scientific pilot starts directly at $d=5$ from random initialization, retaining hidden width 256 and four GRU layers. Numerical compatibility and resource budgets will be measured on the actual CPU allocation. The package documentation and upstream source have been inspected; the proposed adaptation and simulator have not been run.

## 2. Package structure and responsibilities

Use one installable Python package with a `src` layout and one `pyproject.toml`. PyPA documents that the `src` layout separates importable application code from the repository root and requires an installation step, typically an editable installation during development. This lets scripts, notebooks, and cluster jobs import the same installed package from any invocation directory. The P0 entry point is declared in `[project.scripts]`. [1, 2]

The proposed separation of source, datasets, run artifacts, and analysis follows common research-project conventions exemplified by Cookiecutter Data Science. The QEC-specific organization follows the Lange repository's separation of circuits, shot sources, graph features, models, training, and experiment scripts; Sinter additionally illustrates separating sampling/decoding from persistent Monte Carlo statistics. These are organizational references, not additional runtime dependencies. [6, 34, 37]

The following is a proposed layout. Files are grouped by responsibility; additional subpackages should be introduced only when a module becomes difficult to maintain.

| Path | Responsibility |
|---|---|
| `README.md` | Environment setup, the minimal experiment sequence, and links to scientific definitions. |
| `pyproject.toml` | Package metadata and optional `experiment` scientific dependencies; the CLI entry point is added in P0. |
| `environment.yml` | Minimal Conda base specification containing Python 3.11 and pip; not a resolved environment lock. |
| `docs/` | Scientific protocol, label conventions, repository design, and experiment records. |
| `src/nn_window_ablation/cli.py` | Command dispatch and Hydra entry point. |
| `src/nn_window_ablation/config.py` | Typed configuration validation and conversion to ordinary Python objects. |
| `src/nn_window_ablation/conf/` | Packaged Hydra groups, CPU runtime profiles, and `pilot_d5`/later campaign presets. |
| `src/nn_window_ablation/simulation.py` | Physical circuit construction, auxiliary observable annotations, sampling, and circuit metadata. |
| `src/nn_window_ablation/windows.py` | Core partitions, observation intervals, detector availability, and window masks. |
| `src/nn_window_ablation/features.py` | Conversion of the available raw detectors into GNN inputs. |
| `src/nn_window_ablation/data.py` | Packed-array shards, manifests, minibatches, and deterministic training-chunk generation. |
| `src/nn_window_ablation/models/gnn_rnn.py` | Adapted Lange encoder, recurrent module, query metadata, and one shared scalar head. |
| `src/nn_window_ablation/training.py` | Loss, optimization, checkpoint selection, and resumable training state. |
| `src/nn_window_ablation/evaluation.py` | Inference on fixed shots, core aggregation, and baseline decoding. |
| `src/nn_window_ablation/analysis.py` | Paired error comparisons, confidence intervals, tables, and figures. |
| `src/nn_window_ablation/workflows.py` | Composition of these components into the five CLI workflows. |
| `scripts/pbs/*.pbs.example` | Site-adjustable runtime-probe and pilot job templates; no automatic submission or assumed queue. |
| `tests/` | Scientific invariants and a small end-to-end execution check. |
| `third_party/` | Upstream license, source provenance, and an adaptation record. |

The Python source placeholders document these responsibilities; their presence does not mean their behavior is implemented. The simulation, window, feature, and model interfaces should accept explicit arrays and typed parameters. Scientific modules should not import Hydra, read its global state, discover files from the working directory, or decide which experiment to run. `workflows.py` will connect these operations and write run records. This separation allows the same window-construction function to be used for training, frozen-model evaluation, and leakage tests.

Default configurations can be packaged under `conf/`, so the installed entry point does not depend on an implicit repository-relative path. User configuration directories can be supplied explicitly. Large datasets, checkpoints, and run outputs should live under configurable data and run roots outside tracked source files. Notebooks may read completed results, but dataset generation, training, and the statistics used in the report should remain available through the package.

## 3. Dependencies and upstream integration

The initial runtime dependencies should be limited to the packages in the following table. The table states their proposed use and the constraints needed for this experiment; package availability does not supply the study-specific window and target semantics.

| Package | What it supplies and how it is used | Constraint for this study |
|---|---|---|
| Stim | Build the memory circuit; use a compiled detector sampler to obtain detector events and original/auxiliary observable labels jointly. [10, 11, 19] | Preserve explicit cut and availability metadata. Use the Pauli-observable-capable sampling path for teacher labels; a measurement-record-only conversion cannot recover those labels. |
| PyMatching | Build `Matching.from_detector_error_model` from the original global-only DEM and run `decode_batch` on the fixed detector shots. [20] | Record DEM decomposition and matching options. Auxiliary teacher observables are absent from this decoding model. This is the full-history matching reference. |
| PyTorch | Reuse the GRU backbone, train the common scalar logit with `BCEWithLogitsLoss`, and use `torch.optim` for parameter updates. [7, 21, 22] | Supply logits to the loss; global and core queries use the same head and weights. Match optimizer-reset and update-budget policies across continuation arms. |
| PyTorch Geometric | Use `GraphConv`, graph batching, and graph pooling for the adapted Lange encoder. [23, 24] | Crop raw observations first. Check that batching preserves graph, shot, temporal-slot, and query identities, including empty observed slots. |
| Hydra and OmegaConf | Compose configuration groups, validate/resolve the CLI configuration, record overrides, and enumerate declared sweeps. [15–17, 30–32] | Keep them at the configuration/orchestration boundary; no scientific module imports Hydra. |
| NumPy | Pack bits into `uint8`, create/read `.npy` shards, memory-map fixed datasets, and manipulate predictions. [12–14] | Record valid bit counts, bit order, shapes, and row identity. |
| SciPy | Use `binomtest(...).proportion_ci(method="wilson")` for Wilson intervals; `bootstrap(..., paired=True)` resamples common shot indices for paired comparisons. [25, 26] | Set the CI method explicitly. Follow the protocol's conservative exact-count construction for sparse discordances; a marginal LER interval is not a paired-difference interval. |
| Matplotlib | Generate reproducible LER, paired-difference, and buffer-dependence figures from recorded results. [27] | Plot stored statistics and their declared intervals; keep plotting independent of model execution. |
| Conda and pip | Create an isolated Python environment, install the CPU PyTorch candidate, then install the project's `experiment` extra; use `conda run` for explicit batch execution. [3, 4, 38] | The base YAML is not a lock. Validate the actual CPU environment before recording resolved package inventories. |
| pytest | Run the scientific invariants and minimal execution checks. [28] | Development dependency; tests focus on target, availability, pairing, and checkpoint contracts. |

Python's standard `csv` and `json` modules are sufficient for initial metric tables and manifests; pandas is not required. PyArrow for larger Parquet result tables, Sinter for separately managed CPU baseline campaigns, and MLflow for later run tracking are optional additions. Sinter is not the source of the paired NN test data. SpeechBrain's streaming Conformer material is a conceptual reference for dynamic context training; the QEC repository has no ASR runtime dependency. [29, 33–36]

Use `environment.yml` to create the Python 3.11/pip base, then bootstrap the candidate PyTorch 2.7.0 CPU wheel from the official CPU index before installing the editable project with its `experiment` extra. This installation order preserves explicit selection of the PyTorch build. Conda documents using an isolated environment and installing remaining pip dependencies after Conda packages. PyG supports a minimal installation without mandatory extension packages; add CPU-compatible compiled extensions only if the selected operators require them. [3, 5, 38]

Python 3.11 and the upstream-derived package constraints are starting candidates, not a validated combination. P0 checks installation and imports; later phases add a generated Stim batch, the selected PyG operators, and a genuine $d=5$ forward/backward step. Save the final Conda and pip package inventories after these checks, distinguishing portable dependency intent from an actual Linux environment export. No `uv.lock` is supplied. Each run records Python and package versions, CPU identity/topology, allocated CPU resources, effective thread/worker counts, and precision settings. The environment record alone does not specify the execution allocation. [3]

The inspected upstream requirements provide a concrete reproduction candidate: PyTorch 2.7.0, PyG 2.6.1, Stim 1.15.0, PyMatching 2.2.2, NumPy 2.2.6, SciPy at least 1.15, Matplotlib 3.10.3, and SciencePlots 2.1.1. These are upstream declarations, not a validated lock for this project. [18]

The source basis for the neural model is the official repository accompanying Lange et al., *Modular Neural Decoder for Surface Code Memory and Logic on Superconducting Processors*. Use the inspected snapshot `458210ac74ed0047760b8789c169e6b9eff732b8`, retain the MIT license and copyright notices, and record which files were adapted. This is a source integration, not an assumption that the upstream repository provides a supported pip-installable decoder API. [6]

The initial adaptation should preserve the GNN/CNN/GRU backbone as far as the new task permits, while adding global/core query metadata and availability information before the shared scalar readout. The upstream model returns a final sequence output; its two-round GNN feature groups are not a commit/buffer protocol. The upstream data path also has a nontrivial-shot option. Our main data path should retain zero-event shots, and an observed empty window must remain distinguishable from unavailable context. [7–9]

The upstream checkpoint, if used for a reproduction check, and the new query-conditioned model should have distinct model identifiers. Train $\theta_F$ jointly on global and core queries, all with full closed context. G, F, and S then load exactly that checkpoint and shared head. F and S differ only in observation context; both retain the same core targets, threshold, state-reset rule, and XOR aggregation. Subsequent S+/W/D continuation arms retain the same query-conditioned architecture and the global objective, following the protocol document.

The first $d=5$ pilot initializes the backbone, distance-5 spatial aggregator, query-metadata projection, and shared readout randomly; all parameters are trainable. Retain hidden width 256 and four GRU layers on CPU. A $d=3$ parent checkpoint is not a prerequisite. Distance-transfer initialization may be studied later as a separate policy, recording the parent checkpoint and copied parameter names. The inspected upstream `init_from` copies matching names and shapes, including the scalar head, and can inform that later implementation. Do not mix initialization policies inside the primary context-training comparison. [7]

## 4. Scientific interfaces and data boundaries

Use small dataclasses or typed records at module boundaries. Their initial contracts are:

| Record | Essential fields |
|---|---|
| `CircuitBundle` | Original and teacher circuits, cut instruction positions, detector coordinates/availability, observable schema, hashes. |
| `ShotBatch` | Packed detectors, global labels, core labels, dataset/shard/row identities. |
| `WindowSpec` | Target core or global query, selected detector indices, release phase, buffer lengths, physical-boundary flags. |
| `ModelBatch` | Event graphs, sequence lengths, query metadata, and availability metadata; no target columns. |
| `PredictionBatch` | Aligned shot/query identities, logits, and hard decisions. |

`features.py` converts a `ShotBatch` and `WindowSpec` into a `ModelBatch`. The model's forward call receives only that `ModelBatch`. Training/evaluation code retains the labels separately. These records specify responsibilities; their Python implementation remains to be written.

### 4.1 Circuit and target construction

`simulation.py` should construct one physical memory experiment and return both its Stim circuit and explicit structural metadata. In addition to detector coordinates, metadata should identify measurement availability, the initialization and final-readout regions, and the exact instruction boundaries chosen as temporal cuts. These boundaries should be recorded by the circuit builder rather than reconstructed later from the maximum detector time coordinate.

For the proposed target construction, a clean complete-round cut is placed immediately after the ancilla measurement/reset operations and before the after-reset noise assigned to the next round. Auxiliary logical Pauli observables are inserted at these recorded cuts. The final segment incorporates the actual final logical measurement record, so the initial and terminal contributions are part of the same labeling convention. The precise convention and its algebra belong in the scientific-protocol document; this module should implement that convention once.

Keep two separately identified circuit/DEM artifacts: the original circuit with its global logical observable, and the circuit carrying additional teacher observables. The added annotations supply labels; they do not add physical measurements. Stim documents Pauli-target `OBSERVABLE_INCLUDE` annotations and their use in detector/error-model sampling. Their treatment differs from a measurement-record-only conversion, so the labeling sampler must be chosen accordingly. [10, 11]

Generate detector bits, the original global label, and auxiliary labels jointly for each shot from the annotated circuit using the appropriate detector-sampling interface. Separate observable columns by the recorded schema. Independently sampling labels from a second circuit would not preserve the shotwise correspondence required by this study.

The original global-only DEM is used for the ordinary memory-decoding baseline. Teacher labels, auxiliary observable indices, and simulator frame information must never become decoder input features. The initial teacher-label schema stores one label per core in the fixed $c=d$ partition. The 12 labels for $T=12d$ must satisfy the prescribed shotwise XOR identity with the original global observable.

### 4.2 Window and graph construction

`windows.py` should return a window specification containing the core boundaries, available detector indices, observation-arrival cutoff, left and right context, and initialization/final-readout status. The same object determines the target query and the model masks. It should distinguish an unavailable detector from an available detector whose measured value is zero.

Availability must include an event phase: the final ancilla round at $T$ and the subsequent destructive data readout at $T^+$ are distinct. A nonfinal core released after the last ancilla round commits before the final data readout, even when its nominal right-context endpoint equals $T$. Under this policy, only the final core receives the destructive-readout detectors; full offline context includes them. Apply the same endpoint policy to all finite-window and all-past comparison conditions. A test based only on a detector coordinate being at most $T$ is insufficient.

The processing order is fixed: load raw detector bits, select the available observations, then construct graph features. A full-history graph must not be built and subsequently sliced. This matters for the upstream two-round feature grouping: a group beginning at one round may use the following round, which must also be available. Graph topology, pooling, sequence length, positional features, and normalization must all obey the same observation restriction. [8]

For the first finite-window study, recurrent state should be reset for each window. Persisting state from earlier windows changes the accessible information and should be a separately configured experiment. The architecture may support a hidden-state interface later, but an unrecorded state carried across windows would prevent a well-defined finite-window comparison.

## 5. Dataset format and experiment records

### 5.1 Immutable validation and test data

Generate validation and test datasets once for each physical setting, then evaluate all relevant decoder conditions on the same shot identities and the same raw detector records. A dataset is immutable after its manifest is finalized. New samples or changed label conventions produce a new dataset identifier.

Write each shard to a temporary path, finish and checksum it, then rename it into place. Publish the finalized manifest only after all listed shards are complete. Readers consume only finalized manifests. Dataset identity includes the physical circuit, label schema, split, and generation recipe; a human-readable alias such as `paired_d5` is not the authoritative identity.

Use sharded, uncompressed `.npy` files containing packed `uint8` arrays. For a shard with $N$ shots and $M$ detector bits per shot, the detector payload is $N\lceil M/8\rceil$ bytes, excluding the small file header. Store rows as shots, pack along the final axis, and specify `bitorder="little"` in the format. NumPy documents both bit ordering and zero padding in `packbits`; loading a `.npy` array with `mmap_mode="r"` permits access to selected portions without reading the whole array. `open_memmap` supports incremental creation of such files. [12–14]

The initial dataset contents should be:

| Artifact | Contents |
|---|---|
| `manifest.json` | Dataset identity, physical setting, split, schema versions, source provenance, seed recipe, shard inventory, array shapes, bit counts, and checksums. |
| `circuit_global.stim`, `global.dem` | The physical experiment and its original global-only decoding model. |
| `circuit_teacher.stim` | The same physical operations with the teacher-label annotations. |
| `structure.json` and numeric metadata arrays | Cut locations, detector availability, coordinate mapping, observable meanings, and allowed core queries. |
| `shards/<id>/detectors.npy` | Packed raw detector events. |
| `shards/<id>/global_observables.npy` | Packed global logical labels. |
| `shards/<id>/core_labels.npy` | Packed labels for the fixed 12-core partition, with $c=d$. |

The initial schema fixes $c=d$ and stores these per-core labels directly. Changing buffers does not change the core partition or require new labels. Labels at every round cut would support later core-width experiments, but are a separate schema extension rather than an initial storage requirement. Padding bits and the number of valid bits must be specified separately. A shot identifier can be the tuple `(dataset_id, shard_id, row_index)`; duplicating it in every array is unnecessary.

Unpack only the minibatch and temporal columns required for the current operation. Keep graph objects transient initially. A persistent graph cache would duplicate data for different windows and feature versions; introduce one only after profiling shows that graph construction limits throughput.

### 5.2 Training data and reproducibility

The initial training default is on-demand generation in deterministic chunks. Derive each chunk's seed from a recorded master seed, dataset family, split, and chunk identifier. Assign chunk identifiers independently of worker scheduling, so changing which worker handles a chunk does not accidentally duplicate or omit it. Record the sampler version, chunk size, and seed derivation version as well as the seed.

This is a reproducible generation recipe within its specified implementation and environment; it is not a substitute for immutable evaluation files. Reusable training shards remain an option when generation or graph construction dominates runtime. Training-context conditions should use the same physical distribution and, where feasible, the same chunk sequence, with their observation policy applied after sampling. Zero-event shots remain in both training and evaluation distributions.

### 5.3 Run and checkpoint records

Each run should save its fully resolved configuration, command, project Git commit and local-change status, upstream model SHA, environment information, random seeds, dataset IDs, and label/feature schema versions. A resumable training checkpoint should also contain optimizer/scheduler state, step number, random-generator states, and data-position information. The primary protocol uses fixed-budget checkpoints: $U_0$ full-context updates for G/F/S and $U_1$ additional updates for each continuation arm. Pilot validation determines these budgets; main-run validation diagnoses failed runs and does not select a more favorable checkpoint after the budget is fixed.

Evaluation outputs should preserve a per-shot final prediction or failure indicator accompanied by the exact global-label dataset identity. Store the 12 core decisions as a packed array for every core-query evaluation so cancellation and neighboring-core diagnostics can be reproduced. Save full-precision logits on a predeclared diagnostic subset, or throughout when the storage budget permits. Paired comparisons require row alignment, not merely aggregate error totals. Store compact metric tables in CSV initially; Parquet is an optional extension when the number of runs or table size justifies it. Plots should be regenerated from these tables and recorded per-shot results.

Separate three checkpoint operations: `checkpoint.path` loads a model for evaluation; `checkpoint.init_from` initializes a distance-transfer model or forks an adaptation arm with fresh optimizer/scheduler state; `checkpoint.resume_from` restores the same interrupted run, including optimizer state, random states, and data position. Initialization and resumption are mutually exclusive. A resumed run must match its saved scientific configuration and sampling schedule.

## 6. Configuration and command-line design

Hydra should compose settings and record runs; it should not appear in the numerical implementation. Its documentation specifies per-run output directories, saved configuration files, and configurable working-directory behavior. Use `version_base="1.3"` in the Hydra entry point, set `hydra.job.chdir: false` explicitly, and use absolute data/run roots after resolving the configuration. [15, 32]

A thin `nnwa` dispatcher should consume one command verb and pass the remaining arguments to a Hydra-decorated runner. This preserves Hydra's command-line overrides and multirun facility. Using only the Compose API would require separately implementing multirun and output management, because Hydra explicitly excludes those facilities from the Compose API. [16, 17]

The five commands have the following proposed contracts:

| Command | Input and result |
|---|---|
| `generate` | Physical setting, split, seed recipe, and shot count; produces immutable detector/label shards and a manifest. |
| `train` | Model and training-context configuration plus a training source and validation dataset; produces checkpoints and learning records. |
| `evaluate` | Checkpoint or baseline, fixed dataset, and observation policy; produces aligned predictions and metrics. |
| `analyze` | Completed evaluation records; produces paired statistics, tables, and figures. |
| `validate` | Configuration, dataset, or checkpoint selection; checks compatibility and the relevant scientific invariants. |

Configuration groups should correspond to actual experimental choices. The following file paths are relative to `src/nn_window_ablation/conf/` and define the initial group names used by the proposed CLI examples.

| Configuration path | Initial contents or meaning |
|---|---|
| `config.yaml` | Defaults composition, seed, checkpoint paths, data/run roots, run name, and Hydra settings. |
| `code/surface_rotated_z.yaml` | `family: surface_code`, `layout: rotated`, `memory_basis: Z`, `distance: 5`, `rounds_per_distance: 12`. |
| `noise/uniform_circuit.yaml` | `name: uniform_circuit`, `probability: 0.003`; the physical instruction probabilities follow the protocol's generator. |
| `window/fixed_d.yaml` | `core_per_distance: 1`, `left_per_distance: 1`, `right_per_distance: 1`, `reset_recurrent_state: true`. |
| `model/query_gnn_rnn.yaml` | `name: query_gnn_rnn`, `head: shared_scalar`; one head supports global and core queries. |
| `training/full_closed.yaml` | Global and sampled-core queries both use full closed context; equal total objective weights; `max_steps`, `batch_size`, and `learning_rate` initially `null`, to be determined from the pilot. |
| `data/paired.yaml` | Paired dataset identity/split, `train_source: generated_chunks`, `bitorder: little`, and `num_workers: 0`. |
| `runtime/cpu.yaml` | Cluster CPU profile: `device: cpu`, `intraop_threads`, and `interop_threads`; thread settings must fit the approved allocation. |
| `runtime/local_cpu.yaml` | Conservative local CPU settings for the initial correctness checks; scientific parameters are unchanged. |
| `evaluation/G.yaml` | `query: global`, `context: full_closed`, `aggregation: identity`. |
| `evaluation/F.yaml` | `query: all_cores`, `context: full_closed`, `aggregation: xor`. |
| `evaluation/S.yaml` | `query: all_cores`, `context: deployment_window`, `aggregation: xor`. |
| `analysis/primary.yaml` | Predeclared G/F/S and continuation contrasts, paired inference, and result-selection rules. |
| `experiment/pilot_d5.yaml` | Default pilot: $d=5$, $T=60$, $c=b_L=b_R=5$, $p=0.003$, random initialization, and undecided numerical training budgets. |
| `experiment/paired_d5.yaml`, `experiment/paired_d7.yaml` | Later main-campaign presets with the same dimensionless ratios; activated after the pilot sets concrete resource budgets. |

The proposed `conf/config.yaml` is:

```yaml
# Hydra task configuration.
defaults:
  - code: surface_rotated_z
  - noise: uniform_circuit
  - window: fixed_d
  - model: query_gnn_rnn
  - training: full_closed
  - data: paired
  - evaluation: G
  - analysis: primary
  - runtime: cpu
  - _self_
  - experiment: pilot_d5

schema_version: 1
seed: 101
# P0 CLI assigns a UUID-based name when the user does not supply one.
run_name: ???
paths:
  data_root: ${oc.env:NNWA_DATA_ROOT,./data}
  run_root: ${oc.env:NNWA_RUN_ROOT,./runs}
checkpoint:
  path: null
  init_from: null
  resume_from: null
hydra:
  job:
    chdir: false
  run:
    dir: ${paths.run_root}/${run_name}/hydra
  sweep:
    dir: ${paths.run_root}/${run_name}/hydra_sweep
    subdir: ${hydra.job.num}
```

The default `conf/experiment/pilot_d5.yaml` is:

```yaml
# @package _global_
study_stage: pilot
code:
  distance: 5
  rounds_per_distance: 12
noise:
  probability: 0.003
window:
  core_per_distance: 1
  left_per_distance: 1
  right_per_distance: 1
data:
  family_id: pilot_d5
```

The experiment preset has global scope, so its fields override the corresponding composed groups rather than create an unrelated `experiment.code` object. Placing `_self_` before `experiment` makes the preset apply after root defaults; explicit CLI overrides then select the requested group or field. This order follows Hydra's Defaults List composition rules. [30]

P0 resolves `paths.data_root` and `paths.run_root` into absolute paths relative to the original invocation directory when their configured values are relative. Their environment-variable defaults are convenience settings, not evidence that those paths are shared across cluster nodes. The runtime profile controls CPU execution through `runtime.device`, `runtime.intraop_threads`, and `runtime.interop_threads`; `data.num_workers` controls loader processes. These fields cannot modify code distance, model width/depth, noise, or query semantics. Training fields left `null` are undecided and must be assigned measured values before a training invocation is accepted.

The typed resolution step in `config.py` computes $T=\mathtt{rounds\_per\_distance}\,d$ and $c=\mathtt{core\_per\_distance}\,d$. For buffers it uses $b_L=\lceil d\,\mathtt{left\_per\_distance}\rceil$ and the analogous right-buffer expression. Thus the initial setting resolves to $T=60$, $c=b_L=b_R=5$, while a buffer ratio of 0.5 gives the prescribed $\lceil d/2\rceil$. Validate integer $T,c$, odd $d\ge3$, the fixed $c=d$, divisibility of $T$ by $c$, and initial buffer ratios in $\{0,0.5,1\}$. Save both the input ratios and resolved integer lengths. Plain OmegaConf node interpolation is not an arithmetic evaluator: a string such as `${code.distance} * 12` does not calculate $12d$. No arithmetic resolver is needed when the typed configuration-resolution step performs these calculations. [31]

The CLI supplies a UUID-based `run_name` before invoking Hydra if the caller omits it; the YAML requires a resolved name. Both Hydra output roots include that name. New runs must not silently reuse an existing unrelated directory. Multirun jobs receive unique child directories under their uniquely named parent. This prevents concurrent PBS starts from colliding at second-resolution timestamps.

Training and evaluation configurations have different roles. A training configuration describes the information used to optimize a checkpoint. The `evaluation` group selects G, F, or S from a loaded checkpoint and must not alter its weights, architecture, or recorded training objective. Additional `training/continued_full.yaml`, `training/fixed_window.yaml`, and `training/dynamic_window.yaml` presets encode S+, W, and D from the protocol, without changing the model architecture or core definitions.

Continuation `training.max_steps` counts new phase-local updates, excluding the parent checkpoint count. The `data_stream_phase` is `base` for the offline model and `continuation` for all S+/W/D arms. The latter namespace is separate from base training and shared across the three arms; it must not be derived from their differing training names. Resume restores the phase-local cursor and optimizer state.

The statistical condition identifier combines checkpoint provenance with inference mode. In particular, `evaluation.id=S` applied to full, continued-full, fixed-window, and dynamic-window checkpoints denotes S, S+, W, and D, respectively. Record both the inference mode and immutable checkpoint training regime; the caller's current training preset cannot relabel an existing checkpoint.

The first command uses P0's validation workflow. The scientific commands illustrate interfaces for later phases. They assume an active Conda environment, completed storage checks, and validated numerical budgets in the pilot configuration. `pilot_d5` is the default experiment. Artifact paths and dataset identifiers are illustrative.

```bash
nnwa validate experiment=pilot_d5 runtime=local_cpu
nnwa generate experiment=pilot_d5 runtime=cpu data.split=validation
nnwa generate experiment=pilot_d5 runtime=cpu data.split=test
nnwa train experiment=pilot_d5 runtime=cpu training=full_closed seed=101
nnwa evaluate --multirun experiment=pilot_d5 runtime=cpu evaluation=G,F,S checkpoint.path=/runs/pilot_d5/full_closed/101/model.pt data.split=test data.id=pilot_d5_test_v1
nnwa train --multirun experiment=pilot_d5 runtime=cpu training=continued_full,fixed_window,dynamic_window checkpoint.init_from=/runs/pilot_d5/full_closed/101/model.pt seed=101
nnwa analyze experiment=pilot_d5 runtime=local_cpu analysis=primary
```

In a noninteractive batch shell, the equivalent prefix is `conda run --no-capture-output -p /path/to/nnwa_cpu nnwa ...`, using an explicit Conda executable or a correctly initialized shell. The `.pbs.example` templates are reviewed and configured before any later submission; no queue is assumed and no example submits itself. Detailed cluster preparation and result export are specified in `04_cluster_workflow.md`. [4]

After the $d=5$ pilot fixes the budgets, a main-campaign sweep can use `nnwa train --multirun experiment=paired_d5 runtime=cpu training=full_closed seed=101,102,103`. It does not require distance-3 preparation. Add `paired_d7` only after a feasible resource estimate exists, with the initialization policy recorded. Each G/F/S evaluation reuses one completed checkpoint and the same dataset. Buffer evaluations follow only the predeclared sweep in the protocol. Hydra enumerates combinations; it neither allocates PBS resources nor infers checkpoint dependencies. Configuration composition occurs when jobs launch, so recorded sweeps use a fixed code/configuration snapshot. [17]

## 7. Sweep budgeting and validation gates

Separate the number of training jobs from the number of evaluation conditions. For example, if there are $N_s$ training seeds, $N_a$ training strategies, and $N_m$ distinct model/physical training settings, then the planned number of training jobs is $N_sN_aN_m$. Applying each trained checkpoint to $N_w$ window settings multiplies evaluation jobs by $N_w$, without requiring the offline model to be retrained for each window. A window-aware model may have its own specified training-window distribution; that distribution belongs to the training-strategy definition.

Before a large sweep, save a plan table listing every job's configuration identity, checkpoint dependency, dataset identity, shot count or optimization-step budget, and output directory. Measure runtime and memory at $d=5$ with the retained 256-wide, four-layer architecture before assigning shard size, batch size, and update budgets. Start with one CPU training process, inter-operation threads set to one, and no data-loader workers. Benchmark intra-operation thread counts within the actual PBS allocation; the cluster workflow proposes 1, 8, 16, 32, 48, and 96 when all 96 CPU resources are allocated. A thread count is not a request for that many training or loader processes. PBS wrappers select execution resources without moving scientific logic into scheduler scripts. [39, 40]

The implementation order follows `05_implementation_phases.md`: P0 installs the CPU package and implements configuration/CLI handling; P1 establishes the circuit and labels; P2 establishes windows and features; P3 adapts the model and profiles the $d=5$ workload; P4 implements data, training, and resume; P5 evaluates frozen G/F/S; P6 evaluates matched continuation; P7 consolidates PBS campaigns and artifact export. Bring the minimal runtime-probe and pilot PBS wrappers forward when P3/P4 first require cluster execution. No long local training or $d=3$ warm-start is a prerequisite for cluster integration.

Implementation should first satisfy a small set of scientific invariants: the teacher annotations preserve the intended physical circuit and detector statistics; the local targets obey the final XOR identity; initialization and final readout are covered; graph inputs contain no observations later than their availability cutoff; and changing unavailable future bits cannot change a window's features or prediction. Include checks for the $T$/$T^+$ endpoint policy, zero-event shots, rejection of a partial final core under the initial divisible-length contract, and packed-array round trips. These are gates for the experimental interpretation, rather than an attempt to reproduce every implementation detail in tests.

## 8. Decisions to settle before implementation

| Decision | Proposed initial choice |
|---|---|
| Repository, package, and CLI names | `nn-window-ablation`, `nn_window_ablation`, and `nnwa`, respectively. |
| Primary model task | One global/core query-conditioned model and shared head; G/F/S reuse exactly $\theta_F$. The unmodified upstream model is only a reproduction reference. |
| Recurrent information across windows | Reset state for the finite-window ablation. |
| Training data source | Deterministically generated chunks; switch to reusable shards if profiling supports it. |
| Validation/test data | Immutable packed shards with common shot identities for all paired comparisons. |
| Teacher-label granularity | Direct labels for the fixed $c=d$ partition: 12 per trajectory when $T=12d$. |
| Pilot and initialization | Begin at $d=5$ from random initialization; retain the 256-wide, four-layer architecture. |
| Dependency versions | Resolve the Conda/Python 3.11/CPU-PyTorch candidate on the actual machine and record verified inventories; the scaffold supplies no environment lock. |
| Numerical budgets | Keep training steps, batch size, and learning rate `null` until the $d=5$ profile and pilot provide their values. |
| Initial orchestration | Local development, future CLI/Hydra entry point, and reviewed CPU PBS templates; Conda execution and compressed result export. |

## References

Sources were inspected on 22 November 2026. The architectural and storage choices above are proposals; cited documentation establishes the relevant package behavior and upstream interfaces.

1. PyPA, [src layout vs flat layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/).
2. PyPA, [Writing your pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/).
3. Conda, [Managing environments](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html).
4. Conda, [conda run](https://docs.conda.io/projects/conda/en/stable/commands/run.html).
5. PyTorch Geometric, [Installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
6. Lange et al., [official source repository, snapshot 458210a](https://github.com/LangeMoritz/modular-gnn-rnn-decoder/tree/458210ac74ed0047760b8789c169e6b9eff732b8), accompanying [arXiv:2610.03869](https://arxiv.org/abs/2610.03869).
7. Lange et al., [model.py at the inspected snapshot](https://github.com/LangeMoritz/modular-gnn-rnn-decoder/blob/458210ac74ed0047760b8789c169e6b9eff732b8/gnnrnn/model.py).
8. Lange et al., [graphs.py at the inspected snapshot](https://github.com/LangeMoritz/modular-gnn-rnn-decoder/blob/458210ac74ed0047760b8789c169e6b9eff732b8/gnnrnn/graphs.py).
9. Lange et al., [data.py at the inspected snapshot](https://github.com/LangeMoritz/modular-gnn-rnn-decoder/blob/458210ac74ed0047760b8789c169e6b9eff732b8/gnnrnn/data.py).
10. Stim, [v1.15 gate reference: OBSERVABLE_INCLUDE](https://github.com/quantumlib/Stim/blob/v1.15.0/doc/gates.md#the-observable_include-instruction).
11. Stim, [release notes](https://github.com/quantumlib/Stim/releases).
12. NumPy, [numpy.packbits](https://numpy.org/doc/stable/reference/generated/numpy.packbits.html).
13. NumPy, [numpy.load](https://numpy.org/doc/stable/reference/generated/numpy.load.html).
14. NumPy, [numpy.lib.format.open_memmap](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.open_memmap.html).
15. Hydra 1.3, [Output/working directory](https://hydra.cc/docs/1.3/tutorials/basic/running_your_app/working_directory/).
16. Hydra 1.3, [Compose API](https://hydra.cc/docs/1.3/advanced/compose_api/).
17. Hydra 1.3, [Multi-run](https://hydra.cc/docs/1.3/tutorials/basic/running_your_app/multi-run/).
18. Lange et al., [requirements.txt at the inspected snapshot](https://github.com/LangeMoritz/modular-gnn-rnn-decoder/blob/458210ac74ed0047760b8789c169e6b9eff732b8/requirements.txt).
19. Stim, [Python API reference: circuit and compiled detector sampling](https://github.com/quantumlib/Stim/wiki/Stim-v1.13-Python-API-Reference). Pauli-observable semantics use the later v1.15 references above.
20. PyMatching, [Python API: Matching and decode_batch](https://pymatching.readthedocs.io/en/stable/api.html).
21. PyTorch, [BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.7/generated/torch.nn.BCEWithLogitsLoss.html).
22. PyTorch, [torch.optim](https://docs.pytorch.org/docs/2.7/optim.html).
23. PyTorch Geometric, [GraphConv](https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.nn.conv.GraphConv.html).
24. PyTorch Geometric, [Advanced Mini-Batching](https://pytorch-geometric.readthedocs.io/en/2.6.1/advanced/batching.html).
25. SciPy, [BinomTestResult.proportion_ci](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html).
26. SciPy, [bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).
27. Matplotlib, [Quick start guide](https://matplotlib.org/stable/users/explain/quick_start.html).
28. pytest, [Get Started](https://docs.pytest.org/en/stable/getting-started.html).
29. Python standard library, [csv](https://docs.python.org/3/library/csv.html) and [json](https://docs.python.org/3.11/library/json.html).
30. Hydra 1.3, [The Defaults List](https://hydra.cc/docs/1.3/advanced/defaults_list/).
31. OmegaConf, [Usage: interpolation and configuration resolution](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html).
32. Hydra 1.3, [version_base](https://hydra.cc/docs/1.3/upgrades/version_base/).
33. Apache Arrow, [Reading and Writing Single Parquet Files](https://arrow.apache.org/docs/python/parquet/parquet.html).
34. Sinter, [maintainer's package documentation](https://pypi.org/project/sinter/).
35. MLflow, [ML Experiment Tracking](https://mlflow.org/docs/latest/ml/tracking/).
36. SpeechBrain, [Streaming Speech Recognition with Conformers](https://speechbrain.readthedocs.io/en/latest/tutorials/nn/conformer-streaming-asr.html).
37. DrivenData, [Cookiecutter Data Science: directory structure](https://cookiecutter-data-science.drivendata.org/#directory-structure).
38. PyTorch, [Previous versions and official CPU wheel commands](https://pytorch.org/get-started/previous-versions/).
39. PyTorch, [Threading environment variables](https://docs.pytorch.org/docs/stable/threading_environment_variables.html).
40. PyTorch, [Multiprocessing best practices: CPU oversubscription](https://docs.pytorch.org/docs/2.7/notes/multiprocessing.html#cpu-in-multiprocessing).
