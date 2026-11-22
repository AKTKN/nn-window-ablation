# nn-window-ablation

Repository for a neural surface-code decoder study. The project
compares complete, closed syndrome-history inference with finite-window inference
and then measures the effect of training with the deployment context. The model
uses the modular Lange GNN–RNN backbone with a study-specific, shared global/core
query interface [1, 2]. The window targets follow the direct logical-contribution
composition described in the scientific documents, with Zhang et al. as the
reference for that construction [3].

## Current state

P0 provides an installed `nnwa validate` command, typed configuration, and
local CPU verification. P1 provides the Stim circuit and joint sampler for
detectors and exact logical labels. P2 provides observation windows,
Lange-compatible features, and batch assembly. P3 provides the shared query
decoder, its objective, and a bounded CPU profile. Persistent training and
artifact export await their implementation phases.

See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for phase status and
[AGENTS.md](AGENTS.md) for development contracts.

## Scientific pilot

The first scientific pilot is a rotated surface-code $Z$-memory experiment with

$$
d=5,\qquad T=12d=60,\qquad c=b_L=b_R=5,\qquad W=15,\qquad p=0.003.
$$

The four circuit-noise probabilities specified in the protocol are equal to $p$.
Each complete trajectory has 12 disjoint core targets. Retain the 256-dimensional
representation, four GRU layers, and the distance-5 spatial aggregator. Initialize
the entire query-conditioned model randomly; distance-3 pretraining is not a
prerequisite. Fixed-budget main comparisons at $d=5,7$ follow the pilot's measured
runtime and failure/discordance rates.

One offline checkpoint defines G (global/full context), F (core/full context/XOR),
and S (core/window context/XOR). S+, W, and D subsequently fork the same checkpoint
with matched additional trajectories and optimizer updates. The complete
definition is in [02_ablation_protocol.md](docs/02_ablation_protocol.md).

## Documentation

| File | Contents |
|---|---|
| [01_theory_and_decoder.md](docs/01_theory_and_decoder.md) | Logical contributions, exact label construction, window composition, and the Lange decoder. |
| [02_ablation_protocol.md](docs/02_ablation_protocol.md) | Physical task, G/F/S and S+/W/D, statistical estimands, and pilot-to-main budgeting. |
| [03_repository_design.md](docs/03_repository_design.md) | Module interfaces, Hydra, dataset format, dependencies, and planned commands. |
| [04_cluster_workflow.md](docs/04_cluster_workflow.md) | Conda, PBS, CPU profiling, storage visibility, and compressed result transfer. |
| [05_implementation_phases.md](docs/05_implementation_phases.md) | P0–P7 scope, acceptance criteria, and integration checks for later implementation. |
| [conf/README.md](src/nn_window_ablation/conf/README.md) | Configuration groups and unresolved budget fields. |
| [environments/README.md](environments/README.md) | Candidate dependency installation and environment records. |
| [scripts/README.md](scripts/README.md) | PBS template use and scope. |

## Repository organization

| Path | Role |
|---|---|
| `src/nn_window_ablation/` | Future simulation, feature, model, training, evaluation, and analysis modules. |
| `src/nn_window_ablation/conf/` | Hydra groups; the initial experiment is `pilot_d5`. |
| [tests/](tests/README.md) | Shared test setup, CPU checks, and phase test plan. |
| `scripts/pbs/` | Site-adjustable `.pbs.example` files; no scheduler submission logic in scientific modules. |
| `third_party/` | Upstream provenance and future source-integration record. |
| `environment.yml`, `pyproject.toml` | Conda base intent and candidate Python dependencies. |
| `data/`, `runs/`, `exports/` | Untracked default output locations, created later when required. |

The user's workflow is local development, subsequent GitHub publication, cluster
checkout, Conda installation, PBS execution, and compressed artifact transfer
back to the local computer. The supplied compute-node record lists 96 PBS CPU
resources and 350 GB of memory. CPU execution is the initial profile. Neither
physical-core topology nor GPU availability on all nodes is inferred from the
login-node report.

## Validate a configuration

Set up the [Conda environment](environments/README.md), then run the installed
command from the directory where you want relative paths to resolve.

```bash
conda run -n nn_window_decoding nnwa validate runtime=local_cpu
conda run -n nn_window_decoding nnwa validate --multirun \
  experiment=pilot_d5,paired_d7 runtime=local_cpu window.right_per_distance=0.5
```

Validation writes `resolved_config.yaml`, `overrides.yaml`, and `metadata.yaml`
for each accepted configuration. The default run root is `./runs`. Set
`NNWA_RUN_ROOT` or `paths.run_root` to change it. See the
[configuration reference](src/nn_window_ablation/conf/README.md) for run names,
paths, and CPU allocation.

`generate`, `train`, `evaluate`, and `analyze` report the implementation phase
they require. For circuit sampling, see the Python API example and local
evidence. [P1 verification](docs/verification/P1.md)

For buffered observations and graph batching, see the P2 interface and
independent test results. [P2 verification](docs/verification/P2.md)

For the decoder, checkpoint payloads, and local CPU timings, see the P3
verification record and P4 handoff. [P3 verification](docs/verification/P3.md)

## References

1. M. Lange et al., *Modular Neural Decoder for Surface Code Memory and Logic on
   Superconducting Processors*. https://arxiv.org/abs/2610.03869
2. Lange author implementation, pinned source snapshot.
   https://github.com/LangeMoritz/modular-gnn-rnn-decoder/tree/458210ac74ed0047760b8789c169e6b9eff732b8
3. K. Zhang et al., *Learning to Decode in Parallel: Self-Coordinating Neural
   Network for Real-Time Quantum Error Correction*. https://arxiv.org/html/2601.09921v1
4. X. Li et al., *Dynamic Chunk Convolution for Unified Streaming and Non-Streaming
   Conformer ASR*. https://arxiv.org/abs/2304.09325
5. Hydra 1.3, Defaults List. https://hydra.cc/docs/1.3/advanced/defaults_list/
