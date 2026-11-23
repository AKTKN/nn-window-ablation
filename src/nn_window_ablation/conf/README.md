# Configuration examples

The installed `nnwa` entry point composes these YAML files with Hydra 1.3.
Root `_self_` precedes `experiment`, and experiment files use
`# @package _global_`, so an experiment overrides the composed groups. Explicit
CLI field overrides apply after the experiment [1].

| Group | Initial choice | Purpose |
|---|---|---|
| `code` | `surface_rotated_z` | Distance and rounds-per-distance. |
| `noise` | `uniform_circuit` | Four specified circuit-noise probabilities equal to $p$. |
| `window` | `fixed_d` | Fixed core partition and deployment buffer ratios. |
| `model` | `query_gnn_rnn` | Width 256, four GRU layers, common scalar query head. |
| `training` | `full_closed` | Base objective and matched S+/W/D continuation presets. |
| `data` | `paired` | Shot identity, storage, split, and data-loader settings. |
| `evaluation` | `G` | Individual G/F/S or paired inference from a terminal checkpoint. |
| `analysis` | `primary` | Whole-shot LER and paired contrasts. Use `adaptation` for S+−W and W−D. |
| `runtime` | `cpu` | CPU threads/processes independent of scientific conditions. |
| `experiment` | `pilot_d5` | Initial distance-5 pilot; `paired_d5/paired_d7` are later presets. |

`config.resolve_config` returns a typed `ResolvedConfig` with the configured
groups and a `geometry` record. The resolver computes
$T=d\,\mathtt{rounds\_per\_distance}$, $c=d$,
$b_L=\lceil d\,\mathtt{left\_per\_distance}\rceil$, and the analogous $b_R$.
The saved configuration includes the input ratios, derived integers, nominal
width, and core count. At the default pilot these are
$T=60,c=b_L=b_R=5,W=15,J=12$. OmegaConf resolves field references and
environment variables before the typed resolver checks the values [2].

`training.initialization=random` applies to the initial full-context model.
Continuation groups require a full study checkpoint through
`checkpoint.init_from`. `checkpoint.path` selects a model for evaluation.
`checkpoint.resume_from` continues the same interrupted run with its
restored optimizer and data stream. `init_from` and
`resume_from` are mutually exclusive.

`training.max_steps`, `training.batch_size`, `training.learning_rate`, optimizer
and scheduler choices, and data sizes remain null until the pilot choices are
made. `validate` accepts unresolved budgets for inspection. `generate` requires
`data.shots` and `data.shard_shots`, and uses `data.split` to select a set of
complete trajectories. `train` accepts `training.optimizer=sgd` or `adam` and
`training.scheduler=none` or `step`. The `step` scheduler requires
`training.scheduler_step_size` in updates and `training.scheduler_gamma`.
`training.batch_size` is the effective trajectory count per update, and
`training.microbatch_size` defaults to that count.

`data.train_source=generated_chunks` requires `data.training_chunk_shots`.
`data.train_source=stored_shards` requires `data.id` for a finalized train
split with a matching circuit, seed, family, replicate, and training phase.
`data.replicate` is an integer namespace component and defaults to zero.

The CPU profile's 16 intra-operation threads are an initial benchmark candidate,
not an established optimum. Its candidate sweep is 1, 8, 16, 32, 48, and 96 threads,
bounded by the actual PBS allocation. Start with one Python process, one
inter-operation thread, and zero data-loader workers. `local_cpu` starts with one
thread. `test_cpu` declares one application process, one intra-operation thread,
one inter-operation thread, and an eight-CPU allocation cap for local tests.
Thread settings are independent of distance and model dimensions.

For numerical startup, provide `runtime.allocated_cpus` explicitly. The CPU
budget charges each application process for the intra-op count times the
inter-op count, plus one CPU for each loader worker and each parallel sampler
worker. A single sampler runs inline. `runtime.configure_cpu_runtime` checks
the declaration and process affinity before applying native and PyTorch thread
settings. `validate` records the plan without starting numerical execution.

Set `NNWA_DATA_ROOT` and `NNWA_RUN_ROOT` or override their `paths.*` fields.
The fallback paths are `./data` and `./runs`. Storage and checkpoint paths are
resolved against the invocation directory before publishing records.
Hydra's working directory stays at the invocation directory [3].

The CLI supplies a UUID-based `run_name` before invoking Hydra when none is
given. An explicit name may contain ASCII letters, digits, dots, underscores,
and hyphens, must begin with a letter or digit, and must be at most 128
characters with no trailing dot. The command reserves a new name exclusively.
To resume training, repeat the original `run_name`, roots, and resolved
settings with `checkpoint.resume_from` pointing to the run's `checkpoint.pt`.
For a continuation originally started with `checkpoint.init_from`, omit
that argument when resuming. The saved checkpoint records the parent.

Hydra's basic launcher composes each sweep child in a temporary directory.
Records are published after the sweep passes validation, with one directory
per child under `<run_root>/<run_name>/hydra_sweep/<job-number>`. A single run
writes to `<run_root>/<run_name>/hydra`. Sweep children share the parent
`run_name` and `paths.run_root`. Place Hydra options such as `--multirun` before
the overrides [4].

`metadata.yaml` records the installed package versions, CPU identity, affinity,
configured allocation and worker counts, and source provenance. For a wheel
whose installer does not record a Git revision, supply `NNWA_SOURCE_REVISION`
from the source revision that built the wheel. A detached wheel without that
information reports the revision as unavailable.

For continuation presets, `training.max_steps` is the number $U_1$ of new updates
in that phase. Record parent, phase-local, and cumulative update counts
separately. Resuming preserves progress toward $U_1$; it neither resets the
optimizer nor starts another $U_1$ updates.

`training.data_stream_phase` is `base` for the offline model and `continuation`
for all three adaptation arms. Derive and record separate trajectory/core-query
streams for the two phases, with the same continuation namespace and ordering
across S+/W/D. Do not key these streams by the arm's training name. Keep the
phase-local chunk cursor in resumable state. This produces new additional draws
without breaking cross-arm pairing.

A `train --multirun training=continued_full,fixed_window,dynamic_window` sweep
checks parent, data, optimizer, budget, and execution settings before creating
jobs. D independently draws the left and right buffers once per effective batch
and uses the pair for every accumulation microbatch. Set
`training.validation_data_id` to run a terminal-only deployment-window diagnostic
on the selected immutable validation split for each arm. The terminal update
determines the checkpoint. [P6 verification](../../../docs/verification/P6.md)

`evaluation.id` selects an inference procedure. Statistical result identity also
includes the checkpoint's immutable training regime: window inference applied
to `full_closed`, `continued_full`, `fixed_window`, and `dynamic_window`
checkpoints yields S, S+, W, and D, respectively. Preserve both identifiers in
prediction manifests; do not relabel a checkpoint by a caller's current
`training` override.

`evaluation=paired` runs G/F/S from one terminal checkpoint on the same held-out
dataset. Use `evaluation.batch_size` to bound trajectories per forward and
`evaluation.matching=true` to add PyMatching. `evaluate` and `analyze` default to
`data.split=test`. Explicit `data.split=validation` selects a validation artifact.

`analyze` uses the same `data.id` and the manifest paths in `analysis.predictions`.
It requires `analysis.bootstrap_replicates`, `analysis.bootstrap_seed`, and
`analysis.sparse_rule`. The sparse rule is an inclusive cutoff on the sum of the
two discordance counts. Choose the main-study values after the pilot and before
inspecting the main test results. For relative ratios, explicitly set
`analysis.ratio_min_denominator_failures`. Ratios use the second condition's LER
as the denominator and are unresolved if that count is below the minimum or any
resampled denominator is zero. Analysis reports one training seed and replicate
per invocation. [P5 verification](../../../docs/verification/P5.md)

`analysis=adaptation` selects the two prescribed S+/W/D contrasts. Supply prediction
manifests for the three arms on the same held-out dataset and deployment buffers.
An adapted contrast requires matching parent, initial model, training controls,
and trajectory/core trace identities.

## References

1. Hydra 1.3, Defaults List. https://hydra.cc/docs/1.3/advanced/defaults_list/
2. OmegaConf, usage. https://omegaconf.readthedocs.io/en/2.3_branch/usage.html
3. Hydra 1.3, working directory.
   https://hydra.cc/docs/1.3/tutorials/basic/running_your_app/working_directory/
4. Hydra 1.3, multirun.
   https://hydra.cc/docs/1.3/tutorials/basic/running_your_app/multi-run/
5. Lange et al. https://arxiv.org/abs/2610.03869
