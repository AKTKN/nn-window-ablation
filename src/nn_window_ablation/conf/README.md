# Configuration examples

These YAML files specify the Hydra 1.3 interface. The shared tests compose the
test CPU profile. P0 will implement application composition, typed validation,
and the `nnwa` entry point. Root `_self_` precedes `experiment`, and experiment
files use `# @package _global_`, so an experiment overrides the composed groups [1].

| Group | Initial choice | Purpose |
|---|---|---|
| `code` | `surface_rotated_z` | Distance and rounds-per-distance. |
| `noise` | `uniform_circuit` | Four specified circuit-noise probabilities equal to $p$. |
| `window` | `fixed_d` | Fixed core partition and deployment buffer ratios. |
| `model` | `query_gnn_rnn` | Width 256, four GRU layers, common scalar query head. |
| `training` | `full_closed` | Full-context objective; continuation groups are separate. |
| `data` | `paired` | Shot identity, storage, split, and data-loader settings. |
| `evaluation` | `G` | G, F, or S inference from a recorded checkpoint. |
| `analysis` | `primary` | Whole-shot LER and paired contrasts. |
| `runtime` | `cpu` | CPU threads/processes independent of scientific conditions. |
| `experiment` | `pilot_d5` | Initial distance-5 pilot; `paired_d5/paired_d7` are later presets. |

The future typed resolver computes $T=12d$, $c=d$,
$b_L=\lceil d\,\mathtt{left\_per\_distance}\rceil$, and the analogous $b_R$.
It saves both ratios and resolved integers. OmegaConf interpolations are used for
field references and environment variables, not unimplemented arithmetic
expressions [2]. At the default pilot these lengths are $T=60,c=b_L=b_R=5$.

`training.initialization=random` applies to the initial full-context model.
Continuation groups require a full study checkpoint through
`checkpoint.init_from`; no model layer is reinitialized. `checkpoint.path`
selects a model for evaluation, and `checkpoint.resume_from` denotes continuation
of the same interrupted run, including optimizer and data-stream state.

`training.max_steps`, `training.batch_size`, `training.learning_rate`, optimizer
and scheduler choices, and data sizes remain null until the pilot choices are
made. Null means unresolved, not zero, unlimited, or permission to choose a hidden
default. P0 validation can inspect an incomplete design but must reject an
execution requiring an unresolved value. Later implementations must document
explicit choices such as a deliberately disabled scheduler.

The CPU profile's 16 intra-operation threads are an initial benchmark candidate,
not an established optimum. Its candidate sweep is 1, 8, 16, 32, 48, and 96 threads,
bounded by the actual PBS allocation. Start with one Python process, one
inter-operation thread, and zero data-loader workers. `local_cpu` starts with one
thread. `test_cpu` declares one application process, one intra-operation thread,
one inter-operation thread, and an eight-CPU allocation cap for local tests.
Thread settings are independent of distance and model dimensions.

Set `NNWA_DATA_ROOT` and `NNWA_RUN_ROOT` for cluster storage. Their fallback paths
are `./data` and `./runs`; P0 must resolve them to absolute paths before I/O.
`hydra.job.chdir=false` avoids implicitly relocating relative paths [3]. Hydra
multirun enumerates configurations; its basic launcher does not submit PBS jobs
or infer checkpoint dependencies [4]. Use a fixed checkout while jobs are queued
or running.

The future CLI assigns a UUID-based `run_name` before invoking Hydra when none
is supplied. The YAML marks it mandatory, and both Hydra output directories use
it. Explicit names must be unique for new runs; an interrupted run is resumed
only through the recorded resume contract. Directory creation must not silently
reuse an unrelated run. A multirun receives a unique parent name and Hydra job
numbers distinguish its children.

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

`evaluation.id` selects an inference procedure. Statistical result identity also
includes the checkpoint's immutable training regime: window inference applied
to `full_closed`, `continued_full`, `fixed_window`, and `dynamic_window`
checkpoints yields S, S+, W, and D, respectively. Preserve both identifiers in
prediction manifests; do not relabel a checkpoint by a caller's current
`training` override.

The number of bootstrap replicates and the criterion for using the sparse-count
interval are also unresolved fields. Fix them before the main test analysis;
whole trajectories remain the resampling unit under either policy.

## References

1. Hydra 1.3, Defaults List. https://hydra.cc/docs/1.3/advanced/defaults_list/
2. OmegaConf, usage. https://omegaconf.readthedocs.io/en/2.3_branch/usage.html
3. Hydra 1.3, working directory.
   https://hydra.cc/docs/1.3/tutorials/basic/running_your_app/working_directory/
4. Hydra 1.3, multirun.
   https://hydra.cc/docs/1.3/tutorials/basic/running_your_app/multi-run/
5. Lange et al. https://arxiv.org/abs/2610.03869
