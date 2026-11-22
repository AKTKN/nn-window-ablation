# CPU cluster workflow for nn-window-ablation

## 1. Available information and execution scope

The repository name is `nn-window-ablation`. The initial pilot uses the rotated surface-code memory experiment at $d=5$, $T=12d=60$, $c=d=5$, and default buffers $b_L=b_R=d$. It starts from random initialization without a prerequisite $d=3$ model and retains the planned 256-dimensional, four-layer GRU. The scientific protocol, architecture, and noise model are unchanged by the choice of execution hardware.

The supplied login-node information identifies Rocky Linux 8.8, 20 CPU cores, approximately 187 GiB of memory, and no available `nvidia-smi` command. The supplied compute-node record advertises `resources_available.ncpus=96`, `resources_available.mem=350gb`, and a `job-busy` state. No GPU resource appears in that record. These observations support a CPU-first workflow; they do not establish that every cluster node lacks a GPU or that the listed compute node is currently available. PBS commands are located under `/opt/pbs/bin`.

This document specifies the planned cluster workflow. P3 has a separate
local CPU profile. Compute-node environment validation, timing, and scheduler
execution remain pending.

## 2. Local development and cluster checkout

Develop and review the source locally, then publish an approved version to GitHub. On the cluster, clone the repository and check out the exact commit selected for the experiment. Record the commit in every run. Keep the checkout unchanged while its jobs are queued or running, because batch jobs can start after submission and read the files then present on disk.

The repository uses `environment.yml` for the Conda base environment, PBS templates under `scripts/pbs/*.pbs.example`, and `src/nn_window_ablation/conf/runtime/cpu.yaml` for the CPU profile. P0 provides `nnwa validate` for inspecting the resolved settings. Commands that run scientific workflows become available in later phases. Large datasets and results belong under explicitly configured storage roots.

Use the login node for editing, environment preparation where site policy permits, submission, and result transfer. Perform the $d=5$ execution checks and training benchmark in an approved compute allocation. The login node's CPU count is not the resource budget for a scheduled job.

## 3. Conda environment and CPU dependencies

Use a dedicated Conda environment whose base specification contains Python 3.11 and pip. The proposed first dependency candidate bootstraps PyTorch 2.7.0 from the official CPU wheel index before installing the editable project and its selected PyG, Stim, PyMatching, NumPy, SciPy, Hydra/OmegaConf, and plotting dependencies. This is an upstream-derived candidate, not a validated cluster environment. PyTorch's official instructions provide Linux/Pip/CPU installation and version-specific CPU wheel commands. No CUDA toolkit, GPU driver package, MPI runtime, or ASR package is needed for this initial execution path. [1, 2]

The environment specification is initially a declaration to resolve, not a verified lock; the scaffold supplies no `uv.lock`. The upstream decoder's requirements provide a starting candidate, but their compatibility with the cluster must be checked. In particular, inspect the compute node's operating system, CPU architecture, available instruction set, and library compatibility before accepting an environment prepared on the login node. Validate imports, one generated batch, one model forward/backward step, and checkpoint reload on the allocated node.

Use Conda to establish the isolated base environment before installing any remaining pip dependencies. Keep the Conda and pip specifications together and record the final resolved package inventories. Conda documents this installation order and recommends recreating an environment when later changes would mix the package managers inconsistently. Record successful resolution and execution checks before accepting the environment specification for experiments. [3]

For a noninteractive PBS job, use an explicit Conda executable and environment prefix with `conda run --no-capture-output`. This avoids depending on interactive shell initialization and streams the program's output to the batch log. An alternative is to source the installation's `etc/profile.d/conda.sh` and activate the environment within the script. An environment activated in a login shell should not be assumed to be active in a later batch shell. [3, 4]

## 4. Single-process execution and CPU threading

Start with one Python process and multithreaded PyTorch operators. Set data-loader workers to zero and inter-operation threads to one. Benchmark intra-operation thread counts

$$
n_{\mathrm{thread}}\in\{1,8,16,32,48,96\},
$$

subject to the approved CPU allocation. These are thread counts for one training process, not a request for 96 data-loader workers or 96 copies of the model. The scheduler's `ncpus` value describes allocated CPU resources; inspect the actual topology and CPU affinity before interpreting it as a count of physical cores.

Set `OMP_NUM_THREADS` and `MKL_NUM_THREADS` consistently, and have the application's runtime initialization call `torch.set_num_threads` before model execution. PyTorch documents that `MKL_NUM_THREADS` takes precedence over `OMP_NUM_THREADS`, and that intra-operation thread settings should be applied before eager/JIT/autograd work. Apply the inter-operation setting once at process startup. [5–7]

Benchmark identical $d=5$ workloads, model parameters, batch sizes, and update counts. Record warm-up separately, then measure trajectories per second, optimizer-step time, peak resident memory, and total wall time. Select the thread count by measured throughput; a larger thread pool is not assumed to be faster for small graphs. Keep generation and graph-construction time visible in the measurements. Parallel data preparation or concurrent independent jobs can be considered after this benchmark, with their total threads bounded by the allocation. PyTorch's multiprocessing guidance describes the oversubscription that otherwise results. [8]

The P3 profiling template invokes the installed profiling module once per
fresh process. Supply the declared allocation and chosen intra-operation
thread count explicitly. It runs the bounded full-architecture workload
and writes timings and effective runtime settings to a new JSON file.
[P3 PBS template](../scripts/pbs/profile.pbs.example)

## 5. PBS resource request and template

PBS resource requests use `select` chunks. A candidate request reflecting the supplied node record is `select=1:ncpus=96:mem=350gb`. It is a site-adjustable request, not a guarantee that the queue accepts that memory limit or that all resources are presently free. Queue, account, walltime, placement, and allocation limits must follow the site's actual configuration; no queue name is assumed here. After benchmarking, request the resources needed by the chosen workload. [9, 10]

The following illustrative script assumes that the submission directory, environment, data, and output paths have already been verified as accessible from the compute node. The placeholder paths and proposed CLI runtime fields require implementation and local substitution. The one-hour walltime is a placeholder for a short pilot allocation.

```bash
#!/bin/bash
#PBS -N nnwa_d5
#PBS -l select=1:ncpus=96:mem=350gb
#PBS -l walltime=01:00:00
#PBS -j oe

set -euo pipefail
cd "${PBS_O_WORKDIR:?Missing PBS submission directory}"

NNWA_CONDA=/path/to/miniconda3/bin/conda
NNWA_ENV_PREFIX=/path/to/envs/nnwa_cpu
NNWA_THREADS=16
export OMP_NUM_THREADS="$NNWA_THREADS"
export MKL_NUM_THREADS="$NNWA_THREADS"

"$NNWA_CONDA" run --no-capture-output -p "$NNWA_ENV_PREFIX" \
  nnwa train experiment=pilot_d5 runtime=cpu \
  runtime.intraop_threads="$NNWA_THREADS" \
  runtime.interop_threads=1 data.num_workers=0 seed=101
```

The intended runtime profile explicitly selects `runtime.device=cpu`. The training budget fields remain unresolved in the scaffold; actual launch requires explicit pilot update, batch, optimizer, and learning-rate choices. The more complete `scripts/pbs/pilot.pbs.example` requires those choices as inputs. After CLI implementation, a reviewed `.pbs.example` file can be copied to a run-specific `.pbs` file, its placeholders replaced, and submitted using `/opt/pbs/bin/qsub`. Status and allocation details are then inspected through the site's PBS commands. Merely saving the template submits no job.

PBS defines `PBS_O_WORKDIR` as the submission working directory; it does not establish that the directory is mounted on every execution host. `PBS_JOBID` identifies the scheduler job. Record it as metadata, and use an independently assigned portable run name such as `pilot_d5_cpu16_seed101` for result paths. Avoid colons, array brackets, or unsanitized job identifiers in archive names. The template leaves log naming to PBS and joins stderr into stdout rather than assuming shell-variable expansion in `#PBS` output directives. [9, 11]

Hydra configuration sweeps and PBS submission are separate operations: the initial design uses reviewed single-job PBS templates around the future CLI. Hydra's basic launcher does not allocate cluster resources; see the repository design document for this interface. [12]

## 6. Storage visibility and result transfer

The supplied filesystem command used a narrow filter and returned no matching network filesystem. This does not establish that home, project, or scratch paths are local or inaccessible from compute nodes. Before a real run, inspect the intended paths with `findmnt -T` or `df -T`, then verify visibility from inside an approved compute allocation. A small marker file and its checksum can establish whether the same path resolves to the same content; verify output write access there as well. These checks are proposed for the user's cluster session, not remotely performed here.

If the paths are shared, run from the stable checkout and write outputs to the designated shared result directory. If the site requires node-local scratch, explicitly stage the checkout and required immutable data into the allocation and copy selected outputs back before the job ends. Discover scratch location and retention from site configuration; do not assume that a particular temporary-directory variable or path exists.

After execution, assemble one export directory for the selected run or comparison. Include resolved configurations, source/environment provenance, completion status, logs, metric tables, figures, paired prediction arrays, matching global/core label arrays or their identified dataset subset, and manifests with checksums. Include the selected checkpoint when local inference or continuation is required. The full immutable test dataset is optional when local analysis already has the corresponding labels and identities; raw detector records are needed if local re-decoding is intended.

Compress that explicit export directory, transfer the archive to the local machine with `scp`, and verify its checksum after transfer. Do not copy generated training streams, graph caches, or all intermediate checkpoints by default. Preserve the same shot identities across exported predictions and labels so that local paired analysis reproduces the cluster result.

## References

1. PyTorch, [Start Locally: Linux CPU installation](https://docs.pytorch.org/get-started/locally/).
2. PyTorch, [Previous versions and CPU wheel commands](https://pytorch.org/get-started/previous-versions/).
3. Conda, [Managing environments](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html).
4. Conda, [conda run](https://docs.conda.io/projects/conda/en/stable/commands/run.html).
5. PyTorch, [Threading environment variables](https://docs.pytorch.org/docs/stable/threading_environment_variables.html).
6. PyTorch, [torch.set_num_threads](https://docs.pytorch.org/docs/2.7/generated/torch.set_num_threads.html).
7. PyTorch, [torch.set_num_interop_threads](https://docs.pytorch.org/docs/2.7/generated/torch.set_num_interop_threads.html).
8. PyTorch, [Multiprocessing best practices: CPU oversubscription](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html).
9. OpenPBS, [qsub manual source](https://github.com/openpbs/openpbs/blob/master/doc/man1/qsub.1B).
10. OpenPBS, [Resource specification manual source](https://github.com/openpbs/openpbs/blob/master/doc/man1/pbs_resources.7B).
11. Altair, [PBS Professional 2024.1 User's Guide](https://help.altair.com/2024.1.0/PBS%20Professional/PBSUserGuide2024.1.pdf), sections on job input/output and execution environment.
12. Hydra 1.3, [Multi-run](https://hydra.cc/docs/1.3/tutorials/basic/running_your_app/multi-run/).
