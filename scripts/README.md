# PBS templates

`pbs/probe.pbs.example` describes a short compute-node environment probe.
`pbs/pilot.pbs.example` describes the future distance-5 training launch. These are
templates only; they were not submitted. The pilot requires the implemented CLI,
the scientific phases, and explicitly chosen pilot budgets.

The supplied node record lists 96 PBS CPU resources and 350 GB of memory. The
pilot template's corresponding `select` request is a candidate for the site to
validate, not a reservation. The probe uses a smaller candidate allocation. Edit
queue/account, resources, and walltime according to site policy; no queue or host
name is assumed. PBS directives are not shell-expanded variable assignments [1].

Copy a reviewed example to a run-specific `.pbs` file and supply its explicit
`NNWA_*` variables using edited assignments or the site's `qsub -v` mechanism.
The generated `.pbs` file is ignored by Git because it can contain local paths.
Do not rely on an interactive Conda activation being inherited. The examples use
an explicit Conda executable and environment prefix [2].

| Variable | Meaning |
|---|---|
| `NNWA_REPO_ROOT` | Absolute, fixed source checkout accessible from the compute node. |
| `NNWA_CONDA_EXE` | Absolute path to the Conda executable. |
| `NNWA_CONDA_PREFIX` | Absolute environment prefix accessible from the compute node. |
| `NNWA_DATA_ROOT`, `NNWA_RUN_ROOT` | Verified storage roots. |
| `NNWA_EXPECTED_COMMIT` | Source commit selected for the job. |
| `NNWA_THREADS` | Chosen intra-operation thread count, bounded by the allocation. |
| `NNWA_ALLOCATED_CPUS` | CPU allocation requested for this reviewed job. |
| `NNWA_MAX_STEPS`, `NNWA_BATCH_SIZE`, `NNWA_LEARNING_RATE` | Explicit pilot optimization settings. |
| `NNWA_OPTIMIZER`, `NNWA_SCHEDULER` | Pilot-selected algorithm and schedule; use an explicit disabled-scheduler value when supported. |
| `NNWA_RUN_NAME` | Portable identifier for the run. |

The application must also record effective resources and check CPU/thread/worker
settings at startup. A 96-CPU allocation does not imply that 96 threads or 96
training processes improve throughput. Begin with one process and zero
data-loader workers, then measure the declared thread candidates [3].

Hydra's basic multirun launcher runs configurations locally in the allocated
environment; it is not a PBS launcher. Initially submit a reviewed single-job
template per independent training job, and pass checkpoint dependencies explicitly
[4]. Campaign and export automation is P7.

Before substantive computation, verify repository, environment, and data path
visibility from an allocated node. `PBS_O_WORKDIR` identifies the submission
directory but does not prove that the path is shared [1]. The result workflow is
completed artifacts plus manifest, checksums, selected paired predictions/labels,
and optional checkpoint; compress on the cluster and transfer with `scp`.

## References

1. OpenPBS, qsub manual source.
   https://github.com/openpbs/openpbs/blob/master/doc/man1/qsub.1B
2. Conda, run. https://docs.conda.io/projects/conda/en/stable/commands/run.html
3. PyTorch, CPU oversubscription.
   https://docs.pytorch.org/docs/2.7/notes/multiprocessing.html#cpu-in-multiprocessing
4. Hydra 1.3, multirun.
   https://hydra.cc/docs/1.3/tutorials/basic/running_your_app/multi-run/
