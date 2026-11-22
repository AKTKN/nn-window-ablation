# Environment specification

## Candidate installation

The cluster workflow uses Conda for the Python base environment and pip for the
selected scientific packages. `environment.yml` declares Python 3.11 and pip.
`pyproject.toml` is the single source of the candidate scientific dependency
versions under its `experiment` extra. The upstream Lange requirements provide
the starting versions [1]. The [implementation status](../IMPLEMENTATION_STATUS.md)
records the local dependency imports and test environment. Compute-node
validation remains pending.

Use these commands in the cloned checkout to set up the `nn_window_decoding`
development environment.

```bash
conda env create -f environment.yml
conda run -n nn_window_decoding python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
conda run -n nn_window_decoding python -m pip install -e '.[experiment,dev]'
```

Bootstrap the CPU wheel before installing the extra so the declared PyTorch
dependency uses the intended CPU build. The official PyTorch version guide
provides that CPU index [2]. TorchVision and TorchAudio are not required by this
study. PyG's minimal installation is the starting path; add compiled extensions
only if the chosen operators require them [3]. There is no ASR package dependency.

Conda recommends installing Conda packages first and pip packages afterward [4].
The candidate does not constitute a lock. Validate imports, the selected graph
operators, and later one genuine forward/backward step in the compute allocation.
The login node's environment alone is insufficient evidence of compute-node
compatibility. If local development uses a different platform, retain separate
platform-resolved environment records while keeping scientific settings fixed.

## Recreate the locally verified Linux environment

The [Conda explicit file](local-linux-64.conda.txt) records the local Linux
x86-64 builds. The [pip inventory](local-linux-64.pip.txt) pins the added
packages, including the CPU PyTorch build.

```bash
conda create -n nn_window_decoding --file environments/local-linux-64.conda.txt
conda activate nn_window_decoding
python -m pip install --constraint environments/local-linux-64.pip.txt \
  'torch==2.7.0+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps -r environments/local-linux-64.pip.txt
python -m pip install --no-build-isolation --constraint environments/local-linux-64.pip.txt -e '.[experiment,dev]'
python -m pip check
```

This inventory describes P0's local verification. Record a distinct inventory
after validating on an allocated compute node. See the
[P0 evidence record](../docs/verification/P0.md).

## Environment records after validation

Save the resolved configuration and the actual Conda and pip inventories inside
each run's provenance directory. Candidate commands after creating that directory
are:

```bash
conda env export -n nn_window_decoding --from-history > environment.intent.yml
conda env export -n nn_window_decoding > environment.resolved.yml
conda list -n nn_window_decoding --explicit > conda.explicit.txt
conda run -n nn_window_decoding python -m pip freeze > pip.freeze.txt
```

The history export captures user-requested Conda specifications. The explicit
Conda record is platform-specific and does not by itself reproduce pip-installed
packages; retain the pip inventory, CPU wheel index, Python/PyTorch build details,
and source revision with it. Record compute-node OS, CPU topology, allocated CPU
resources, effective threads/workers, and dtype. Do not claim cross-platform
bitwise identity from package versions alone [5].

For PBS, use the absolute Conda executable and environment prefix with
`conda run --no-capture-output -p ...`; no interactive shell activation is assumed
[6]. Cluster-specific paths belong in reviewed job files or explicit runtime
configuration, not in the portable environment specification.

## References

1. Lange author requirements at the selected snapshot.
   https://github.com/LangeMoritz/modular-gnn-rnn-decoder/blob/458210ac74ed0047760b8789c169e6b9eff732b8/requirements.txt
2. PyTorch, previous versions. https://pytorch.org/get-started/previous-versions/
3. PyTorch Geometric, installation.
   https://pytorch-geometric.readthedocs.io/en/2.6.1/install/installation.html
4. Conda, managing environments.
   https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html
5. PyTorch, reproducibility. https://docs.pytorch.org/docs/2.7/notes/randomness.html
6. Conda, run. https://docs.conda.io/projects/conda/en/stable/commands/run.html
