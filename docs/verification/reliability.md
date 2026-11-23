# Reliability verification

## Scope and implementation order

The work started from `e34bc4d3221c81daaed1f37b1c24820eab2b17d2`, the
reviewed P7 baseline. The implementation first separated dataset metadata
from payload ownership, then added periodic training saves, then made
evaluation publish and recover one shard at a time. The commit containing
this record is the ending revision.
The tested 110 Python and YAML files under `src` and `tests` had combined
SHA-256 `61bc82d2d7dbc29a67c5135292296ea0b04a97c1b7de0f468d2e092821c896cc`.
The fingerprint hashes each sorted relative path, a NUL separator, and its
binary SHA-256 digest.

The physical experiment, labels, observation masks, architecture, inference
threshold, objectives, and reported estimands retain their existing
definitions. The tests exercise this composition on small fixtures and the
full distance-5 CPU model. [Theory](../01_theory_and_decoder.md) ·
[Protocol](../02_ablation_protocol.md)

## Dataset ownership

`open_dataset(path)` reads the dataset and shard manifests, validates
descriptor consistency, and verifies required physical metadata. It returns
the physical bundle, recipe, and descriptors with indices, rows, paths,
shapes, dtypes, and integrity records. `open_labels(path)` supports the same
descriptor schema when a portable archive omits detectors and physical
files. Payload bytes are verified when a particular shard is opened.

```python
from pathlib import Path
from nn_window_ablation.data import open_dataset

dataset = open_dataset(Path("data/held"))
with dataset.open_shard(0) as shard:
    batch = shard.read(slice(0, 2))
# batch owns its detector, label, and shot-ID data here.
```

Each dataset reader permits one active shard. The reader owns at most four
read-only mappings for a full shard or three for labels and IDs. The mapping
adapter hashes the same file it maps and checks its NPY header and exact
length. The shard checks padding, exact ordered IDs, and core/global parity.
No mapped views escape. Explicit close is idempotent and the closed handle
rejects reads. A context unwinds mappings on partial opens and exceptions.
Use `open_shard(index)` for payloads that were previously read directly
from `dataset.shards[index]`. That tuple now contains descriptors.

Stored training holds one verified shard across repeated minibatches and
closes it on advance, early exit, or run completion. A cross-shard batch
owns the slices from its constituent shards. The finite source still requires
at least `max_steps * batch_size` matching stored trajectories. The
permutation and cursor semantics remain those of P4. Statistics opens
labels-only shards even when detectors are available. Export and PBS
artifact verification can perform a full sequential validation scan with
`dataset.verify_payloads()`. [NumPy mapping lifecycle](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html) ·
[Python mmap](https://docs.python.org/3.11/library/mmap.html)

## Checkpoint publication and recovery

`checkpoint.save_every_updates` is a positive integer, defaults to 100,
and uses the phase-local optimizer update. Set it to `null` to save only the
initial state, explicit pauses, and terminal state. The existing Python
workflow `train_workflow(config, stop_after=k)` pauses after phase update
`k`, including zero. Repeat the original resolved settings and cadence with
`checkpoint.resume_from`. The fixed-update scientific model-selection policy
is unchanged.

For example, from the same invocation directory:

```bash
cpu=(runtime=local_cpu runtime.allocated_cpus=1)
opts=(training.max_steps=3 training.batch_size=2 training.microbatch_size=1
      training.optimizer=adam training.learning_rate=0.001 training.scheduler=none
      checkpoint.save_every_updates=2)
nnwa generate "${cpu[@]}" data.split=train data.id=training data.shots=7 data.shard_shots=3
nnwa train "${cpu[@]}" "${opts[@]}" run_name=stored data.train_source=stored_shards data.id=training
# Resume the same run if it was interrupted:
nnwa train "${cpu[@]}" "${opts[@]}" run_name=stored data.train_source=stored_shards data.id=training \
  checkpoint.resume_from=runs/stored/checkpoint.pt
```

The single replaceable checkpoint contains one consistent completed-update
state. With an interval of `N`, up to `N-1` completed updates can be replayed
between periodic boundaries. If interruption occurs during publication at
the next boundary, up to `N` completed updates can be replayed along with work
from an in-flight update. With `null`, replay can cover the entire segment
since the last explicit save. Recovery reads the validated visible checkpoint
and its actual cursor. Before replacement the prior checkpoint is preserved.
After replacement the visible checkpoint may already be the complete new
one, even if directory sync or a later operation fails. Training cleans
abandoned checkpoint and run-metadata temporaries under its writer lock.
Terminal validation and final run metadata can be retried without advancing
the optimizer.

Serialized checkpoints keep the original envelope and model schemas, so
compatible historical terminal checkpoints can be used for inference and
initialization. Exact resume requires the original package/checkout,
environment, and resolved settings. Use the original revision and
environment for a checkpoint from before this change.
[Checkpoint replacement](https://docs.python.org/3.11/library/os.html#os.replace) ·
[Directory and file sync](https://docs.python.org/3.11/library/os.html#os.fsync)

## Evaluation transaction and schema

New runs reserve a directory exclusively. An immutable `contract.json`
binds the dataset manifest and checkpoint content digests, checkpoint
provenance and state hash, query definitions, geometry, availability and
recurrent policy, numerical settings, label/feature schemas, and the
source and runtime replay environment. Optional matching records are bound
with their original DEM and library version. The semantic comparison uses
content identities. The resume flag, output paths, input aliases and other
invocation metadata are outside that comparison.

A writer holds the run's advisory lock and processes one shard at a time.
It verifies source payloads and fills all declared prediction modes in
bounded inference batches. It writes and validates output bytes in
`shards/.pending`, then publishes the directory with `receipt.json` as
`shards/<index>`. The publication is a same-filesystem rename after file and
directory sync. The receipt binds the exact source shard inventory,
row count, contract digest, and verified prediction files. Retrying under
the lock discards only the run's reserved staging directory and temporary
contract/final-manifest files. It validates committed receipts, source arrays,
and semantic payload contents before reuse. An incompatible or damaged
commit fails explicitly. Publication errors after the rename preserve the
committed directory.

Repeat the original evaluation with `evaluation.resume=true`:

```bash
nnwa evaluate runtime=local_cpu runtime.allocated_cpus=1 evaluation=paired \
  run_name=paired data.id=held checkpoint.path=runs/stored/checkpoint.pt evaluation.resume=true
```

Prediction schema 2 adds the semantic contract and its digest plus each
source shard's integrity record to the final `manifest.json`. The manifest
is written once after all required shards validate and carries enough
information for analysis and archive reanalysis without live resume
receipts. A matching completed run verifies as a no-op. Analysis/export
still read completed schema-1 prediction manifests and existing bundles.
Schema-1 predictions are read-only results for this purpose. Use a fresh
evaluation run name to obtain a schema-2 resume contract. PBS verification
opens source shards sequentially and verifies each live receipt against the
final manifest for schema 2. The PBS evaluate stage can be planned with
`evaluation.resume=true` and a new job receipt directory.

Evaluation holds at most one shard's input mappings and one shard's
prediction working arrays, including validation copies, plus a bounded
inference batch and the frozen model. The additional temporary prediction
space is one shard of output plus the atomic write in progress and small
metadata. Committed result space and file count grow linearly with shards.
There is no per-shot progress file or per-update permanent checkpoint.

## Local test evidence

The test families cover packed-byte and ownership oracles, exact data order,
cadence-independent state, pre/post-publication failures, corruption after
resealing digests, and committed-shard reuse. The composed distance-5
pipeline compares finite stored training through a fresh-process resume,
kills evaluation at a shard barrier, checks exact output values and
independently counted paired failures, and verifies completed results through
analysis, export, reanalysis, and PBS's artifact validator. The matched P6
fixture also uses one stored continuation dataset and terminal validation for
all three arms.

The tests ran in `nn_window_decoding` on Linux WSL2 x86-64 with an Intel Core
i7-13700H. The environment used Python 3.11.18, Torch 2.7.0+cpu, PyG 2.6.1,
NumPy 2.2.6, Stim 1.15.0, SciPy 1.17.1, PyMatching 2.2.2, Hydra 1.3.2,
pytest 8.4.2, and xdist 3.8.0. CPU affinity was 0–7. Numerical workers used
one Torch intra-op and inter-op thread and the suite's `runtime=test_cpu`
settings, with zero data-loader workers and inline sampling. The resource
pool has two workers and at most one active single-threaded child per worker.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
taskset -c 0-7 conda run --no-capture-output -n nn_window_decoding \
  python -m pytest -n 8 --dist=load --strict-config --strict-markers \
  -m 'not resource_process and not performance and not cluster' tests/unit/test_reliability_*.py
taskset -c 0-7 conda run --no-capture-output -n nn_window_decoding \
  python -m pytest -n 2 --dist=load --strict-config --strict-markers -m resource_process tests
taskset -c 0-7 conda run --no-capture-output -n nn_window_decoding \
  python -m pytest -n 8 --dist=load --strict-config --strict-markers \
  -m 'not resource_process and not performance and not cluster' tests
```

The final focused ordinary run passed 41 tests in 31.45 seconds. The isolated
resource run passed four tests in 43.68 seconds. The cumulative ordinary
run passed all 281 tests in 131.27 seconds, including 240 inherited tests.
The final runs had zero failures or skips. The cumulative run reported the
104 third-party Matplotlib/Pyparsing deprecation warnings observed in P7.
The first crash-fixture run
found that analysis raised a raw missing-file error for an absent final
manifest. Analysis now identifies this as a missing artifact. A legacy
fixture also needed to account for the shared evaluation metadata in a
checkpoint record. Both paths passed after correction.

The resource child traversed 96 single-row shards with `RLIMIT_NOFILE=24`.
Its baseline was four descriptors and no array mappings. Full reads peaked
at eight descriptors and four mappings. Labels-only reads peaked at seven
descriptors and three mappings. It returned to baseline after each shard.
The instrumented stream opened each of the four arrays only once per
consumed shard across several minibatches. Repeated failures left only one
committed shard during recovery, and phase checkpoints occupied one
replaceable weight file. The process-termination tests used explicit pipe
barriers at checkpoint replacement and a partially written evaluation shard.

Identical verified observations and IDs support the same ordered training
inputs. Equal replayed random, model, and optimizer state supports the same
terminal model in the recorded environment. Verified shard reuse preserves
coverage and predictions. Those same predictions and labels reproduce the
whole-trajectory statistics.

## Compute-node checks still pending

The local tests exercise filesystem locking, rename, and sync behavior in
this environment. Verify those guarantees on the actual shared filesystem
and measure production throughput and memory in an allocated compute job.
No cluster jobs or large workloads were launched for this change.
