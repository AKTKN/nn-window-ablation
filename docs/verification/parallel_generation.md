# Phase 4 parallel generation

## Work identity and publication

`nnwa generate` uses one task per immutable storage shard. It can run
inline or use the same bounded Linux spawn scheduler as evaluation. The
default is inline generation. In a pool set `data.sampling_processes=N`
and explicitly allocate at least `N` logical CPUs. Use `runtime=local_cpu`
or set both runtime thread limits to one. Each worker runs Stim with a
single native thread on one CPU. The coordinator shares one worker CPU
for receipt checks when there is no spare CPU. The allocator uses
observed affinity and any visible cgroup-v2 bandwidth limit.
[Python 3.11 process-pool lifecycle](https://docs.python.org/3.11/library/concurrent.futures.html)

The descriptor identifies the dataset plan, shard index, fixed interval,
existing circuit seed and attempt. The coordinator derives its output
path from the index under the owned attempt root. Workers reconstruct the
circuit once each. Each shard compiles a fresh Stim sampler with
`namespace.seed("circuit", shard_index)`. It keeps the existing single
sampling call with that shard's exact count and unchanged contiguous IDs,
including a short final shard. The compiled sampler does not carry state
between shards. Bitwise comparisons apply to fixed Stim version, SIMD
backend and sampling call partition.
[Stim 1.15 sampling contract](https://github.com/quantumlib/Stim/blob/v1.15.0/src/stim/circuit/circuit.pybind.cc)

The worker validates its arrays and writes four packed-format payloads
and one shard manifest. It returns the task identity, staged manifest
size and checksum, worker PID and start time, with all per-shard arrays
released. Bytes, timing and resource observations appear in JSONL. No
model or Torch runtime loads in a sampler. The coordinator checks each
result against its own task and staged directory, checks exact filenames,
opens and verifies the stored payloads, and validates IDs, shapes, padding
and label parity. Workers share the same data-schema-1 format with the
existing lazy reader. The coordinator writes physical metadata once.

An interprocess lock guards each destination. Workers stage into the
sibling `.<data-id>.generation/dataset/shards/<index>/` during an owned
attempt. The coordinator writes the final manifest after exact coverage,
orders its inventory by index and publishes the whole directory by a
durable same-filesystem rename. Reader and archive commands can verify the
sealed manifest with their existing data schema. The worker's unique
file-write staging is inside the owned root. A new invocation refuses an
existing destination. It discards an incomplete staged attempt only after
taking the destination lock and checking its writer lease. A partially
written dataset has no complete destination. Failed attempts restart
from the agreed seeds, so a retry reproduces their scientific payloads.
There is no per-shard generation resume.

A worker error or exit stops submissions and triggers bounded pool
shutdown. Linux parent-death signals and the writer lease keep a new
coordinator from deleting staging while a surviving worker can still
write. If failure occurs after the final rename, the visible complete
dataset is preserved. Attempt logs are kept under
`.generation-attempts/<data-id>/<attempt>/` alongside the dataset root.

## Resource and identity checks

`data.sampling_max_in_flight` limits submitted but unverified tasks,
defaulting to twice the worker count. Each worker keeps one raw shard,
bounded packing temporaries and its one immutable circuit cache. The
coordinator maps at most four payload files of one shard at a time.
The bounded queue carries descriptors and compact receipts. The only
whole-dataset state in coordinator memory is the lightweight inventory
needed for the final manifest. Events cover circuit preparation, sampler
compile, sampling, label and ID construction, validation, packing,
writes/checksums, receipt verification, queue wait, pool shutdown and
publication. Per-process observations distinguish CPU time, RSS, native
threads, affinity, descriptors and I/O. Task time can overlap across
workers.

The content projection compares schema, completion status, dataset ID,
sampling recipe, physical receipts and canonical shard receipts.
Execution provenance contains settings including worker count, resolved
overrides, the Stim backend and attempt identity. Dataset manifest files
can therefore differ while the complete scientific projection and every
physical and shard artifact match. A given training or evaluation run
still binds its own complete dataset manifest digest.

The genuine distance-five fixture verified byte-for-byte equality with
one, two and four workers, including 17 rows in nine shards with a short
tail. Controlled barriers exercised out-of-order replenishment, and
process tests compared the retry after killing a writer or coordinator
during a partial write. Tests also checked one-shard and undersubscribed
layouts, worker circuit reuse, zero-event shots, output paths, receipt
and payload integrity, immutable publication, and PBS artifact readback.
[Saved-data commands](../GETTING_STARTED.md#run-all-four-saved-data-commands-on-four-cpus)

## Executed local validation

The working tree started from
`ae4aae6d1d82fc20e49eadb0de400bfd73d18b2e`. Tests ran in Conda
`nn_window_decoding` with Python 3.11.18, Stim 1.15.0 and the SSE2 backend.
All numerical environment thread limits were one and the suite used
affinity CPUs 0–7. The focused generation process suite passed six
tests in 20.77 s. The focused ordinary generation, reader, runtime and
PBS checks passed 40 tests in 11.34 s. The cumulative ordinary suite
passed 322 tests in 100.72 s. The cumulative resource suite passed 32 in
270.92 s, including the added split/namespace case, the process monitor
check and genuine distance-5 interrupted DDP and evaluation checks.
Matplotlib emitted PyParsing deprecation warnings.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
taskset -c 0-7 conda run --no-capture-output -n nn_window_decoding \
  python -m pytest -n 8 --dist=load --strict-markers --strict-config \
  -m 'not resource_process and not performance and not cluster' tests
taskset -c 0-7 conda run --no-capture-output -n nn_window_decoding \
  python -m pytest -n 1 --dist=load --strict-markers --strict-config -m resource_process tests
```

The installed CLI probe produced an uneven nine-row distance-five dataset
with four workers. Its runtime receipts and artifacts are under the local
workspace sibling `../nnwa-local-evidence/phase4-generation/`. The complete
pipeline and controlled source checks are recorded in the
[local multicore audit](local_multicore_audit.md).
