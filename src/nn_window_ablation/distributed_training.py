"""Equal-batch CPU DDP with the same global update plan as inline training."""

from datetime import timedelta
import gc
from time import monotonic, sleep

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .artifacts import ArtifactError, canonical_json
from .continuation import model_tensors_sha256
from .streams import ChunkCursor
from .training import DISTRIBUTED_STATE_SCHEMA, Trainer, distributed_state
from .timing import span


class TrainingGroup:
    def __init__(self, rank, placement, rendezvous, peer_receipt, timeout_seconds, launcher_pid, launched):
        self.rank = rank
        self.placement = placement
        self.world_size = len(placement)
        self.rendezvous = rendezvous
        self.peer_receipt = peer_receipt
        self.timeout = timedelta(seconds=timeout_seconds)
        self.launcher_pid = launcher_pid
        self.launched = launched
        self.active = False

    def __enter__(self):
        if not dist.is_gloo_available():
            raise ArtifactError("installed PyTorch lacks the Gloo backend")
        dist.init_process_group(
            "gloo", init_method=self.rendezvous, rank=self.rank, world_size=self.world_size, timeout=self.timeout,
        )
        self.active = True
        return self

    def __exit__(self, *_exc):
        self.close()

    def close(self):
        if self.active:
            dist.destroy_process_group()
            self.active = False

    @property
    def owner(self):
        return self.rank == 0

    def share(self, value):
        record = [value]
        dist.broadcast_object_list(record, src=0)
        return record[0]

    def collect(self, value):
        items = [None] * self.world_size if self.owner else None
        dist.gather_object(value, items, dst=0)
        return items

    def barrier(self):
        dist.barrier()

    def await_peer_exits(self):
        """Wait after Gloo teardown for the supervisor to reap all peer ranks."""
        deadline = monotonic() + self.timeout.total_seconds()
        while not self.peer_receipt.is_file():
            if monotonic() >= deadline:
                raise ArtifactError("DDP peers did not exit by the completion deadline")
            sleep(0.02)

    def mean_loss(self, value):
        result = torch.tensor(value, dtype=torch.float64)
        dist.all_reduce(result)
        return result.item() / self.world_size


class DistributedTrainer(Trainer):
    def __init__(self, *args, group: TrainingGroup, **kwargs):
        super().__init__(*args, **kwargs)
        self.group = group
        self.local_batch, remainder = divmod(self.budget.batch_size, group.world_size)
        if self.stream.dataset is None or remainder or self.local_batch < self.budget.microbatch_size:
            raise ValueError("DDP requires stored data and equal local batches with the same accumulation schedule")
        self.ddp = None
        self._distributed_checkpoint = None
        self._gathered_update = None

    def attach(self):
        # The reducer's used map includes hooks from the preceding no_sync
        # backwards. Globally unused parameters keep a None gradient.
        self.ddp = DistributedDataParallel(
            self.model, device_ids=None, find_unused_parameters=True, static_graph=False,
        )

    def detach(self):
        """Release reducer references before destroying the default process group."""
        self.ddp = None
        gc.collect()

    def step(self, objective):
        self._begin_step()
        count = self.budget.batch_size
        with span("global_plan"):
            identities = self.stream.planned_identities(count)
            indices, buffers = self._draw_queries()
            start = self.stream.position
        offset = self.group.rank * self.local_batch
        rank_rows = slice(offset, offset + self.local_batch)
        chunk_rows = self.stream.plan.chunk_rows
        with span("source_read_verify_unpack"):
            self.stream.seek(ChunkCursor(*divmod(start + offset, chunk_rows)))
            local = self.stream.take(self.local_batch)
            if local.identities != identities[rank_rows]:
                raise ArtifactError("DDP local rows disagree with the ordered global batch")
            self.stream.seek(ChunkCursor(*divmod(start + count, chunk_rows)))
        loss = self._optimize(
            objective, self.ddp, local.shots, indices[rank_rows], buffers, suppress_sync=self.ddp.no_sync,
        )
        with span("update_commit_collective"):
            loss = self.group.mean_loss(loss)
        return self._finish_step(loss, identities, indices, buffers)

    def topology(self):
        return {
            "schema_version": DISTRIBUTED_STATE_SCHEMA, "backend": "gloo", "partition": "equal_contiguous_v1",
            "world_size": self.group.world_size, "local_batch": self.local_batch,
            "local_microbatch": self.budget.microbatch_size, "torch_intraop": torch.get_num_threads(),
            "torch_interop": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
            "mkldnn_deterministic": torch.backends.mkldnn.deterministic,
        }

    def verify_bootstrap(self, source, state_digest):
        local = canonical_json(self.contract()["source"])
        if local != canonical_json(source) or model_tensors_sha256(self.model.state_dict()) != state_digest:
            raise ArtifactError("DDP rank has incompatible data or initial/restored model state")

    def gather_checkpoint(self):
        """Collect rank-local RNGs at a completed update boundary."""
        identity = (self.phase_updates, self.input_audit["sha256"], model_tensors_sha256(self.model.state_dict()))
        records = self.group.collect({"identity": identity, "rng": self.randomness.model_state()})
        if self.group.owner:
            if any(record["identity"] != identity for record in records):
                raise ArtifactError("DDP ranks disagree on checkpoint update, global inputs, or model weights")
            self._distributed_checkpoint = {
                **self.topology(), "rank_rng": [record["rng"] for record in records],
            }
            self._gathered_update = self.phase_updates

    def snapshot(self):
        if self._gathered_update != self.phase_updates or not self.group.owner:
            raise ArtifactError("DDP checkpoint needs a synchronized rank-0 snapshot")
        return {**super().snapshot(), "distributed_state": self._distributed_checkpoint}

    def restore(self, payload):
        self._usable = False
        record = distributed_state(payload)
        if record is None or {key: value for key, value in record.items() if key != "rank_rng"} != self.topology():
            raise ArtifactError("DDP checkpoint requires the same world size and numerical replay contract")
        super().restore({key: value for key, value in payload.items() if key != "distributed_state"})
        try:
            self.randomness.restore_model(record["rank_rng"][self.group.rank])
        except (ValueError, TypeError, RuntimeError) as exc:
            self._usable = False
            raise ArtifactError("invalid DDP rank-local model RNG state") from exc
        self._distributed_checkpoint = record
        self._gathered_update = self.phase_updates
