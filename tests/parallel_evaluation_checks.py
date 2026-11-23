"""Spawn-importable controls for parallel evaluation lifecycle tests."""

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import pickle
from time import monotonic, sleep

from nn_window_ablation import paired_workflows as workflows
from nn_window_ablation.evaluation_pool import append_event


production_initializer = workflows._initialize_evaluation_worker
production_task = workflows._run_worker_task


def await_file(path, *, timeout=35):
    deadline = monotonic() + timeout
    while not path.is_file():
        if monotonic() >= deadline:
            raise TimeoutError(f"test barrier not reached: {path}")
        sleep(0.02)


def actual_model_loads(*args):
    from nn_window_ablation.models.gnn_rnn import QueryDecoder

    log = args[-1] / f"worker-{os.getpid()}.jsonl"
    original = QueryDecoder.from_checkpoint.__func__

    def audited(cls, state):
        append_event(log, "actual_model_load")
        return original(cls, state)

    QueryDecoder.from_checkpoint = classmethod(audited)
    return production_initializer(*args)


class ControlledEvaluator:
    def __init__(self):
        import torch

        self.model = torch.nn.Linear(1, 1).eval()

    def views(self, mode):
        return range(1 if mode == "G" else 12)

    def logits(self, mode, detectors, ids, *, timings):
        import numpy as np

        if os.environ.get("NNWA_TEST_BEHAVIOR") == "mutate":
            self.model.bias.data.add_(1)
        offsets = np.arange(len(self.views(mode)))
        return ((ids[:, None] + offsets + ord(mode)) % 5 - 2).astype(np.float32)


def controlled_initialize(config, semantic, modes, _threads, stage, log_dir):
    from nn_window_ablation.evaluation import model_state_sha256

    _, mapped = workflows._dataset(config)
    evaluator = ControlledEvaluator()
    workflows._worker_runner = workflows._TaskRunner(
        mapped, evaluator, None, modes, config.evaluation.batch_size, model_state_sha256(evaluator.model),
        workflows._semantic_digest(semantic), stage, log_dir / f"worker-{os.getpid()}.jsonl",
    )
    return {}


def controlled_task(task):
    behavior = os.environ.get("NNWA_TEST_BEHAVIOR", "")
    runner = workflows._worker_runner
    barrier = runner.stage.parents[1] / "test-barriers"
    if behavior in ("reorder", "pause") and task.index == 0:
        barrier.mkdir(exist_ok=True)
        original = workflows._write_array

        def blocked_write(path, values):
            integrity = original(path, values)
            if path.name == "shot_ids.npy":
                (barrier / "ready-0").write_text(str(os.getpid()))
                await_file(barrier / "release-0")
            return integrity

        workflows._write_array = blocked_write
        try:
            result = production_task(task)
        finally:
            workflows._write_array = original
    else:
        if behavior in ("reorder", "pause") and task.index == 1:
            await_file(barrier / "ready-0")
        if behavior in ("reorder", "pause") and task.index == 2:
            await_file(runner.stage.parents[1] / "shards/1/receipt.json")
            (barrier / "started-2").touch()
            if behavior == "reorder":
                (barrier / "release-0").touch()
            else:
                await_file(barrier / "release-2")
        if behavior == "write-error" and task.index == 0:
            original = workflows._write_array

            def bad_write(path, values):
                original(path, values)
                raise OSError("injected write failure")

            workflows._write_array = bad_write
            try:
                result = production_task(task)
            finally:
                workflows._write_array = original
        else:
            result = production_task(task)
    # A descriptor or outcome with any ndarray, tensor or model fails JSON
    # serialization here. Keep the pickled result below the atomic pipe budget.
    json.dumps(asdict(task))
    json.dumps(asdict(result))
    append_event(runner.log, "ipc", index=task.index, task_bytes=len(pickle.dumps(task)), result_bytes=len(pickle.dumps(result)))
    if isinstance(result, workflows.EvaluationResult) and task.index == 0:
        if behavior == "wrong-task":
            return replace(result, task=replace(task, index=1))
        if behavior == "wrong-contract":
            return replace(result, task=replace(task, contract_sha256="0" * 64))
        if behavior == "wrong-receipt":
            return replace(result, receipt={**result.receipt, "sha256": "0" * 64})
        if behavior == "wrong-payload":
            path = runner.stage / "0/G.logits.npy"
            raw = path.read_bytes()
            path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    return result


def main():
    from nn_window_ablation.cli import main as nnwa_main

    workflows._initialize_evaluation_worker = controlled_initialize
    # Keep a reference to the production wrapper so the injected wrapper can call it.
    workflows._run_worker_task = controlled_task
    return nnwa_main()


# Spawn imports this module again without running the CLI guard.
if __name__ == "__main__":
    # Use the canonical module name for the functions sent through spawn.
    from tests.parallel_evaluation_checks import main

    raise SystemExit(main())
