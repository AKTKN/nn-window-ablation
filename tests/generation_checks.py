"""Spawnable controls for exercising the real generation worker."""

from contextlib import contextmanager
from dataclasses import asdict
import json
import os
import pickle

from nn_window_ablation import data, generation
from nn_window_ablation.artifacts import write_json
from nn_window_ablation.evaluation_pool import append_event
from tests.parallel_evaluation_checks import await_file


original_init = generation._initialize_sampler
original_task = generation._run_sampler


def audited_initializer(recipe, backend, shards, attempt, log_dir):
    original_build = generation.build_circuit_bundle
    builds = 0

    def build(spec):
        nonlocal builds
        builds += 1
        return original_build(spec)

    generation.build_circuit_bundle = build
    try:
        details = original_init(recipe, backend, shards, attempt, log_dir)
    finally:
        generation.build_circuit_bundle = original_build
    import sys

    assert builds == 1
    assert "torch" not in sys.modules and "nn_window_ablation.models.gnn_rnn" not in sys.modules
    return {**details, "real_circuit_builds": builds}


def controlled_generation(task):
    action = os.environ.get("NNWA_GENERATION_TEST", "")
    worker = generation._sampler
    barrier = worker.shards.parents[2] / "generation-test-barriers"
    if task.index == 0 and action in ("reorder", "pause"):
        barrier.mkdir(exist_ok=True)
        original = data.atomic_file

        @contextmanager
        def blocked(path):
            with original(path) as file:
                yield file
                if path.name == "detectors.npy":
                    write_json(barrier / "before-write-publication", {"pid": os.getpid(), "path": str(path)})
                    await_file(barrier / "release")
        data.atomic_file = blocked
        try:
            result = original_task(task)
        finally:
            data.atomic_file = original
    else:
        if task.index == 1 and action in ("reorder", "pause"):
            await_file(barrier / "before-write-publication")
        if task.index == 2 and action == "reorder":
            (barrier / "replenished").touch()
            (barrier / "release").touch()
        result = original_task(task)
    json.dumps(asdict(result))
    append_event(worker.log, "ipc", index=task.index, task_bytes=len(pickle.dumps(task)), result_bytes=len(pickle.dumps(result)))
    return result


def main():
    from nn_window_ablation.cli import main as cli

    generation._run_sampler = controlled_generation
    return cli()


if __name__ == "__main__":
    from tests.generation_checks import main

    raise SystemExit(main())
