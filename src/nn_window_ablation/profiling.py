"""Bounded local CPU profiling for the distance-5 query decoder."""

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import resource
from statistics import mean
from time import perf_counter

from .config import ResolvedConfig, resolve_config
from .runtime import configure_cpu_runtime
from .workflows import _environment_record


def _profile(config: ResolvedConfig, batch_size: int, warmup: int, steps: int) -> dict:
    # The CLI configures native and Torch threads before numerical imports.
    import numpy as np
    import torch

    from .features import batch_features, build_feature_geometry, build_features
    from .models.gnn_rnn import QueryDecoder, prepare_model_batch
    from .models.objective import query_loss, sample_core_indices
    from .simulation import CircuitSpec, build_circuit_bundle, sample_shots
    from .windows import Buffers, build_observation_layout, crop_observations, select_window

    torch.manual_seed(config.seed)
    core_rng = np.random.default_rng(config.seed + 1)
    geometry = config.geometry
    bundle = build_circuit_bundle(CircuitSpec(
        geometry.distance, geometry.rounds, geometry.core_length, config.noise.probability,
    ))
    layout = build_observation_layout(bundle)
    tiles = build_feature_geometry(layout)
    global_view = select_window(layout, layout.global_query)
    full = [select_window(layout, query) for query in layout.queries]
    windows = [
        select_window(layout, query, Buffers(geometry.left_buffer, geometry.right_buffer)) for query in layout.queries
    ]
    model = QueryDecoder(geometry.distance)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    timings: dict[str, list[float]] = {}
    payload_size = 0

    for step in range(warmup + steps):
        start = perf_counter()
        shots = sample_shots(bundle, batch_size, seed=config.seed + step, first_shot_id=step * batch_size)
        sampled = sample_core_indices(batch_size, len(full), rng=core_rng)
        global_labels = torch.from_numpy(shots.global_labels)
        core_labels = torch.from_numpy(shots.core_labels)
        core_indices = torch.from_numpy(sampled)
        sampled_at = perf_counter()

        global_examples = build_features(crop_observations(global_view, shots.detectors, shots.shot_ids), tiles)
        sampled_examples = tuple(
            build_features(
                crop_observations(full[core], shots.detectors[row:row + 1], shots.shot_ids[row:row + 1]), tiles,
            )[0]
            for row, core in enumerate(sampled)
        )
        inputs = prepare_model_batch(batch_features(global_examples + sampled_examples), distance=model.distance)
        graphs_at = perf_counter()

        model.train()
        optimizer.zero_grad(set_to_none=True)
        forward_start = perf_counter()
        logits = model(inputs)
        forward_at = perf_counter()
        loss = query_loss(
            logits[:batch_size], global_labels, logits[batch_size:], core_labels, core_indices=core_indices,
        )
        loss_at = perf_counter()
        loss.total.backward()
        optimizer.step()
        update_at = perf_counter()
        with BytesIO() as stream:
            torch.save(model.checkpoint_payload(), stream)
            payload_size = stream.tell()
        saved_at = perf_counter()

        with torch.inference_mode():
            model.eval()
            evaluation_start = perf_counter()
            evaluation_examples = tuple(
                example for view in windows
                for example in build_features(crop_observations(view, shots.detectors, shots.shot_ids), tiles)
            )
            evaluation = prepare_model_batch(batch_features(evaluation_examples), distance=model.distance)
            eval_graphs_at = perf_counter()
            evaluation_logits = model(evaluation)
            eval_forward_at = perf_counter()
        if not torch.isfinite(loss.total) or not torch.isfinite(evaluation_logits).all():
            raise RuntimeError("nonfinite result in the bounded CPU profile")

        if step >= warmup:
            measured = {
                "sample_and_labels": sampled_at - start,
                "training_graphs": graphs_at - sampled_at,
                "training_forward": forward_at - forward_start,
                "loss": loss_at - forward_at,
                "backward_and_update": update_at - loss_at,
                "serialize_to_memory": saved_at - update_at,
                "training_cycle": saved_at - start,
                "window_evaluation_graphs": eval_graphs_at - evaluation_start,
                "window_evaluation_forward": eval_forward_at - eval_graphs_at,
            }
            for phase, seconds in measured.items():
                timings.setdefault(phase, []).append(seconds)

    averages = {phase: mean(seconds) for phase, seconds in timings.items()}
    train_cycle = averages["training_cycle"]
    eval_cycle = averages["window_evaluation_graphs"] + averages["window_evaluation_forward"]
    environment = _environment_record()
    return {
        "experiment": {
            "distance": geometry.distance, "rounds": geometry.rounds, "core_length": geometry.core_length,
            "left_buffer": geometry.left_buffer, "right_buffer": geometry.right_buffer,
            "bulk_width": geometry.width, "probability": config.noise.probability, "seed": config.seed,
        },
        "architecture": model.architecture(),
        "runtime": {
            "processes": config.runtime.processes, "allocated_cpus": config.runtime.allocated_cpus,
            "torch_intraop": torch.get_num_threads(), "torch_interop": torch.get_num_interop_threads(),
            "native_threads": {name: os.environ[name] for name in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
            )},
            "data_loader_workers": config.data.num_workers, "sampling_processes": config.data.sampling_processes,
        },
        "environment": environment,
        "batch_size": batch_size,
        "warmup_steps": warmup,
        "measured_steps": steps,
        "training_queries_per_trajectory": 2,
        "window_evaluation_queries_per_trajectory": len(windows),
        "mean_seconds": averages,
        "training_trajectories_per_second": batch_size / train_cycle,
        "training_queries_per_second": 2 * batch_size / train_cycle,
        "window_evaluation_trajectories_per_second": batch_size / eval_cycle,
        "window_evaluation_queries_per_second": len(windows) * batch_size / eval_cycle,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "serialized_bytes": payload_size,
    }


def main() -> None:
    from hydra import compose, initialize_config_module

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--allocated-cpus", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.steps < 1 or args.warmup < 0:
        parser.error("batch size and measured steps must be positive and warmup nonnegative")
    with initialize_config_module(config_module="nn_window_ablation.conf", version_base="1.3"):
        composed = compose(config_name="config", overrides=[
            "experiment=pilot_d5", "runtime=test_cpu", "runtime.processes=1",
            "data.num_workers=0", "data.sampling_processes=1", "run_name=p3-profile",
            f"runtime.intraop_threads={args.threads}", f"runtime.allocated_cpus={args.allocated_cpus}",
        ])
    config = resolve_config(composed, invocation_dir=Path.cwd())
    configure_cpu_runtime(config.runtime, config.data)
    report = _profile(config, args.batch_size, args.warmup, args.steps)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        with args.output.open("x") as file:
            file.write(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
