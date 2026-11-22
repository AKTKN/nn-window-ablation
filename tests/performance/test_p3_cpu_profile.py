import json
import os
import subprocess
import sys

import pytest


@pytest.mark.performance
@pytest.mark.parametrize("threads", [1, 2, 4, 8])
def test_p3_cpu_profile(tmp_path, threads):
    affinity = len(os.sched_getaffinity(0))
    declared = min(affinity, 8)
    if threads > declared:
        pytest.skip(f"profile requires {threads} numerical CPUs, allocation provides {declared}")
    variables = (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    )
    env = dict(os.environ, **{variable: str(threads) for variable in variables})
    output = tmp_path / f"p3-cpu-{threads}.json"
    process = subprocess.run(
        [
            sys.executable, "-m", "nn_window_ablation.profiling",
            "--threads", str(threads), "--allocated-cpus", str(declared),
            "--batch-size", "2", "--warmup", "1", "--steps", "2", "--output", str(output),
        ],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120, check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads(output.read_text())
    assert json.loads(process.stdout) == report
    assert report["experiment"] == {
        "distance": 5, "rounds": 60, "core_length": 5, "left_buffer": 5, "right_buffer": 5,
        "bulk_width": 15, "probability": 0.003, "seed": 101,
    }
    assert report["architecture"]["hidden_size"] == 256 and report["architecture"]["gru_layers"] == 4
    assert report["architecture"]["spatial_grids_inner_first"] == [2]
    assert report["batch_size"] == report["measured_steps"] == 2 and report["warmup_steps"] == 1
    assert report["runtime"]["processes"] == report["runtime"]["torch_interop"] == 1
    assert report["runtime"]["torch_intraop"] == threads
    assert report["runtime"]["native_threads"] == {variable: str(threads) for variable in variables}
    assert report["runtime"]["allocated_cpus"] == declared
    assert report["runtime"]["data_loader_workers"] == 0 and report["runtime"]["sampling_processes"] == 1
    assert report["training_queries_per_trajectory"] == 2
    assert report["window_evaluation_queries_per_trajectory"] == 12
    assert set(report["mean_seconds"]) == {
        "sample_and_labels", "training_graphs", "training_forward", "loss", "backward_and_update",
        "serialize_to_memory", "training_cycle", "window_evaluation_graphs", "window_evaluation_forward",
    }
    assert all(value > 0 for value in report["mean_seconds"].values())
    assert report["training_queries_per_second"] == pytest.approx(2 * report["training_trajectories_per_second"])
    assert report["window_evaluation_queries_per_second"] == pytest.approx(
        12 * report["window_evaluation_trajectories_per_second"]
    )
    assert report["serialized_bytes"] > 0
    assert report["peak_rss_kib"] * 1024 >= report["serialized_bytes"]
    assert report["environment"]["cpu"]["model"]
