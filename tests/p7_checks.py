"""Authored complete-shot data and predictions for portable export tests."""

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pytest

from nn_window_ablation.config import resolve_config
from nn_window_ablation.data import sample_chunk, write_dataset
from nn_window_ablation.paired_workflows import _dataset_record, _query_record, analyze_workflow
from nn_window_ablation.simulation import ShotBatch
from nn_window_ablation.data import open_dataset
from nn_window_ablation.windows import Buffers, build_observation_layout, select_window
from tests.p4_checks import setup_plan


def digest(path: Path):
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": sha256(payload).hexdigest()}


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def cli(directory, *args, env=None):
    result = subprocess.run(
        [str(Path(sys.executable).with_name("nnwa")), *map(str, args)],
        cwd=directory, env=env, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture
def truth_artifacts(tmp_path, compose_settings):
    bundle, plan = setup_plan(split="test", total=4, chunk=4, rounds=36)
    target = tmp_path / "data/held"
    target.parent.mkdir()
    physical = sample_chunk(plan, bundle, 0)
    core_labels = np.zeros((4, 12), dtype=bool)
    core_labels[0, 3] = core_labels[2, 5] = True
    global_labels = np.array([1, 0, 1, 0], dtype=bool)
    write_dataset(target, plan, bundle, chunks=(
        ShotBatch(physical.detectors, global_labels, core_labels, physical.shot_ids, None),
    ), provenance={"fixture": "literal P7 truth table"})
    mapped = open_dataset(target)
    checkpoint = tmp_path / "runs/base/checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    expected_failures = {
        "G": np.array([1, 0, 1, 0], dtype=bool),
        "F": np.array([0, 1, 1, 0], dtype=bool),
        "S": np.array([0, 1, 1, 1], dtype=bool),
    }
    bits = {"G": (global_labels ^ expected_failures["G"])[:, None]}
    for mode in ("F", "S"):
        bits[mode] = core_labels.copy()
        bits[mode][:, 0] ^= expected_failures[mode]
    bits["S"][0, [1, 2]] ^= True
    layout = build_observation_layout(bundle)
    config = resolve_config(compose_settings(
        "runtime=test_cpu", "code.distance=3", "run_name=pred", "data.split=test", "data.id=held",
        "evaluation=paired", "checkpoint.path=runs/base/checkpoint.pt",
    ), invocation_dir=tmp_path)
    pred_root = tmp_path / "runs/pred"
    shard = pred_root / "shards/0"
    shard.mkdir(parents=True)
    files = {}

    def array(name, values, width=None):
        path = shard / name
        np.save(path, values, allow_pickle=False)
        files[name] = {**digest(path), "shape": list(values.shape), "dtype": values.dtype.name}
        if width is not None:
            files[name]["valid_bits"] = width

    array("shot_ids.npy", physical.shot_ids)
    conditions = {}
    for mode, decisions in bits.items():
        array(f"{mode}.decisions.npy", np.packbits(decisions, axis=1, bitorder="little"), decisions.shape[1])
        array(f"{mode}.logits.npy", np.where(decisions, 1, -1).astype(np.float32))
        queries = (layout.global_query,)
        if mode != "G":
            queries = layout.queries
            array(f"{mode}.final.npy", (global_labels ^ expected_failures[mode]).astype(np.uint8)[:, None], 1)
        conditions[mode] = {
            "mode": mode, "condition": mode, "model_state_sha256": "b" * 64,
            "checkpoint": {
                "sha256": "a" * 64, "training_regime": "full_closed", "training_seed": 87,
                "training_replicate": 2, "training_family": "truth", "parent_sha256": None,
                "cumulative_updates": 1,
            },
            "queries": [_query_record(select_window(layout, q, Buffers(3, 3) if mode == "S" else None)) for q in queries],
        }
    prediction = pred_root / "manifest.json"
    write_json(prediction, {
        "schema_version": 1, "status": "complete", "data": _dataset_record(mapped), "conditions": conditions,
        "shards": [{"index": 0, "rows": 4, "files": files}], "resolved_config": config.to_record(),
    })
    analysis = resolve_config(compose_settings(
        "runtime=test_cpu", "code.distance=3", "data.split=test", "data.id=held", "run_name=stats",
        "analysis.predictions=[runs/pred/manifest.json]", "analysis.bootstrap_seed=17",
        "analysis.bootstrap_replicates=41", "analysis.sparse_rule=1",
    ), invocation_dir=tmp_path)
    report = analyze_workflow(analysis)
    return target, prediction, checkpoint, report, expected_failures
