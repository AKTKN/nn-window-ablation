import json
import os
from pathlib import Path

import numpy as np
import torch

from nn_window_ablation.checkpoints import load_evaluation_model
from nn_window_ablation.data import open_dataset
from nn_window_ablation.evaluation import FrozenEvaluator, model_state_sha256
from nn_window_ablation.execution import generate_workflow, train_workflow
from nn_window_ablation.paired_workflows import analyze_workflow, evaluate_workflow
from nn_window_ablation.windows import Buffers
from tests.p5_checks import frozen_pilot
from tests.p6_checks import ARMS, TEST_BUDGET, continuation_pilot
from tests.p7_checks import cli


def test_p7_d5_stats_clean_room_and_separate_redecode(frozen_pilot, tmp_path):
    resolve, mapped, dataset, checkpoint = frozen_pilot
    original = load_evaluation_model(checkpoint)
    assert original.rnn.hidden_size == 256 and original.rnn.num_layers == 4
    assert mapped.plan.spec.distance == 5 and mapped.plan.spec.rounds == 60 and mapped.plan.spec.probability == 0.003
    initial = model_state_sha256(original)
    prediction = evaluate_workflow(resolve(
        "run_name=pred", "data.id=held", "data.split=test", "evaluation=paired",
        "checkpoint.path=runs/base/checkpoint.pt",
    ))
    report = analyze_workflow(resolve(
        "run_name=stats", "data.id=held", "data.split=test", "analysis.predictions=[runs/pred/manifest.json]",
        "analysis.bootstrap_seed=71", "analysis.bootstrap_replicates=39", "analysis.sparse_rule=1",
    ))
    expected = json.loads(report.read_text())
    rows = mapped.shards[0].read().shots
    np.testing.assert_array_equal(np.logical_xor.reduce(rows.core_labels, axis=1), rows.global_labels)
    truth = {}
    for mode in ("G", "F", "S"):
        logits = np.load(prediction.parent / "shards/0" / f"{mode}.logits.npy")
        bits = logits >= 0
        composed = np.logical_xor.reduce(bits, axis=1)
        if mode in ("F", "S"):
            np.testing.assert_array_equal(
                np.logical_xor.reduce(bits ^ rows.core_labels, axis=1), composed ^ rows.global_labels,
            )
        truth[mode] = logits.copy()

    archive = tmp_path / "bits.tar.gz"
    full = tmp_path / "full.tar.gz"
    cli(tmp_path, "export", "--dataset", dataset, "--report", report, "--output", archive)
    cli(tmp_path, "export", "--dataset", dataset, "--report", report, "--output", full, "--include-detectors", "--include-checkpoints")
    local = tmp_path / "workstation"
    local.mkdir()
    cli(local, "extract", archive, "imported")
    cli(local, "extract", full, "redecode")
    stored_directory = tmp_path / "originals-hidden"
    stored_directory.mkdir()
    (tmp_path / "data").rename(stored_directory / "data")
    (tmp_path / "runs").rename(stored_directory / "runs")
    assert not dataset.exists() and not checkpoint.exists() and not prediction.exists()
    imported = local / "imported"
    assert not any(imported.rglob("*.stim")) and not any(imported.rglob("*.pt"))
    assert not any(imported.rglob("detectors.npy"))
    guard = local / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text('''
import os
from pathlib import Path
import sys

def audited(event, args):
    if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0]))
    if path.name in {"structure.json", "detectors.npy", "global.dem", "checkpoint.pt"} or path.suffix == ".stim":
        raise RuntimeError("reanalysis attempted prohibited physical or model access")
    if "originals-hidden" in path.parts:
        raise RuntimeError("reanalysis reached original artifacts")
sys.addaudithook(audited)

from nn_window_ablation import simulation, execution
from nn_window_ablation.models.gnn_rnn import QueryDecoder
def forbidden(*args, **kwargs):
    raise RuntimeError("reanalysis attempted simulation, training, or inference")
simulation.build_circuit_bundle = simulation.sample_shots = forbidden
execution.train_workflow = QueryDecoder.forward = forbidden
Path("guard-active").touch()
''')
    env = {**os.environ, "PYTHONPATH": str(guard)}
    cli(local, "reanalyze", "--bundle", "imported", "--output", "replayed", env=env)
    assert (local / "guard-active").exists()
    reconstructed = json.loads((local / "replayed/report.json").read_text())
    for key in ("conditions", "data", "rates", "contrasts", "analysis", "primary_decomposition"):
        assert reconstructed[key] == expected[key]
    for name in ("rates.csv", "contrasts.csv"):
        assert (local / "replayed" / name).read_bytes() == (imported / "analysis" / name).read_bytes()

    decode_root = local / "redecode"
    manifest = json.loads((decode_root / "manifest.json").read_text())
    assert manifest["capabilities"]["redecode"]
    decoder = load_evaluation_model(decode_root / manifest["runs"][0]["checkpoint"])
    assert model_state_sha256(decoder) == initial
    restored = open_dataset(decode_root / manifest["dataset"]["path"], expected_split="test")
    same = restored.shards[0].read().shots
    np.testing.assert_array_equal(same.shot_ids, rows.shot_ids)
    np.testing.assert_array_equal(same.detectors, rows.detectors)
    evaluator = FrozenEvaluator(decoder, restored.bundle, Buffers(5, 5))
    assert not decoder.training and torch.get_num_threads() == 1
    for mode in truth:
        logits = evaluator.logits(mode, same.detectors[:1], same.shot_ids[:1])
        np.testing.assert_allclose(logits, truth[mode][:1], atol=2e-6, rtol=1e-5)
    assert model_state_sha256(decoder) == initial


def test_p7_d5_adapted_condition_receipts_survive_reanalysis(continuation_pilot, tmp_path):
    resolve, _ = continuation_pilot
    dataset = generate_workflow(resolve("data.split=test", "data.id=held", "data.shots=2", "data.shard_shots=2"))
    paths = []
    for index, arm in enumerate(ARMS):
        checkpoint = train_workflow(resolve(
            f"run_name=arm{index}", f"training={arm}", "training.max_steps=1", "data.split=train",
            "checkpoint.init_from=runs/base/checkpoint.pt", *TEST_BUDGET,
        ))
        paths.append(evaluate_workflow(resolve(
            f"run_name=pred{index}", "evaluation=S", "data.split=test", "data.id=held", f"checkpoint.path={checkpoint}",
        )))
    report = analyze_workflow(resolve(
        "run_name=stats", "analysis=adaptation", "data.split=test", "data.id=held",
        f"analysis.predictions=[{','.join(map(str, paths))}]", "analysis.bootstrap_seed=11",
        "analysis.bootstrap_replicates=10", "analysis.sparse_rule=1",
    ))
    archive = tmp_path / "adaptation.tar.gz"
    cli(tmp_path, "export", "--dataset", dataset.parent, "--report", report, "--output", archive)
    cli(tmp_path, "extract", archive, "relocated")
    cli(tmp_path, "reanalyze", "--bundle", "relocated", "--output", "replayed")
    expected, actual = (json.loads(path.read_text()) for path in (report, tmp_path / "replayed/report.json"))
    assert actual["conditions"] == expected["conditions"]
    assert set(actual["conditions"]) == {"S+", "W", "D"}
    assert actual["contrasts"] == expected["contrasts"]
