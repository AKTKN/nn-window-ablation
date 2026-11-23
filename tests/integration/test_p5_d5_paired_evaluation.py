from hashlib import sha256
import json

import numpy as np
import pytest

from nn_window_ablation.artifacts import ArtifactError, file_integrity, write_json
from nn_window_ablation.checkpoints import load_evaluation_model
from nn_window_ablation.evaluation import FrozenEvaluator, model_state_sha256
from nn_window_ablation.paired_workflows import analyze_workflow, evaluate_workflow
from nn_window_ablation.simulation import ArrivalPhase
from nn_window_ablation.windows import Buffers, crop_observations
from tests.p5_checks import frozen_pilot


def _literal_parity(bits):
    return np.array([sum(int(bit) for bit in row) % 2 for row in bits], dtype=bool)


def test_p5_full_d5_paired_pipeline_and_order_invariance(frozen_pilot):
    resolve, dataset, _, checkpoint = frozen_pilot
    decoder = load_evaluation_model(checkpoint)
    assert (decoder.rnn.hidden_size, decoder.rnn.num_layers) == (256, 4)
    before = model_state_sha256(decoder)
    evaluator = FrozenEvaluator(decoder, dataset.bundle, Buffers(5, 5))
    row = dataset.shards[0].read().shots
    direct = {mode: evaluator.logits(mode, row.detectors, row.shot_ids) for mode in ("G", "F", "S")}
    for mode in ("S", "G", "F"):
        reordered = evaluator.logits(mode, row.detectors[::-1], row.shot_ids[::-1])
        partitioned = np.concatenate([
            evaluator.logits(mode, row.detectors[i:i + 1], row.shot_ids[i:i + 1]) for i in range(2)
        ])
        np.testing.assert_allclose(reordered[::-1], direct[mode], atol=2e-6, rtol=1e-5)
        np.testing.assert_allclose(partitioned, direct[mode], atol=2e-6, rtol=1e-5)
    assert model_state_sha256(decoder) == before
    penultimate, final = evaluator.views("S")[-2:]
    assert (penultimate.metadata.release.round, penultimate.metadata.release.phase) == (60, ArrivalPhase.ANCILLA)
    assert (final.metadata.release.round, final.metadata.release.phase) == (60, ArrivalPhase.DESTRUCTIVE_READOUT)
    readout = {d.detector_index for d in dataset.bundle.detectors if d.arrival.phase == ArrivalPhase.DESTRUCTIVE_READOUT}
    assert readout and readout <= set(final.detector_indices) and not readout & set(penultimate.detector_indices)
    changed = row.detectors.copy()
    changed[:, sorted(readout)] ^= True
    np.testing.assert_array_equal(
        crop_observations(penultimate, row.detectors, row.shot_ids).bits,
        crop_observations(penultimate, changed, row.shot_ids).bits,
    )
    np.testing.assert_array_equal(evaluator.logits("S", changed, row.shot_ids)[:, 10], direct["S"][:, 10])

    # A caller-side training preset has no bearing on the checkpoints' conditions.
    path = evaluate_workflow(resolve(
        "run_name=eval", "data.id=held", "data.split=test", "evaluation=paired",
        "evaluation.batch_size=2", "training=dynamic_window", "checkpoint.path=runs/base/checkpoint.pt",
    ))
    manifest = json.loads(path.read_text())
    assert manifest["resolved_config"]["training"]["name"] == "dynamic_window"
    assert manifest["data"]["manifest_sha256"] == dataset.manifest_sha256
    stored = manifest["conditions"]
    checksum = sha256(checkpoint.read_bytes()).hexdigest()
    for mode in ("G", "F", "S"):
        assert stored[mode]["condition"] == mode
        assert stored[mode]["checkpoint"]["training_regime"] == "full_closed"
        assert stored[mode]["checkpoint"]["sha256"] == checksum
        assert stored[mode]["model_state_sha256"] == before
        assert [q["core_index"] for q in stored[mode]["queries"]] == ([None] if mode == "G" else list(range(12)))
    assert stored["S"]["queries"][10]["release"] == {"round": 60, "phase": "ancilla"}
    assert stored["S"]["queries"][11]["release"] == {"round": 60, "phase": "destructive_readout"}

    all_errors = {mode: [] for mode in stored}
    all_core_counts = {mode: [] for mode in ("F", "S")}
    for shard in dataset.shards:
        shots = shard.read().shots
        raw = path.parent / "shards" / str(shard.index)
        np.testing.assert_array_equal(np.load(raw / "shot_ids.npy"), shots.shot_ids)
        np.testing.assert_array_equal(_literal_parity(shots.core_labels), shots.global_labels)
        for mode in ("G", "F", "S"):
            width = 1 if mode == "G" else 12
            logits = np.load(raw / f"{mode}.logits.npy")
            assert logits.shape == (len(shots.shot_ids), width) and np.isfinite(logits).all()
            bits = np.unpackbits(np.load(raw / f"{mode}.decisions.npy"), axis=1, count=width, bitorder="little")
            np.testing.assert_array_equal(bits, logits >= 0)
            if shard.index == 0:
                np.testing.assert_allclose(logits, direct[mode], atol=2e-6, rtol=1e-5)
            final_predictions = _literal_parity(bits)
            expected = final_predictions != shots.global_labels
            all_errors[mode].extend(expected)
            if mode in ("F", "S"):
                matches = bits != shots.core_labels
                np.testing.assert_array_equal(_literal_parity(matches), expected)
                np.testing.assert_array_equal(np.load(raw / f"{mode}.final.npy")[:, 0], final_predictions)
                all_core_counts[mode].extend(matches.sum(axis=1).tolist())

    overrides = (
        "data.id=held", "data.split=test", "analysis.predictions=[runs/eval/manifest.json]",
        "analysis.sparse_rule=1", "analysis.bootstrap_seed=23", "analysis.bootstrap_replicates=25",
        "analysis.ratio_min_denominator_failures=2",
    )
    analysis = analyze_workflow(resolve("run_name=report", *overrides))
    report = json.loads(analysis.read_text())
    assert report["analysis"]["unit"] == "whole_trajectory"
    assert report["analysis"]["bootstrap_replicates"] == 25
    assert report["primary_decomposition"]["direct_numerator"] == report["primary_decomposition"]["sum_of_parts_numerator"]
    assert (analysis.parent / "rates.csv").is_file() and (analysis.parent / "contrasts.csv").is_file()
    for mode, errors in all_errors.items():
        row = report["rates"][mode]
        assert row["shots"] == 3 and row["failures"] == sum(errors) and row["ler"] == sum(errors) / 3
        if mode in ("F", "S"):
            expected_histogram = [all_core_counts[mode].count(number) for number in range(13)]
            assert row["core_diagnostics"]["disagreement_count_histogram"] == expected_histogram
            assert row["core_diagnostics"]["odd_disagreement_rate"] == row["ler"]
    for name, pair in report["contrasts"].items():
        pairs = list(zip(all_errors[pair["first"]], all_errors[pair["second"]], strict=True))
        assert (pair["n10"], pair["n01"]) == (pairs.count((True, False)), pairs.count((False, True))), name

    # A forged endpoint release is rejected before analysis.
    release = manifest["conditions"]["S"]["queries"][10]["release"]
    release["phase"] = "destructive_readout"
    write_json(path, manifest)
    with pytest.raises(ArtifactError, match="query partition"):
        analyze_workflow(resolve("run_name=wrong-endpoint", *overrides))
    release["phase"] = "ancilla"

    # Reseal a reordered index payload to test pairing separately from byte checksums.
    ids_path = path.parent / "shards/0/shot_ids.npy"
    np.save(ids_path, np.load(ids_path)[::-1], allow_pickle=False)
    manifest["shards"][0]["files"]["shot_ids.npy"].update(file_integrity(ids_path))
    write_json(path, manifest)
    with pytest.raises(ArtifactError, match="shot identities differ or are reordered"):
        analyze_workflow(resolve("run_name=wrong-order", *overrides))
