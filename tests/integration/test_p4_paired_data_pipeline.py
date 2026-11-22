import json

import numpy as np
import pytest
import torch

from nn_window_ablation.artifacts import ArtifactError
from nn_window_ablation.checkpoints import load_checkpoint
from nn_window_ablation.config import resolve_config
from nn_window_ablation.data import TrajectoryID, open_dataset, sample_chunk
from nn_window_ablation.execution import generate_workflow, train_workflow
from nn_window_ablation.features import attach_supervision, batch_features, build_feature_geometry
from nn_window_ablation.models.gnn_rnn import QueryDecoder
from nn_window_ablation.simulation import CircuitSpec, build_circuit_bundle
from nn_window_ablation.windows import Buffers, build_observation_layout, select_window
from tests.p3_checks import examples_for, model_inputs


@pytest.mark.parametrize("probability", [0.003, 0.0])
def test_genuine_pilot_reloads_paired_observations_and_queries(tmp_path, compose_settings, probability):
    config = resolve_config(compose_settings(
        "runtime=test_cpu", "data.split=test", "data.shots=3", "data.shard_shots=2", "data.id=paired",
        f"noise.probability={probability}",
    ), invocation_dir=tmp_path)
    manifest = generate_workflow(config)
    mapped = open_dataset(manifest.parent, expected_split="test")
    direct_bundle = build_circuit_bundle(CircuitSpec(5, 60, 5, probability))
    assert mapped.bundle == direct_bundle
    assert [(m.index, m.detectors.shape[0]) for m in mapped.shards] == [(0, 2), (1, 1)]
    assert direct_bundle.spec.distance == 5 and direct_bundle.spec.rounds == 60
    torch.manual_seed(117)
    model = QueryDecoder(5).eval()
    observed_events = 0
    received = []
    for shard in mapped.shards:
        expected = sample_chunk(mapped.plan, direct_bundle, shard.index)
        batch = shard.read()
        received.extend(batch.identities)
        assert batch.identities == tuple(
            TrajectoryID(mapped.plan.dataset_id, shard.index, row) for row in range(len(expected.shot_ids))
        )
        for field in ("detectors", "global_labels", "core_labels", "shot_ids"):
            np.testing.assert_array_equal(getattr(batch.shots, field), getattr(expected, field))
        np.testing.assert_array_equal(np.bitwise_xor.reduce(batch.shots.core_labels, axis=1), expected.global_labels)
        observed_events += np.count_nonzero(batch.shots.detectors)

        def get_queries(bundle, rows):
            layout = build_observation_layout(bundle)
            geometry = build_feature_geometry(layout)
            full = select_window(layout, layout.global_query)
            final = select_window(layout, layout.core_query(11), Buffers(5, 5))
            outputs = examples_for(full, geometry, rows) + examples_for(final, geometry, rows)
            labels = attach_supervision(batch_features(outputs), rows).labels
            np.testing.assert_array_equal(labels, np.concatenate((expected.global_labels, expected.core_labels[:, 11])))
            return outputs

        with torch.inference_mode():
            direct = model(model_inputs(get_queries(direct_bundle, expected)))
            reloaded = model(model_inputs(get_queries(mapped.bundle, batch.shots)))
        assert torch.isfinite(direct).all()
        torch.testing.assert_close(direct, reloaded, rtol=0, atol=0)
    assert len(set(received)) == 3
    assert bool(observed_events) == bool(probability)

    payload = manifest.parent / "shards" / "0" / "shot_ids.npy"
    ids = np.load(payload, allow_pickle=False)
    np.save(payload, ids[::-1], allow_pickle=False)
    with pytest.raises(ArtifactError, match="size or checksum mismatch for shot_ids.npy"):
        open_dataset(manifest.parent)


def test_workflows_train_from_finalized_train_shards_and_resume_in_place(tmp_path, compose_settings):
    def resolve(*options):
        return resolve_config(compose_settings("runtime=test_cpu", *options), invocation_dir=tmp_path)

    manifest = generate_workflow(resolve("data.split=train", "data.shots=3", "data.shard_shots=2", "data.id=stored"))
    mapped = open_dataset(manifest.parent, expected_split="train")
    overrides = (
        "data.split=train", "data.train_source=stored_shards", "data.id=stored", "training.max_steps=1",
        "training.batch_size=1", "training.optimizer=sgd", "training.scheduler=none", "training.learning_rate=0.001",
    )
    checkpoint = train_workflow(resolve(*overrides), stop_after=0)
    summary = checkpoint.with_name("run.json")
    assert json.loads(summary.read_text())["status"] == "paused"
    resumed = resolve(*overrides, "checkpoint.resume_from=runs/p0-unit/checkpoint.pt")
    assert train_workflow(resumed) == checkpoint
    state = load_checkpoint(checkpoint)
    assert state["contract"]["source"]["dataset_manifest_sha256"] == mapped.manifest_sha256
    assert state["counters"]["phase_updates"] == 1
    assert state["last_update"]["trajectories"][0]["dataset_id"] == mapped.plan.dataset_id
    assert json.loads(summary.read_text())["status"] == "complete"
    after = checkpoint.read_bytes()
    assert train_workflow(resumed) == checkpoint
    assert checkpoint.read_bytes() == after
