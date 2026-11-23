"""Read back scientific controls and coverage from the completed audit."""

import numpy as np
import torch

from nn_window_ablation.artifacts import file_integrity, read_json
from nn_window_ablation.checkpoints import load_checkpoint
from nn_window_ablation.continuation import model_tensors_sha256
from nn_window_ablation.data import open_dataset
from nn_window_ablation.generation import dataset_semantics
from nn_window_ablation.paired_workflows import _read_manifest, _read_shard


def _prediction(held, checkpoint, output):
    mapped = open_dataset(held.parent, expected_split="test")
    manifest = _read_manifest(output, mapped)
    state = load_checkpoint(checkpoint)
    digest = model_tensors_sha256(state["model"]["state_dict"])
    receipt = file_integrity(checkpoint)
    assert manifest["contract"]["checkpoint"] == receipt
    assert manifest["data"]["manifest_sha256"] == file_integrity(held)["sha256"]
    assert all(item["model_state_sha256"] == digest for item in manifest["conditions"].values())
    counts = {item["condition"]: 0 for item in manifest["conditions"].values()}
    paired = {name: [0, 0] for name in ("S_minus_G", "F_minus_G", "S_minus_F")} if len(counts) == 3 else {}
    rows = 0
    for descriptor in mapped.shards:
        with mapped.open_shard(descriptor.index, labels_only=True) as shard:
            labels, core = shard.read_labels()
            assert np.array_equal(np.bitwise_xor.reduce(core, axis=1), labels)
            errors = {}
            for label, bits, scores, final in _read_shard(output, manifest, shard):
                error = final ^ labels
                if bits.shape[1] == 12:
                    assert np.array_equal(np.bitwise_xor.reduce(bits ^ core, axis=1), error)
                assert np.array_equal(bits, scores >= 0)
                counts[label] += int(error.sum())
                errors[label] = error
            for name, pair in paired.items():
                left, _, right = name.split("_")
                a, b = errors[left], errors[right]
                pair[0] += int(np.count_nonzero(a & ~b))
                pair[1] += int(np.count_nonzero(b & ~a))
            rows += len(labels)
    assert rows == mapped.plan.total_rows
    return {"rows": rows, "failures": counts, "discordances": paired}


def _train_state(checkpoint, dataset, updates, global_batch, phase):
    state = load_checkpoint(checkpoint)
    assert state["contract"]["source"]["dataset_manifest_sha256"] == file_integrity(dataset)["sha256"]
    assert state["counters"]["phase_updates"] == updates
    assert state["counters"]["trajectories"] == updates * global_batch
    assert state["counters"]["query_examples"] == 2 * updates * global_batch
    assert read_json(checkpoint.with_name("run.json"))["status"] == "complete"
    ordered = [row for update in state["training_inputs"]["trace_prefix"] for row in update["trajectories"]]
    assert len(ordered) == len({(row["shard"], row["row"]) for row in ordered}) == updates * global_batch
    assert len({row["dataset_id"] for row in ordered}) == 1
    if phase:
        assert state["counters"]["parent_updates"] > 0
    return state


def _max_error(reference, other, relative, absolute):
    torch.testing.assert_close(other, reference, rtol=relative, atol=absolute)
    return float((reference - other).abs().max())


def _compare_training(serial, distributed):
    assert serial["training_inputs"] == distributed["training_inputs"]
    assert serial["counters"] == distributed["counters"]
    assert serial["scheduler_state"] == distributed["scheduler_state"]
    assert serial["cursor"] == distributed["cursor"]
    world = distributed["distributed_state"]["world_size"]
    assert world in (2, 4, 8)
    weights = max(_max_error(value, distributed["model"]["state_dict"][key], 1e-4, 3e-5)
                  for key, value in serial["model"]["state_dict"].items())
    serial_opt, distributed_opt = serial["optimizer_state"], distributed["optimizer_state"]
    assert serial_opt["param_groups"] == distributed_opt["param_groups"]
    assert serial_opt["state"].keys() == distributed_opt["state"].keys()
    moments = {"exp_avg": 0, "exp_avg_sq": 0}
    for key, values in serial_opt["state"].items():
        paired = distributed_opt["state"][key]
        assert torch.equal(values["step"], paired["step"])
        for name, atol in (("exp_avg", 2e-7), ("exp_avg_sq", 2e-9)):
            moments[name] = max(moments[name], _max_error(values[name], paired[name], 2e-4, atol))
    return {"max_weight_abs": weights, "max_moment_abs": moments}


def verify(audit):
    outputs = {"data": {}, "chains": [], "training_comparisons": {}, "frozen": {}, "continuation": []}
    canonical_semantics = {}
    for name, path, split in audit.datasets:
        manifest = read_json(path)
        mapped = open_dataset(path.parent, expected_split=split)
        mapped.verify_payloads()
        semantics = dataset_semantics(manifest)
        key = mapped.plan.namespace.phase, split
        assert semantics == canonical_semantics.setdefault(key, semantics)
        outputs["data"][name] = {"path": str(path), "sha256": file_integrity(path)["sha256"],
                                 "dataset_id": mapped.plan.dataset_id, "rows": mapped.plan.total_rows}
    global_inputs = {}
    rank_reference = load_checkpoint(audit.chains[0][1])
    for training, checkpoint, held, prediction in audit.chains:
        values = _prediction(held, checkpoint, prediction)
        outputs["chains"].append({"prediction": str(prediction), "training": str(training),
                                  "checkpoint": str(checkpoint), "held": str(held), **values})

    for name, count, distributed, path, dataset, updates, phase in audit.trains:
        state = _train_state(path, dataset, updates, audit.recipe["global_batch"], phase)
        assert (state.get("distributed_state") is not None) == distributed
        if distributed:
            assert state["distributed_state"]["world_size"] == count
        if phase is None:
            trail = state["training_inputs"]
            assert trail == global_inputs.setdefault("base", trail)
        if name.startswith("train-r") and not distributed:
            paired = path.parents[1] / name.replace("-serial", "-ddp") / "checkpoint.pt"
            outputs["training_comparisons"][name] = _compare_training(state, load_checkpoint(paired))
        if name.startswith("train-r") and distributed and count == 8:
            outputs["training_comparisons"][name + "-one-thread-rank-control"] = _compare_training(rank_reference, state)

    canonical = read_json(audit.frozen[1, 1])
    reference_train, reference_checkpoint, reference_held, _ = audit.chains[0]
    reference_statistics = _prediction(reference_held, reference_checkpoint, audit.frozen[1, 1])
    for (count, repeat), manifest in audit.frozen.items():
        record = read_json(manifest)
        assert record["shards"] == canonical["shards"]
        assert record["contract_sha256"] == canonical["contract_sha256"]
        counts = _prediction(reference_held, reference_checkpoint, manifest)
        assert counts == reference_statistics
        outputs["frozen"][f"c{count}-r{repeat}"] = {"manifest": str(manifest), **counts}

    report, replay = read_json(audit.report), read_json(audit.replay)
    for key, value in reference_statistics["failures"].items():
        assert report["rates"][key]["failures"] == value
    for contrast, (n10, n01) in reference_statistics["discordances"].items():
        assert (report["contrasts"][contrast]["n10"], report["contrasts"][contrast]["n01"]) == (n10, n01)
    for key in ("rates", "contrasts", "primary_decomposition"):
        assert report[key] == replay[key]
    outputs["portable_reanalysis_matches"] = True

    controls = set()
    audit_inputs = set()
    for checkpoint, prediction, held, parent in audit.continuations:
        state = load_checkpoint(checkpoint)
        trail = state["continuation_audit"]
        controls.add(trail["controls_sha256"])
        audit_inputs.add(trail["input_sha256"])
        assert trail["controls"]["parent_sha256"] == file_integrity(parent)["sha256"]
        assert trail["validation"]["shots"] == 4
        assert {int(value["step"]) for value in state["optimizer_state"]["state"].values()} == {4}
        buffers = [update["buffers"] for update in state["training_inputs"]["trace_prefix"]]
        if checkpoint.parent.name == "dynamic_window":
            assert all(set(pair.values()) <= {0, 3, 5} for pair in buffers)
        else:
            assert buffers == [None] * 4
        outputs["continuation"].append({"checkpoint": str(checkpoint), "buffers": buffers,
                                        **_prediction(held, checkpoint, prediction)})
    assert len(controls) == len(audit_inputs) == 1
    outputs["skip_reason"] = audit.recipe["skip_reason"]
    return outputs
