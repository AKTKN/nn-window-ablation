from hashlib import sha256
import json

import numpy as np
import pytest
import torch

from nn_window_ablation.artifacts import ArtifactError, canonical_json, write_json
from nn_window_ablation.checkpoints import load_checkpoint, load_evaluation_model
from nn_window_ablation.data import SamplingPlan, open_dataset, sample_chunk
from nn_window_ablation.execution import generate_workflow
from nn_window_ablation.paired_workflows import analyze_workflow, evaluate_workflow
from nn_window_ablation.simulation import CircuitSpec
from tests.p4_checks import read_batch, reference_seed
from tests.p6_checks import ARMS, TEST_BUDGET, continuation_pilot, reference_model_hash, run_nnwa
from tests.reference.p2_observations import PILOT_SCHEDULE


def test_p6_d5_matched_sweep_and_p5_comparisons(continuation_pilot, tmp_path):
    resolve, parent = continuation_pilot
    base = load_checkpoint(parent)
    base_digest = sha256(parent.read_bytes()).hexdigest()
    base_model = reference_model_hash(base["model"]["state_dict"])
    assert base["counters"]["phase_updates"] == 3 and base["scheduler_state"]["last_epoch"] == 3
    assert base["optimizer_state"]["state"]
    held = generate_workflow(resolve("data.split=test", "data.id=held", "data.shots=2", "data.shard_shots=2"))
    valid = generate_workflow(resolve("data.split=validation", "data.id=val", "data.shots=2", "data.shard_shots=2"))
    data = open_dataset(held.parent, expected_split="test")
    validation = open_dataset(valid.parent, expected_split="validation")
    assert data.plan.spec == CircuitSpec(5, 60, 5, 0.003)
    rows = read_batch(data, 0).shots
    np.testing.assert_array_equal(rows.global_labels, np.logical_xor.reduce(rows.core_labels, axis=1))

    generate_workflow(resolve(
        "data.split=train", "training=continued_full", "data.id=continuation", "data.shots=5", "data.shard_shots=3",
    ))
    run_nnwa(
        tmp_path, "train", "--multirun", "runtime=test_cpu", "run_name=matched",
        f"training={','.join(ARMS)}", "checkpoint.init_from=runs/base/checkpoint.pt",
        "training.max_steps=2", "training.validation_data_id=val", "checkpoint.save_every_updates=3",
        "data.train_source=stored_shards", "data.id=continuation", *TEST_BUDGET,
    )
    audits, predictions = [], []
    for index, (arm, condition) in enumerate(zip(ARMS, ("S+", "W", "D"), strict=True)):
        path = tmp_path / f"runs/matched/hydra_sweep/{index}/checkpoint.pt"
        state = load_checkpoint(path)
        model = load_evaluation_model(path)
        assert (model.rnn.hidden_size, model.rnn.num_layers, model.rnn.dropout) == (256, 4, 0.0)
        assert all(torch.isfinite(tensor).all() for tensor in model.state_dict().values())
        assert state["run"]["resolved_config"]["training"]["name"] == arm
        assert state["run"]["parent"]["sha256"] == base_digest
        assert state["counters"] == {
            "parent_updates": 3, "phase_updates": 2, "cumulative_updates": 5, "trajectories": 4, "query_examples": 8,
        }
        assert set(int(moment["step"]) for moment in state["optimizer_state"]["state"].values()) == {2}
        assert state["scheduler_state"]["last_epoch"] == 2
        audit = state["continuation_audit"]
        audits.append(audit)
        controls = audit["controls"]
        assert controls["initial_model_sha256"] == base_model and controls["parent_sha256"] == base_digest
        assert controls["validation"] == {
            "schedule": "terminal_only", "phase_update": 2,
            "dataset_id": validation.plan.dataset_id, "manifest_sha256": validation.manifest_sha256,
            "shots": 2,
        }
        assert audit["validation"]["phase_update"] == 2 and audit["validation"]["shots"] == 2
        assert audit["validation"]["ler"] == audit["validation"]["failures"] / 2
        plan = SamplingPlan.from_record(controls["trainer"]["source"]["plan"])
        for chunk in (0, 1):
            sampled = sample_chunk(plan, data.bundle, chunk)
            np.testing.assert_array_equal(sampled.global_labels, np.logical_xor.reduce(sampled.core_labels, axis=1))
        assert plan.namespace.phase == "continuation"
        assert plan.dataset_id != SamplingPlan.from_record(base["contract"]["source"]["plan"]).dataset_id
        assert audit["updates"] == len(audit["trace_prefix"]) == 2
        assert np.isfinite([row["loss"] for row in audit["trace_prefix"]]).all()
        expected_tokens = 0
        for trace in audit["trace_prefix"]:
            expected_tokens += len(trace["cores"]) * len(PILOT_SCHEDULE.select())
            pair = None if arm == "continued_full" else (5, 5)
            if arm == "dynamic_window":
                pair = (trace["buffers"]["left"], trace["buffers"]["right"])
                assert set(pair) <= {0, 3, 5}
            else:
                assert trace["buffers"] is None
            expected_tokens += sum(len(PILOT_SCHEDULE.select(core, pair)) for core in trace["cores"])
        usage = audit["resources"]
        assert usage["detector_tokens"] == expected_tokens
        assert usage["allocated_cpu_core_hours"] == usage["step_wall_seconds"] * 8 / 3600
        assert usage["step_wall_seconds"] > 0 and usage["process_cpu_seconds"] > 0 and usage["peak_process_rss_kib"] > 0
        assert json.loads(path.with_name("run.json").read_text())["continuation_audit"]["input_sha256"] == audit["input_sha256"]

        prediction = evaluate_workflow(resolve(
            f"run_name=pred-{index}", "evaluation=paired", "data.id=held", "data.split=test",
            f"checkpoint.path={path}", "training=full_closed",
        ))
        predictions.append(prediction)
        manifest = json.loads(prediction.read_text())
        assert manifest["conditions"]["S"]["condition"] == condition
        assert manifest["conditions"]["G"]["condition"] == f"G@{arm}"
        assert manifest["conditions"]["F"]["condition"] == f"F@{arm}"
        provenance = manifest["conditions"]["S"]["checkpoint"]
        assert provenance["continuation_controls_sha256"] == audit["controls_sha256"]
        assert provenance["continuation_input_sha256"] == audit["input_sha256"]
        assert provenance["initial_model_sha256"] == base_model

    assert len({audit["controls_sha256"] for audit in audits}) == len({audit["input_sha256"] for audit in audits}) == 1
    plan = SamplingPlan.from_record(audits[0]["controls"]["trainer"]["source"]["plan"])
    expected_order = [
        (chunk, int(row)) for chunk in range(2)
        for row in np.random.default_rng(reference_seed(plan.namespace, "trajectory_order", chunk)).permutation(plan.chunk_bounds(chunk)[1])
    ][:4]
    expected_cores = np.random.default_rng(reference_seed(plan.namespace, "core_query")).integers(0, 12, size=4).tolist()
    for audit in audits:
        assert [
            (identity["shard"], identity["row"])
            for row in audit["trace_prefix"] for identity in row["trajectories"]
        ] == expected_order
        assert [core for row in audit["trace_prefix"] for core in row["cores"]] == expected_cores

    options = (
        "analysis=adaptation", "data.id=held", "data.split=test", "analysis.bootstrap_seed=55",
        "analysis.bootstrap_replicates=37", "analysis.sparse_rule=2",
        f"analysis.predictions=[{','.join(map(str, predictions))}]",
    )
    report = json.loads(analyze_workflow(resolve("run_name=report", *options)).read_text())
    assert set(report["contrasts"]) == {"Splus_minus_W", "W_minus_D"}
    queries = [report["conditions"][condition]["queries"] for condition in ("S+", "W", "D")]
    assert queries[0] == queries[1] == queries[2]
    for pair in report["contrasts"].values():
        a, b = (report["rates"][condition]["failures"] for condition in (pair["first"], pair["second"]))
        assert pair["difference"] == (a - b) / 2

    changed = json.loads(predictions[-1].read_text())
    changed["conditions"]["S"]["checkpoint"]["continuation_input_sha256"] = "3" * 64
    changed["contract"]["conditions"] = changed["conditions"]
    changed["contract_sha256"] = sha256(canonical_json(changed["contract"])).hexdigest()
    write_json(predictions[-1], changed)
    with pytest.raises(ArtifactError, match="matched initialization, controls, and trajectory/core draws"):
        analyze_workflow(resolve("run_name=unmatched", *options))
