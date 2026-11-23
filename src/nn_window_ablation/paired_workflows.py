"""Publish and verify paired predictions and whole-trajectory analysis records."""

from dataclasses import asdict
from io import StringIO
from pathlib import Path
import csv

from .artifacts import ArtifactError, atomic_file, file_integrity, read_json, staged_directory, verify_file, write_json
from .config import ConfigError, ResolvedConfig, require_execution_config, validate_run_name
from .runtime import configure_cpu_runtime
from .workflows import _environment_record


PREDICTION_SCHEMA = 1
PRIMARY_CONTRASTS = ("S_minus_G", "F_minus_G", "S_minus_F")
EVALUATION_PRESETS = {
    "G": ("global", "full_closed", "identity"),
    "F": ("all_cores", "full_closed", "xor"),
    "S": ("all_cores", "deployment_window", "xor"),
    "paired": ("global_and_all_cores", "paired", "identity_and_xor"),
}


def check_paired_execution(command: str, config: ResolvedConfig) -> None:
    require_execution_config(config, command)
    validate_run_name(config.data.id)
    if config.data.split not in ("test", "validation") or config.geometry.core_count != 12:
        raise ConfigError("paired workflows require a held-out split and the complete 12-core partition")
    if (
        (config.runtime.processes, config.data.num_workers, config.data.sampling_processes) != (1, 0, 1)
        or any(path is not None for path in (config.checkpoint.init_from, config.checkpoint.resume_from))
    ):
        raise ConfigError("paired workflows require a single process and no training checkpoint operations")
    if command == "evaluate":
        settings = config.evaluation
        if (
            EVALUATION_PRESETS.get(settings.id) != (settings.query, settings.context, settings.aggregation)
            or settings.threshold != 0.5 or not settings.reset_recurrent_state_per_query or settings.update_model_state
        ):
            raise ConfigError("evaluation must use its declared G/F/S query, context, threshold, and frozen state")
    else:
        settings = config.analysis
        if (
            (settings.unit, settings.confidence_level, settings.ler_interval, settings.paired_interval,
             settings.sparse_interval, settings.pool_training_seeds_as_independent_shots)
            != ("whole_trajectory", 0.95, "wilson", "paired_bootstrap", "conservative_exact_discordances", False)
        ):
            raise ConfigError("analysis must use the whole-trajectory 95% interval policy and separate training seeds")
        if not settings.predictions or len(set(settings.predictions)) != len(settings.predictions):
            raise ConfigError("analysis.predictions must list distinct evaluation manifest paths")
        if not settings.contrasts or len(set(settings.contrasts)) != len(settings.contrasts):
            raise ConfigError("analysis.contrasts must select distinct paired contrasts")


def _run_path(config: ResolvedConfig, sweep_index: int | None) -> Path:
    root = config.paths.run_root / config.run_name
    if sweep_index is not None:
        root = root / "hydra_sweep" / str(sweep_index)
    root.parent.mkdir(parents=True, exist_ok=True)
    return root


def _dataset(config: ResolvedConfig):
    from .data import BITORDER, LABEL_SCHEMA, open_dataset
    from .simulation import CircuitSpec

    geometry = config.geometry
    data_root = config.paths.data_root / config.data.id
    mapped = open_dataset(data_root, expected_split=config.data.split)
    expected = CircuitSpec(geometry.distance, geometry.rounds, geometry.core_length, config.noise.probability)
    if mapped.plan.spec != expected:
        raise ArtifactError("dataset circuit differs from the resolved evaluation configuration")
    if (config.data.label_schema, config.data.bitorder, config.data.storage) != (LABEL_SCHEMA, BITORDER, "packed_npy"):
        raise ConfigError("dataset configuration differs from the packed label schema")
    return data_root, mapped


def _dataset_record(mapped) -> dict:
    recipe = mapped.plan.record()
    return {
        "dataset_id": mapped.plan.dataset_id, "manifest_sha256": mapped.manifest_sha256,
        "split": mapped.plan.namespace.split, "core_count": mapped.plan.core_count,
        "label_schema": recipe["label_schema"], "feature_schema": recipe["feature_schema"],
        "bitorder": recipe["bitorder"], "shots": mapped.plan.total_rows,
    }


def _write_array(path: Path, values) -> dict:
    import numpy as np

    with atomic_file(path) as file:
        np.save(file, values, allow_pickle=False)
    return {**file_integrity(path), "shape": list(values.shape), "dtype": values.dtype.name}


def _query_record(view) -> dict:
    metadata = view.metadata
    return {
        "core_index": metadata.target.core_index,
        "target_first_round": metadata.target.first_round, "target_last_round": metadata.target.last_round,
        "first_observed_round": metadata.first_observed_round, "last_observed_round": metadata.last_observed_round,
        "release": {"round": metadata.release.round, "phase": metadata.release.phase.name.lower()},
        "preparation": metadata.preparation, "destructive_readout": metadata.destructive_readout,
        "buffers": None if metadata.buffers is None else asdict(metadata.buffers),
        "detector_count": len(view.detectors),
    }


def evaluate_workflow(
    config: ResolvedConfig, *, overrides: tuple[str, ...] = (), sweep_index: int | None = None,
) -> Path:
    check_paired_execution("evaluate", config)
    configure_cpu_runtime(config.runtime, config.data)
    import numpy as np
    import stim

    from .checkpoints import load_checkpoint
    from .evaluation import CheckpointProvenance, FrozenEvaluator, MatchingReference, compose_core_decisions, model_state_sha256
    from .models.gnn_rnn import QueryDecoder
    from .windows import Buffers

    settings = config.evaluation
    data_root, mapped = _dataset(config)
    checkpoint = config.checkpoint.path
    receipt = file_integrity(checkpoint)
    payload = load_checkpoint(checkpoint)
    verify_file(checkpoint, receipt)
    provenance = CheckpointProvenance.from_payload(payload, receipt["sha256"], mapped.plan)
    model = QueryDecoder.from_checkpoint(payload["model"])
    del payload
    geometry = config.geometry
    evaluator = FrozenEvaluator(model, mapped.bundle, Buffers(geometry.left_buffer, geometry.right_buffer))
    state_digest = model_state_sha256(model)
    modes = ("G", "F", "S") if settings.id == "paired" else (settings.id,)
    conditions = {
        mode: {
            "mode": mode, "condition": provenance.condition(mode), "checkpoint": asdict(provenance),
            "model_state_sha256": state_digest, "queries": [_query_record(view) for view in evaluator.views(mode)],
        }
        for mode in modes
    }
    matching = None
    if settings.matching:
        dem_path = data_root / "global.dem"
        dem_receipt = file_integrity(dem_path)
        dem = stim.DetectorErrorModel(dem_path.read_text())
        verify_file(dem_path, dem_receipt)
        matching = MatchingReference(dem, detector_count=mapped.plan.detector_count)
        conditions["MWPM"] = {
            "mode": "MWPM", "condition": "MWPM", "checkpoint": None, "model_state_sha256": None, "queries": [],
            "baseline": {
                "implementation": "pymatching", "version": matching.version, "dem": dem_receipt,
                "decompose_errors": True, "num_observables": dem.num_observables,
                "num_detectors": dem.num_detectors, "options": "PyMatching defaults",
            },
        }
    destination = _run_path(config, sweep_index)
    with staged_directory(destination) as staging:
        inventory = []
        (staging / "shards").mkdir()
        for shard in mapped.shards:
            shard_dir = staging / "shards" / str(shard.index)
            shard_dir.mkdir(parents=True)
            count = len(shard.shot_ids)
            logits = {mode: np.empty((count, len(evaluator.views(mode))), dtype=np.float32) for mode in modes}
            baselines = np.empty((count, 1), dtype=bool) if matching is not None else None
            for start in range(0, count, settings.batch_size):
                rows = slice(start, min(count, start + settings.batch_size))
                shots = shard.read(rows).shots
                for mode in modes:
                    logits[mode][rows] = evaluator.logits(mode, shots.detectors, shots.shot_ids)
                if matching is not None:
                    baselines[rows] = matching.decisions(shots.detectors)
            files = {"shot_ids.npy": _write_array(shard_dir / "shot_ids.npy", shard.shot_ids)}
            decisions = {mode: scores >= 0 for mode, scores in logits.items()}
            if matching is not None:
                decisions["MWPM"] = baselines
            for mode, bits in decisions.items():
                name = f"{mode}.decisions.npy"
                files[name] = {
                    **_write_array(shard_dir / name, np.packbits(bits, axis=1, bitorder="little")),
                    "valid_bits": bits.shape[1],
                }
                if mode in logits:
                    name = f"{mode}.logits.npy"
                    files[name] = _write_array(shard_dir / name, logits[mode])
                if mode in ("F", "S"):
                    name = f"{mode}.final.npy"
                    final = compose_core_decisions(bits).astype(np.uint8)[:, None]
                    files[name] = {**_write_array(shard_dir / name, final), "valid_bits": 1}
            inventory.append({"index": shard.index, "rows": count, "files": files})
        if model_state_sha256(model) != state_digest:
            raise ArtifactError("frozen evaluation changed model parameters or buffers")
        write_json(staging / "manifest.json", {
            "schema_version": PREDICTION_SCHEMA, "status": "complete", "data": _dataset_record(mapped),
            "conditions": conditions, "shards": inventory, "resolved_config": config.to_record(),
            "overrides": list(overrides), "environment": _environment_record(),
        })
    return destination / "manifest.json"


def _read_manifest(path: Path, mapped) -> dict:
    from .continuation import is_digest
    from .evaluation import CheckpointProvenance
    from .windows import Buffers, build_observation_layout, select_window

    record = read_json(path)
    if record.get("schema_version") != PREDICTION_SCHEMA or record.get("status") != "complete":
        raise ArtifactError("incomplete or incompatible prediction manifest")
    if record.get("data") != _dataset_record(mapped):
        raise ArtifactError("prediction dataset identity or label/feature schema differs from the selected dataset")
    try:
        if len(record["shards"]) != len(mapped.shards) or not record["conditions"]:
            raise ArtifactError("prediction manifest is missing shards or conditions")
        resolved = record["resolved_config"]
        settings = resolved["evaluation"]
        expected_modes = set("GFS") if settings["id"] == "paired" else {settings["id"]}
        if settings["matching"]:
            expected_modes.add("MWPM")
        if (
            record["conditions"].keys() != expected_modes
            or EVALUATION_PRESETS.get(settings["id"]) != (settings["query"], settings["context"], settings["aggregation"])
            or settings["threshold"] != 0.5 or not settings["reset_recurrent_state_per_query"]
            or settings["update_model_state"]
        ):
            raise ArtifactError("prediction inference settings differ from the frozen G/F/S policy")
        geometry = resolved["geometry"]
        layout = build_observation_layout(mapped.bundle)
        buffers = Buffers(geometry["left_buffer"], geometry["right_buffer"])
        for mode, entry in record["conditions"].items():
            if entry["mode"] != mode:
                raise ArtifactError("prediction inference mode differs from its manifest identity")
            if mode == "MWPM":
                baseline = entry["baseline"]
                if (
                    entry["condition"] != "MWPM" or entry["checkpoint"] is not None or entry["queries"]
                    or entry["model_state_sha256"] is not None or baseline["implementation"] != "pymatching"
                    or baseline["num_observables"] != 1 or baseline["num_detectors"] != mapped.plan.detector_count
                ):
                    raise ArtifactError("invalid matching baseline provenance")
                continue
            provenance = CheckpointProvenance(**entry["checkpoint"])
            targets = (layout.global_query,) if mode == "G" else layout.queries
            queries = [_query_record(select_window(layout, target, buffers if mode == "S" else None)) for target in targets]
            if (
                provenance.condition(mode) != entry["condition"] or not is_digest(entry["model_state_sha256"])
                or entry["queries"] != queries
            ):
                raise ArtifactError("prediction condition, query partition or checkpoint provenance differs")
        return record
    except ArtifactError:
        raise
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise ArtifactError("invalid prediction manifest contents") from exc


def _read_array(path: Path, entry: dict, *, dtype: str, shape: tuple):
    import numpy as np

    verify_file(path, {key: entry[key] for key in ("bytes", "sha256")})
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.dtype(dtype) or array.shape != shape or entry["dtype"] != dtype or entry["shape"] != list(shape):
        raise ArtifactError(f"prediction array dimensions or dtype differ for {path.name}")
    return array


def _read_shard(path: Path, record: dict, shard):
    import numpy as np

    from .analysis import require_paired_ids
    from .evaluation import compose_core_decisions

    try:
        entry = record["shards"][shard.index]
        row_count = len(shard.shot_ids)
        if (entry["index"], entry["rows"]) != (shard.index, row_count):
            raise ArtifactError("missing, duplicated or reordered prediction shards")
        files = entry["files"]
        directory = path.parent / "shards" / str(shard.index)
        ids = _read_array(directory / "shot_ids.npy", files["shot_ids.npy"], dtype="int64", shape=(row_count,))
        require_paired_ids(shard.shot_ids, ids)
        for mode, condition in record["conditions"].items():
            width = record["data"]["core_count"] if mode in ("F", "S") else 1
            name = f"{mode}.decisions.npy"
            packed = _read_array(directory / name, files[name], dtype="uint8", shape=(row_count, (width + 7) // 8))
            if files[name]["valid_bits"] != width or (width % 8 and np.any(packed[:, -1] >> (width % 8))):
                raise ArtifactError("prediction valid bits or padding differ")
            decisions = np.unpackbits(packed, count=width, axis=1, bitorder="little").astype(bool)
            scores = None
            if mode != "MWPM":
                name = f"{mode}.logits.npy"
                scores = _read_array(directory / name, files[name], dtype="float32", shape=(row_count, width))
                if not np.isfinite(scores).all() or not np.array_equal(decisions, scores >= 0):
                    raise ArtifactError("prediction decisions differ from the recorded finite logits and threshold")
            final = decisions[:, 0]
            if mode in ("F", "S"):
                name = f"{mode}.final.npy"
                composed = _read_array(directory / name, files[name], dtype="uint8", shape=(row_count, 1))
                final = compose_core_decisions(decisions)
                if files[name]["valid_bits"] != 1 or not np.array_equal(final, composed[:, 0]):
                    raise ArtifactError("final predictions disagree with the recorded core-decision XOR")
            yield condition["condition"], decisions, scores, final
    except ArtifactError:
        raise
    except (ValueError, KeyError, TypeError, EOFError, OSError) as exc:
        raise ArtifactError(f"invalid prediction rows in shard {shard.index}: {exc}") from exc


def _check_comparisons(conditions: dict, names: list[str]) -> list[tuple[str, str, str]]:
    requested = []
    namespaces = {
        (cp["training_seed"], cp["training_replicate"], cp["training_family"])
        for item in conditions.values() if (cp := item["checkpoint"]) is not None
    }
    if len(namespaces) > 1:
        raise ArtifactError("analyze one training seed and replicate at a time")
    primary = {
        (conditions[mode]["checkpoint"]["sha256"], conditions[mode]["model_state_sha256"])
        for mode in ("G", "F", "S") if mode in conditions
    }
    if len(primary) > 1:
        raise ArtifactError("primary G/F/S contrasts require the same checkpoint and model state")
    for name in names:
        parts = name.split("_minus_")
        if len(parts) != 2:
            raise ConfigError(f"invalid contrast {name!r}; use A_minus_B")
        first, second = ("S+" if part == "Splus" else part for part in parts)
        if first == second or first not in conditions or second not in conditions:
            raise ArtifactError(f"contrast {name} requires two distinct recorded conditions")
        left = conditions[first]["checkpoint"]
        right = conditions[second]["checkpoint"]
        if {first, second} <= {"S+", "W", "D"} and left["parent_sha256"] != right["parent_sha256"]:
            raise ArtifactError("adapted comparisons require a common base checkpoint")
        if conditions[first]["mode"] == conditions[second]["mode"] == "S" and (
            conditions[first]["queries"] != conditions[second]["queries"]
        ):
            raise ArtifactError("window comparisons require identical deployment queries and buffers")
        if {first, second} <= {"S+", "W", "D"}:
            for field in ("initial_model_sha256", "continuation_controls_sha256", "continuation_input_sha256"):
                if left.get(field) is None or left[field] != right.get(field):
                    raise ArtifactError("adapted comparisons require matched initialization, controls, and trajectory/core draws")
        requested.append((name, first, second))
    return requested


def _write_csv(path: Path, rows: list[dict]) -> None:
    contents = StringIO(newline="")
    writer = csv.DictWriter(contents, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    with atomic_file(path) as file:
        file.write(contents.getvalue().encode("utf-8"))


def analyze_workflow(
    config: ResolvedConfig, *, overrides: tuple[str, ...] = (), sweep_index: int | None = None,
) -> Path:
    check_paired_execution("analyze", config)
    configure_cpu_runtime(config.runtime, config.data)
    import numpy as np

    from .analysis import (
        AnalysisPolicy, CoreDiagnostics, PairedCounts, _count_pairs, binomial_interval, core_disagreements, summarize_pair,
    )
    from .evaluation import compose_core_decisions

    _, mapped = _dataset(config)
    settings = config.analysis
    policy = AnalysisPolicy(
        settings.bootstrap_replicates, settings.bootstrap_seed, settings.sparse_rule,
        settings.ratio_min_denominator_failures,
    )
    inputs = []
    for name in settings.predictions:
        path = Path(name)
        receipt = file_integrity(path)
        inputs.append((path, _read_manifest(path, mapped), receipt))
    conditions = {}
    for _, manifest, _ in inputs:
        for entry in manifest["conditions"].values():
            label = entry["condition"]
            if label in conditions:
                raise ArtifactError(f"duplicate prediction condition {label}")
            conditions[label] = entry
    comparisons = _check_comparisons(conditions, settings.contrasts)
    core = {label: CoreDiagnostics() for label, entry in conditions.items() if entry["mode"] in ("F", "S")}
    failures = dict.fromkeys(conditions, 0)
    losses = {label: np.zeros(12 if label in core else 1) for label, entry in conditions.items() if entry["mode"] != "MWPM"}
    paired = {name: np.zeros(4, dtype=np.int64) for name, _, _ in comparisons}
    for shard in mapped.shards:
        labels = np.unpackbits(shard.global_labels, axis=1, count=1, bitorder="little").astype(bool)
        core_labels = np.unpackbits(shard.core_labels, axis=1, count=12, bitorder="little").astype(bool)
        if not np.array_equal(compose_core_decisions(core_labels), labels[:, 0]):
            raise ArtifactError("dataset core parity differs from global labels")
        shard_failures = {}
        for path, manifest, _ in inputs:
            for condition, decisions, scores, final in _read_shard(path, manifest, shard):
                target = labels
                if condition in core:
                    core[condition].add(core_disagreements(decisions, core_labels, labels[:, 0]))
                    target = core_labels
                error = final ^ labels[:, 0]
                shard_failures[condition] = error
                failures[condition] += int(error.sum())
                if scores is not None:
                    signed = np.where(target, -scores, scores).astype(np.float64)
                    losses[condition] += np.logaddexp(0, signed).sum(axis=0)
        for name, first, second in comparisons:
            paired[name] += _count_pairs(shard_failures[first], shard_failures[second]).categories

    for path, _, receipt in inputs:
        verify_file(path, receipt)
    shots = mapped.plan.total_rows
    rates = {}
    for label, count in failures.items():
        rates[label] = {
            "shots": shots, "failures": count, "ler": count / shots,
            "wilson_interval": list(binomial_interval(count, shots, confidence=0.95, method="wilson")),
        }
        if label in losses:
            rates[label]["mean_query_cross_entropy"] = (losses[label] / shots).tolist()
        if label in core:
            rates[label]["core_diagnostics"] = core[label].report()
    rng = policy.rng()
    contrasts = {
        name: {"first": first, "second": second, **summarize_pair(PairedCounts(*map(int, paired[name])), policy, rng)}
        for name, first, second in comparisons
    }
    decomposition = None
    if set(PRIMARY_CONTRASTS) <= contrasts.keys():
        def net(name):
            row = contrasts[name]
            return row["n10"] - row["n01"]

        direct = net("S_minus_G")
        sum_of_parts = net("F_minus_G") + net("S_minus_F")
        if direct != sum_of_parts:
            raise ArtifactError("primary paired contrast decomposition disagrees at the count level")
        decomposition = {"direct_numerator": direct, "sum_of_parts_numerator": sum_of_parts, "denominator": shots}
    destination = _run_path(config, sweep_index)
    with staged_directory(destination) as staging:
        _write_csv(staging / "rates.csv", [
            {"condition": label, "shots": row["shots"], "failures": row["failures"], "ler": row["ler"],
             "lower": row["wilson_interval"][0], "upper": row["wilson_interval"][1]} for label, row in rates.items()
        ])
        _write_csv(staging / "contrasts.csv", [
            {"contrast": name, "n10": row["n10"], "n01": row["n01"], "difference": row["difference"],
             "lower": row["interval"][0], "upper": row["interval"][1], "interval_method": row["interval_method"]}
            for name, row in contrasts.items()
        ])
        write_json(staging / "report.json", {
            "schema_version": 1, "status": "complete", "data": _dataset_record(mapped), "conditions": conditions,
            "rates": rates, "contrasts": contrasts, "primary_decomposition": decomposition,
            "analysis": {**policy.record(), "contrast_order": settings.contrasts},
            "inputs": [{"path": str(path), **receipt} for path, _, receipt in inputs],
            "resolved_config": config.to_record(), "overrides": list(overrides), "environment": _environment_record(),
        })
    return destination / "report.json"
