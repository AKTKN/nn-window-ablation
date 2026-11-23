"""Export terminal paired results and replay their statistics from archived rows."""

from importlib import metadata
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from .archive import BUNDLE_SCHEMA, MANIFEST, member_path, pack_bundle, verify_bundle
from .artifacts import ArtifactError, file_integrity, read_json, staged_directory, verify_file, write_json


def _complete(path: Path) -> dict:
    record = read_json(path)
    if record.get("schema_version") != 1 or record.get("status") != "complete":
        raise ArtifactError(f"incomplete, failed, or incompatible run record {path.name}")
    return record


def _copy(staging: Path, name: str, source: Path, files: dict, expected: dict | None = None) -> None:
    target = staging / member_path(name)
    if name in files:
        raise ArtifactError(f"duplicate exported file {name}")
    if not source.is_file():
        raise ArtifactError(f"missing required export file {source}")
    expected = file_integrity(source) if expected is None else expected
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    verify_file(target, expected)
    files[name] = expected


def _file_receipt(entry: dict) -> dict:
    return {key: entry[key] for key in ("bytes", "sha256")}


def _copy_dataset(staging: Path, name: str, source: Path, mapped, files: dict, *, detectors: bool) -> None:
    from .data import ARRAY_FILES

    manifest_path = source / MANIFEST
    manifest = _complete(manifest_path)
    receipt = file_integrity(manifest_path)
    if receipt["sha256"] != mapped.manifest_sha256:
        raise ArtifactError("dataset manifest changed after verification")
    _copy(staging, f"{name}/{MANIFEST}", manifest_path, files, receipt)
    if detectors:
        for filename, receipt in manifest["physical"].items():
            member_path(filename)
            _copy(staging, f"{name}/{filename}", source / filename, files, receipt)
    for shard in mapped.shards:
        directory = f"shards/{shard.index}"
        shard_path = source / directory / "shard.json"
        relative = f"{name}/{directory}/shard.json"
        _copy(staging, relative, shard_path, files, manifest["shards"][shard.index]["manifest"])
        arrays = read_json(staging / relative)["arrays"]
        for role, filename in ARRAY_FILES.items():
            if detectors or role != "detectors":
                _copy(staging, f"{name}/{directory}/{filename}", shard_path.parent / filename, files, _file_receipt(arrays[role]))


def _copy_predictions(staging: Path, name: str, path: Path, prediction: dict, files: dict, receipt: dict) -> None:
    _copy(staging, f"{name}/{MANIFEST}", path, files, receipt)
    for shard in prediction["shards"]:
        for filename, array in shard["files"].items():
            if len(member_path(filename).parts) != 1:
                raise ArtifactError("invalid prediction artifact filename")
            relative = f"shards/{shard['index']}/{filename}"
            _copy(staging, f"{name}/{relative}", path.parent / relative, files, _file_receipt(array))


def _summarize(report: dict, mapped, inputs: list) -> dict:
    from .analysis import AnalysisPolicy
    from .paired_workflows import _dataset_record, summarize_predictions

    try:
        if report["data"] != _dataset_record(mapped):
            raise ArtifactError("report dataset identity or label provenance differs")
        recorded = report["analysis"]
        policy = AnalysisPolicy(
            recorded["bootstrap_replicates"], recorded["bootstrap_seed"], recorded["sparse_rule"],
            recorded["ratio_min_denominator_failures"],
        )
        order = recorded["contrast_order"]
        if type(order) is not list or not order or len(order) != len(set(order)):
            raise ArtifactError("invalid recorded contrast order")
        if recorded != {**policy.record(), "contrast_order": order}:
            raise ArtifactError("unsupported saved analysis policy")
        for package in ("numpy", "scipy"):
            if report["environment"]["packages"][package] != metadata.version(package):
                raise ArtifactError(f"recorded analysis requires its original {package} version")
        summary = summarize_predictions(mapped, inputs, policy, order)
        for key, value in summary.items():
            if value != report[key]:
                raise ArtifactError(f"recorded {key} disagrees with reconstructed statistics or provenance")
        return summary
    except ArtifactError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ArtifactError(f"incomplete or incompatible analysis evidence: {exc}") from exc


def _run_evidence(staging: Path, inputs: list, mapped, files: dict, *, checkpoints: bool) -> list[dict]:
    seen = {}
    for _, manifest, _ in inputs:
        path = Path(manifest["resolved_config"]["checkpoint"]["path"])
        entries = [item for item in manifest["conditions"].values() if item["checkpoint"] is not None]
        if not entries:
            continue
        digests = {(item["checkpoint"]["sha256"], item["model_state_sha256"]) for item in entries}
        if len(digests) != 1:
            raise ArtifactError("one prediction run must bind one checkpoint and model state")
        digest, model_digest = digests.pop()
        recorded_provenance = entries[0]["checkpoint"]
        if any(item["checkpoint"] != recorded_provenance for item in entries):
            raise ArtifactError("one prediction run must bind one checkpoint provenance")
        run_path = path.with_name("run.json")
        run = _complete(run_path) if run_path.exists() else None
        if run is not None and run.get("checkpoint", {}).get("sha256") != digest:
            raise ArtifactError("training run record and prediction checkpoint identity differ")
        if digest in seen and (
            seen[digest]["model_state_sha256"] != model_digest or seen[digest]["provenance"] != recorded_provenance
        ):
            raise ArtifactError("checkpoint has conflicting model-state or provenance identities")
        if digest in seen:
            continue
        name = f"runs/{len(seen)}"
        entry = {
            "checkpoint_sha256": digest, "model_state_sha256": model_digest, "provenance": recorded_provenance,
            "run": None, "checkpoint": None,
        }
        if run is not None:
            entry["run"] = f"{name}/run.json"
            _copy(staging, entry["run"], run_path, files)
        if checkpoints:
            from .checkpoints import load_checkpoint
            from .evaluation import CheckpointProvenance
            from .continuation import model_tensors_sha256

            receipt = file_integrity(path)
            if receipt["sha256"] != digest or (run is not None and run["checkpoint"] != receipt):
                raise ArtifactError("export checkpoint byte identity differs from the prediction")
            state = load_checkpoint(path)
            provenance = CheckpointProvenance.from_payload(state, digest, mapped.plan)
            if model_tensors_sha256(state["model"]["state_dict"]) != model_digest or any(
                item["condition"] != provenance.condition(item["mode"])
                or CheckpointProvenance(**item["checkpoint"]) != provenance for item in entries
            ):
                raise ArtifactError("checkpoint condition or canonical model state differs")
            entry["checkpoint"] = f"{name}/checkpoint.pt"
            _copy(staging, entry["checkpoint"], path, files, receipt)
        seen[digest] = entry
    return list(seen.values())


def export_bundle(
    dataset: Path, report_path: Path, output: Path, *, include_detectors: bool = False,
    include_checkpoints: bool = False, training_datasets: tuple[Path, ...] = (), logs: tuple[Path, ...] = (),
) -> Path:
    from .data import open_dataset
    from .paired_workflows import _dataset_record, _read_manifest, write_analysis_tables
    from .workflows import _environment_record

    report_receipt = file_integrity(report_path)
    report = _complete(report_path)
    mapped = open_dataset(dataset)
    if mapped.plan.namespace.split not in ("test", "validation") or mapped.plan.core_count != 12:
        raise ArtifactError("export requires a finalized held-out 12-core dataset")
    inputs = []
    for source in report.get("inputs", []):
        path = Path(source["path"])
        receipt = _file_receipt(source)
        verify_file(path, receipt)
        inputs.append((path, _read_manifest(path, mapped), receipt))
    if not inputs:
        raise ArtifactError("report lacks sealed prediction inputs")
    summary = _summarize(report, mapped, inputs)

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".nnwa-export-") as directory:
        staging = Path(directory)
        files = {}
        _copy_dataset(staging, "dataset", dataset, mapped, files, detectors=include_detectors)
        expected_tables = staging / ".report-check"
        expected_tables.mkdir()
        write_analysis_tables(expected_tables, summary)
        for filename in ("rates.csv", "contrasts.csv"):
            _copy(staging, f"analysis/{filename}", report_path.with_name(filename), files, file_integrity(expected_tables / filename))
        shutil.rmtree(expected_tables)
        _copy(staging, "analysis/report.json", report_path, files, report_receipt)
        prediction_paths = []
        for index, (path, prediction, receipt) in enumerate(inputs):
            name = f"predictions/{index}"
            _copy_predictions(staging, name, path, prediction, files, receipt)
            prediction_paths.append(f"{name}/{MANIFEST}")
        runs = _run_evidence(staging, inputs, mapped, files, checkpoints=include_checkpoints)
        training_paths = []
        for index, source in enumerate(training_datasets):
            training = open_dataset(source, expected_split="train")
            name = f"training/{index}"
            _copy_dataset(staging, name, source, training, files, detectors=True)
            training_paths.append(name)
        log_paths = []
        for index, source in enumerate(logs):
            name = f"logs/{index}/{source.name}"
            _copy(staging, name, source, files)
            log_paths.append(name)
        write_json(staging / MANIFEST, {
            "schema_version": BUNDLE_SCHEMA, "status": "complete",
            "capabilities": {
                "statistics": True, "detectors": include_detectors, "checkpoints": include_checkpoints,
                "redecode": include_detectors and include_checkpoints, "training_shards": bool(training_paths),
            },
            "dataset": {"path": "dataset", **_dataset_record(mapped)}, "report": "analysis/report.json",
            "predictions": prediction_paths, "runs": runs, "training_datasets": training_paths,
            "logs": log_paths, "export_environment": _environment_record(), "files": files,
        })
        return pack_bundle(staging, output)


def reanalyze_bundle(root: Path, destination: Path) -> Path:
    """Recompute the recorded analysis using verified label and prediction rows."""
    from .data import open_labels
    from .paired_workflows import _dataset_record, _read_manifest, write_analysis_tables
    from .workflows import _environment_record

    bundle = verify_bundle(root)
    try:
        if bundle["capabilities"]["statistics"] is not True:
            raise ArtifactError("bundle does not declare statistics capability")
        data = bundle["dataset"]
        mapped = open_labels(root / member_path(data["path"]))
        if data != {"path": data["path"], **_dataset_record(mapped)} or mapped.plan.core_count != 12:
            raise ArtifactError("bundle and stored label dataset provenance differ")
        report_path = root / member_path(bundle["report"])
        report = _complete(report_path)
        predictions = bundle["predictions"]
        if not predictions or len(predictions) != len(report["inputs"]) or len(set(predictions)) != len(predictions):
            raise ArtifactError("bundle prediction inventory differs from the saved analysis")
        inputs = []
        for name, original in zip(predictions, report["inputs"], strict=True):
            path = root / member_path(name)
            receipt = _file_receipt(original)
            verify_file(path, receipt)
            inputs.append((path, _read_manifest(path, mapped, verify_queries=False), receipt))
        summary = _summarize(report, mapped, inputs)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with staged_directory(destination) as staging:
            write_analysis_tables(staging, summary)
            for filename in ("rates.csv", "contrasts.csv"):
                verify_file(staging / filename, bundle["files"][(member_path(bundle["report"]).parent / filename).as_posix()])
            write_json(staging / "report.json", {
                **report, **summary,
                "inputs": [{"path": name, **receipt} for name, (_, _, receipt) in zip(predictions, inputs, strict=True)],
                "reanalysis_environment": _environment_record(),
            })
        return destination / "report.json"
    except ArtifactError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ArtifactError(f"incomplete or incompatible bundle metadata: {exc}") from exc
