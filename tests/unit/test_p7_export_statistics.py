import json
from statistics import NormalDist

import numpy as np
import pytest
from scipy.stats import beta

from nn_window_ablation.archive import extract_bundle
from nn_window_ablation.artifacts import ArtifactError
from nn_window_ablation.export import export_bundle, reanalyze_bundle
from tests.p7_checks import digest, truth_artifacts, write_json


def _exact(successes, total):
    low = beta.ppf(0.0125, successes, total - successes + 1) if successes else 0
    high = beta.ppf(0.9875, successes + 1, total - successes) if successes < total else 1
    return low, high


def test_p7_reconstructs_counts_percentile_and_exact_limits_without_checkpoint(truth_artifacts, tmp_path):
    dataset, _, checkpoint, report, failures = truth_artifacts
    source = export_bundle(dataset, report, tmp_path / "port.tar.gz")
    root = extract_bundle(source, tmp_path / "relocated")
    result = json.loads(reanalyze_bundle(root, tmp_path / "again").read_text())
    old = json.loads(report.read_text())
    assert result["rates"] == old["rates"] and result["contrasts"] == old["contrasts"]
    assert not checkpoint.exists()
    rng = np.random.Generator(np.random.PCG64(17))
    methods = set()
    for name in old["analysis"]["contrast_order"]:
        row = result["contrasts"][name]
        pairs = list(zip(failures[row["first"]], failures[row["second"]], strict=True))
        n00, n01, n10, n11 = [pairs.count(category) for category in [(0, 0), (0, 1), (1, 0), (1, 1)]]
        assert [row[key] for key in ("n00", "n01", "n10", "n11")] == [n00, n01, n10, n11]
        assert row["difference"] == (n10 - n01) / 4
        methods.add(row["interval_method"])
        if n10 + n01 <= 1:
            low10, high10 = _exact(n10, 4)
            low01, high01 = _exact(n01, 4)
            assert row["interval"] == pytest.approx([low10 - high01, high10 - low01], abs=1e-12)
        else:
            table = rng.multinomial(4, [n00 / 4, n01 / 4, n10 / 4, n11 / 4], size=41)
            assert row["interval"] == pytest.approx(np.percentile((table[:, 2] - table[:, 1]) / 4, [2.5, 97.5]))
    assert methods == {"conservative_exact_discordances", "paired_bootstrap_percentile"}
    for name, bits in failures.items():
        rate = bits.sum() / 4
        z = NormalDist().inv_cdf(0.975)
        denominator = 1 + z**2 / 4
        center = (rate + z**2 / 8) / denominator
        spread = z * np.sqrt(rate * (1 - rate) / 4 + z**2 / 64) / denominator
        assert result["rates"][name]["wilson_interval"] == pytest.approx([center - spread, center + spread])


@pytest.mark.parametrize(("damage", "message"), [
    ("predictions", "size or checksum mismatch"), ("labels", "missing or unexpected files"),
    ("condition", "condition, query partition or checkpoint provenance differs"),
])
def test_p7_reanalysis_rejects_modified_scientific_content(truth_artifacts, tmp_path, damage, message):
    dataset, _, _, report, _ = truth_artifacts
    source = export_bundle(dataset, report, tmp_path / "port.tar.gz")
    root = extract_bundle(source, tmp_path / "relocated")
    if damage == "labels":
        (root / "dataset/shards/0/core_labels.npy").unlink()
    elif damage == "predictions":
        path = root / "predictions/0/shards/0/G.decisions.npy"
        path.write_bytes(path.read_bytes()[:-1])
    else:
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        path = root / manifest["predictions"][0]
        record = json.loads(path.read_text())
        record["conditions"]["S"]["condition"] = "W"
        write_json(path, record)
        manifest["files"][manifest["predictions"][0]] = digest(path)
        receipt = root / manifest["report"]
        record = json.loads(receipt.read_text())
        record["inputs"][0].update(digest(path))
        write_json(receipt, record)
        manifest["files"][manifest["report"]] = digest(receipt)
        write_json(manifest_path, manifest)
    with pytest.raises(ArtifactError, match=message):
        reanalyze_bundle(root, tmp_path / "again")
    assert not (tmp_path / "again").exists()
