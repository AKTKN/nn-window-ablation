import json

import numpy as np
import pymatching
import pytest
import stim

from nn_window_ablation.artifacts import file_integrity
from nn_window_ablation.evaluation import MatchingReference
from nn_window_ablation.paired_workflows import evaluate_workflow
from tests.p5_checks import frozen_pilot


def test_p5_matching_uses_original_global_dem_and_the_same_detector_rows(frozen_pilot):
    resolve, dataset, root, _ = frozen_pilot
    path = evaluate_workflow(resolve(
        "run_name=matching", "data.id=held", "data.split=test", "evaluation=G",
        "evaluation.matching=true", "checkpoint.path=runs/base/checkpoint.pt",
    ))
    manifest = json.loads(path.read_text())
    record = manifest["conditions"]["MWPM"]
    assert record["checkpoint"] is None and record["condition"] == "MWPM"
    dem = stim.DetectorErrorModel((root / "global.dem").read_text())
    assert (dem.num_observables, dem.num_detectors) == (1, dataset.plan.detector_count)
    assert record["baseline"]["dem"] == file_integrity(root / "global.dem")
    assert record["baseline"]["version"] == pymatching.__version__
    assert record["baseline"]["num_observables"] == 1 and record["baseline"]["decompose_errors"]
    independent = pymatching.Matching.from_detector_error_model(dem)
    for shard in dataset.shards:
        rows = shard.read().shots
        stored = path.parent / "shards" / str(shard.index)
        np.testing.assert_array_equal(np.load(stored / "shot_ids.npy"), rows.shot_ids)
        decisions = np.unpackbits(np.load(stored / "MWPM.decisions.npy"), count=1, axis=1, bitorder="little")
        expected = np.array([independent.decode(detectors) for detectors in rows.detectors])
        assert expected.shape == (len(rows.shot_ids), 1)
        np.testing.assert_array_equal(decisions, expected)
    wrong_dem = stim.DetectorErrorModel(str(dem) + "\nlogical_observable L1")
    with pytest.raises(ValueError, match="global-only DEM"):
        MatchingReference(wrong_dem, detector_count=dataset.plan.detector_count)
