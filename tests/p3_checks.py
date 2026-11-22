import pytest
import torch

from nn_window_ablation.config import resolve_config
from nn_window_ablation.features import batch_features, build_feature_geometry, build_features
from nn_window_ablation.models.gnn_rnn import prepare_model_batch
from nn_window_ablation.simulation import CircuitSpec, build_circuit_bundle, sample_shots
from nn_window_ablation.windows import build_observation_layout, crop_observations


@pytest.fixture
def pilot(compose_settings, tmp_path):
    config = resolve_config(compose_settings("runtime=test_cpu"), invocation_dir=tmp_path)
    physical = config.geometry
    spec = CircuitSpec(physical.distance, physical.rounds, physical.core_length, config.noise.probability)
    assert spec == CircuitSpec(5, 60, 5, 0.003)
    assert (physical.left_buffer, physical.right_buffer, physical.width) == (5, 5, 15)
    assert (config.runtime.processes, config.runtime.intraop_threads, config.runtime.interop_threads) == (1, 1, 1)
    assert (config.data.num_workers, config.data.sampling_processes) == (0, 1)
    assert torch.get_num_threads() == torch.get_num_interop_threads() == 1
    bundle = build_circuit_bundle(spec)
    layout = build_observation_layout(bundle)
    shots = sample_shots(bundle, 3, seed=307, first_shot_id=124, include_snapshots=True)
    return bundle, layout, build_feature_geometry(layout), shots


def examples_for(view, geometry, shots, rows=slice(None)):
    return build_features(crop_observations(view, shots.detectors[rows], shots.shot_ids[rows]), geometry)


def model_inputs(examples):
    return prepare_model_batch(batch_features(examples), distance=5)
