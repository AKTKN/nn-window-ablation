from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

from nn_window_ablation.features import attach_supervision, batch_features
from nn_window_ablation.models.gnn_rnn import QueryDecoder
from nn_window_ablation.models.objective import query_loss
from nn_window_ablation.windows import Buffers, select_window
from tests.p3_checks import examples_for, model_inputs, pilot


def _assert_logits(actual, expected):
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)


def test_p3_d5_model_contract(pilot):
    _, layout, geometry, shots = pilot
    torch.manual_seed(111)
    model = QueryDecoder(5)
    assert [conv.in_channels for conv in model.gnn.convs] == [3, 64]
    assert [conv.out_channels for conv in model.gnn.convs] == [64, 256]
    assert model.rnn.hidden_size == 256 and model.rnn.num_layers == 4
    assert model.rnn.dropout == 0 and model.head.out_features == 1
    assert len(model.aggregators) == 1 and model.aggregators[0][0].kernel_size == (2, 2)
    assert len([layer for layer in model.modules() if isinstance(layer, nn.GRU)]) == 1
    assert all(parameter.requires_grad for parameter in model.parameters())

    global_examples = examples_for(select_window(layout, layout.global_query), geometry, shots)
    indices = torch.tensor([0, 11, 5])
    core_examples = tuple(
        examples_for(select_window(layout, layout.core_query(index)), geometry, shots, slice(row, row + 1))[0]
        for row, index in enumerate(indices.tolist())
    )
    inputs = model_inputs(global_examples + core_examples)
    assert inputs.sequence_lengths == (60,) * 6
    assert len(inputs.node_features) > 0
    assert len(inputs.node_graph.unique()) < 6 * 60 * 4  # Empty graphs also reach the shared empty vector.
    targets = attach_supervision(batch_features(global_examples + core_examples), shots)
    np.testing.assert_array_equal(targets.labels[:3], shots.global_labels)
    np.testing.assert_array_equal(targets.labels[3:], shots.core_labels[np.arange(3), indices.numpy()])

    logits = model(inputs)
    assert logits.shape == (6,) and torch.isfinite(logits).all()
    loss = query_loss(
        logits[:3], torch.from_numpy(shots.global_labels), logits[3:], torch.from_numpy(shots.core_labels),
        core_indices=indices,
    )
    assert all(term.isfinite() and term.item() > 0 for term in (loss.total, loss.global_term, loss.core_term))
    component_parameters = (
        tuple(model.gnn.convs.parameters()), (model.gnn.empty,),
        tuple(model.aggregators.parameters()), tuple(model.query_projection.parameters()),
        tuple(model.rnn.parameters()), tuple(model.head.parameters()),
    )
    trainable = tuple(parameter for component in component_parameters for parameter in component)
    for term in (loss.global_term, loss.core_term):
        gradients = torch.autograd.grad(term, trainable, retain_graph=True)
        offset = 0
        for component in component_parameters:
            current = gradients[offset:offset + len(component)]
            assert all(gradient.isfinite().all() for gradient in current)
            assert sum(gradient.abs().sum().item() for gradient in current) > 0
            offset += len(component)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    before = model.query_projection.weight.detach().clone()
    loss.total.backward()
    assert all(parameter.grad is not None and parameter.grad.isfinite().all() for parameter in model.parameters())
    optimizer.step()
    assert not torch.equal(before, model.query_projection.weight)
    assert all(parameter.isfinite().all() for parameter in model.parameters())


def test_p3_d5_gfs_batching_state_and_checkpoint_contract(pilot, tmp_path):
    _, layout, geometry, shots = pilot
    torch.manual_seed(240)
    model = QueryDecoder(5).eval()
    target = layout.core_query(10)
    row = slice(0, 1)
    global_query, = examples_for(select_window(layout, layout.global_query), geometry, shots, row)
    full_core, = examples_for(select_window(layout, target), geometry, shots, row)
    window, = examples_for(select_window(layout, target, Buffers(5, 5)), geometry, shots, row)
    final, = examples_for(select_window(layout, layout.core_query(11), Buffers(0, 0)), geometry, shots, row)
    assert full_core.metadata.query.target == window.metadata.query.target
    assert tuple(len(example.metadata.groups) for example in (global_query, full_core, window, final)) == (60, 60, 14, 5)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    with torch.inference_mode():
        independent = torch.cat([model(model_inputs((query,))) for query in (global_query, full_core, window, final)])
        a = model(model_inputs((window,)))
        model(model_inputs((global_query, final)))
        after_b = model(model_inputs((window,)))
        torch.testing.assert_close(a, after_b, atol=0, rtol=0)
        reordered = model(model_inputs((window, final, full_core, global_query)))
        _assert_logits(reordered, independent[[2, 3, 1, 0]])
        mixed = model(model_inputs((global_query, full_core, window, final)))
        _assert_logits(mixed, independent)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], atol=0, rtol=0)

    path = tmp_path / "fixture.pt"
    torch.save(model.checkpoint_payload(), path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["architecture"]["query_schema_version"] == 1
    assert payload["architecture"]["spatial_grids_inner_first"] == [2]
    rng_before = torch.get_rng_state()
    restored = QueryDecoder.from_checkpoint(payload).eval()
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert restored.architecture() == model.architecture()
    with torch.inference_mode():
        torch.testing.assert_close(
            restored(model_inputs((global_query, full_core, window, final))), mixed, atol=0, rtol=0,
        )
    corrupt = deepcopy(payload["architecture"])
    corrupt["query_fields"].reverse()
    with pytest.raises(ValueError, match="architecture or query metadata"):
        QueryDecoder.from_checkpoint({**payload, "architecture": corrupt})
