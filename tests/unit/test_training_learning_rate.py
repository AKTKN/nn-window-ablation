import pytest
import torch

from nn_window_ablation.training import ClippedStepLR


@pytest.mark.parametrize("optimizer_type", [torch.optim.SGD, torch.optim.Adam])
@pytest.mark.parametrize("initial_rate", [0.001, 0.02])
def test_step_decay_clips_at_one_tenth_of_initial_rate(optimizer_type, initial_rate):
    parameter = torch.nn.Parameter(torch.tensor(-2.0, dtype=torch.float64))
    optimizer = optimizer_type([parameter], lr=initial_rate, foreach=False)
    scheduler = ClippedStepLR(optimizer, step_size=2, gamma=0.2)
    factors = [1.0, 1.0, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]
    for factor in factors:
        expected_rate = initial_rate * factor
        assert optimizer.param_groups[0]["lr"] == pytest.approx(expected_rate)
        assert scheduler.get_last_lr() == pytest.approx([expected_rate])
        before = parameter.item()
        optimizer.zero_grad()
        parameter.backward()
        optimizer.step()
        # A constant unit gradient gives this independent recurrence for both optimizers.
        epsilon = optimizer.defaults.get("eps", 0.0)
        assert parameter.item() == pytest.approx(before - expected_rate / (1 + epsilon), abs=1e-14)
        scheduler.step()


def test_parameter_groups_keep_distinct_initial_rate_floors():
    parameters = [torch.nn.Parameter(torch.tensor(1.0)) for _ in range(2)]
    optimizer = torch.optim.SGD([
        {"params": [parameters[0]], "lr": 0.001},
        {"params": [parameters[1]], "lr": 0.02},
    ])
    scheduler = ClippedStepLR(optimizer, step_size=1, gamma=0.01)
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        assert scheduler.get_last_lr() == [0.0001, 0.002]
        assert [group["lr"] for group in optimizer.param_groups] == [0.0001, 0.002]
    optimizer.step()
    with pytest.warns(UserWarning, match="epoch parameter"):
        scheduler.step(epoch=100)
    assert scheduler.get_last_lr() == [0.0001, 0.002]
    assert [group["lr"] for group in optimizer.param_groups] == [0.0001, 0.002]
