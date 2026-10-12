import pytest
import torch
from melody_rl.objective import group_advantages, grpo_loss


def test_no_variance_no_policy_signal():
    assert torch.equal(group_advantages(torch.tensor([1., 1., 1.])), torch.zeros(3))


def test_advantages_have_zero_mean_and_unit_population_variance():
    result = group_advantages(torch.tensor([0., 0.5, 1.]))
    assert result.mean().item() == pytest.approx(0, abs=1e-6)
    assert result.std(unbiased=False).item() == pytest.approx(1, abs=1e-5)


def test_positive_advantage_increases_likelihood_and_negative_decreases():
    for advantage in (-1., 1.):
        logps = torch.tensor([-2., -3.], requires_grad=True)
        loss, stats = grpo_loss(logps, logps.detach(), logps.detach(), advantage, beta=0)
        loss.backward()
        assert all((logps.grad * advantage < 0).tolist())
        assert stats["kl"] == 0


def test_clipping_stops_positive_advantage_above_upper_bound():
    logps = torch.tensor([-1.], requires_grad=True)
    loss, _ = grpo_loss(logps, torch.tensor([-2.]), logps.detach(), 1., beta=0)
    loss.backward()
    assert logps.grad.item() == 0


def test_reference_kl_nonnegative():
    values = torch.tensor([-0.1, -3.])
    _, stats = grpo_loss(values, values, torch.tensor([-1., -2.]), 0.)
    assert stats["kl"] > 0
