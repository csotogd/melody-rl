"""Check token alignment without downloading weights or requiring a GPU."""
from types import SimpleNamespace
import torch
from melody_rl.model import Policy


def test_completion_logps_exclude_prompt_and_predict_next_token():
    logits = torch.tensor([[[0., 1., 2.], [2., 1., 0.], [0., 3., 1.], [1., 0., 4.]]], requires_grad=True)

    class FakeModel:
        def __call__(self, **kwargs):
            return SimpleNamespace(logits=logits)

    policy = Policy.__new__(Policy)
    policy.model = FakeModel()
    ids = torch.tensor([[2, 1, 0, 1]])
    actual = policy.logps(ids, prompt_length=2)
    expected = torch.stack((logits[0, 1].log_softmax(0)[0], logits[0, 2].log_softmax(0)[1]))
    assert torch.allclose(actual, expected)
    assert actual.shape == (2,)
    actual.sum().backward()
    assert logits.grad[0, 0].abs().sum() == 0
    assert logits.grad[0, 3].abs().sum() == 0


def test_tiny_delta_kl_is_stable():
    from melody_rl.objective import grpo_loss
    values = torch.tensor([-1., -2.], requires_grad=True)
    _, stats = grpo_loss(values, values.detach(), values.detach() + 1e-6, 0.)
    assert 0 <= stats["kl"] < 1e-10
