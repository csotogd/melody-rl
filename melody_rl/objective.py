"""Explicit token-level GRPO objective, intentionally small enough to audit.

Rollouts sample the untruncated softmax at temperature=1. Recomputed log-probs
therefore match the behavior distribution. Loss is mean over tokens, then sequences.
This baseline uses population-standardized group advantages, not Dr. GRPO.
"""
import torch


def group_advantages(rewards: torch.Tensor, eps=1e-6):
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("GRPO requires at least two rewards per prompt")
    return (rewards - rewards.mean()) / (rewards.std(unbiased=False) + eps)


def grpo_loss(logps, old_logps, reference_logps, advantage, *, clip=0.2, beta=0.02):
    if not (logps.shape == old_logps.shape == reference_logps.shape) or logps.numel() == 0:
        raise ValueError("All log-prob tensors must have the same nonempty shape")
    ratio = (logps - old_logps.detach()).exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    delta = reference_logps.detach() - logps
    kl = torch.expm1(delta) - delta  # stable sampled k3 estimate near zero
    loss = (-surrogate + beta * kl).mean()
    return loss, {"kl": kl.detach().mean().item(),
                  "clip_fraction": ((ratio.detach() - 1).abs() > clip).float().mean().item(),
                  "ratio_mean": ratio.detach().mean().item()}
