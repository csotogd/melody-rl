import importlib.metadata
import json
import platform
import random
import resource
import time
from pathlib import Path

import torch

from .data import read_rows
from .model import Policy
from .music import Task, evaluate_score, reward
from .objective import group_advantages, grpo_loss


def append_json(path, row):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def metadata(policy, stage, started, **extra):
    # ru_maxrss is bytes on macOS, KiB on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"stage": stage, "config": policy.config, "resolved_revision": policy.resolved_revision,
            "device": str(policy.device), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "numpy")},
            "seconds": time.perf_counter() - started,
            "process_peak_rss_gb": rss / (1024 ** 3 if platform.system() == "Darwin" else 1024 ** 2),
            "trainable_parameters": sum(p.numel() for p in policy.trainable_parameters()),
            "paid_compute_eur": 0, **extra}


def train_sft(config, data_dir, output):
    output = Path(output)
    if output.exists():
        raise ValueError(f"Output already exists: {output}. Choose a new run directory.")
    output.mkdir(parents=True)
    rows = read_rows(Path(data_dir) / "train.jsonl")
    random.Random(config["seed"]).shuffle(rows)
    started = time.perf_counter()
    policy = Policy(config, training="sft")
    optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=config["sft_lr"])
    for step in range(config["sft_steps"]):
        row = rows[step % len(rows)]
        ids, prefix = policy.target_ids(row["prompt"], row["target"])
        policy.model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = -policy.logps(ids, prefix).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite SFT loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        append_json(output / "train.jsonl", {"step": step + 1, "loss": loss.item(),
                    "grad_norm": norm.item(), "task_id": row["id"], "seconds": time.perf_counter() - started})
    info = metadata(policy, "sft", started, updates=config["sft_steps"])
    policy.save(output / "adapter", info)
    (output / "summary.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def train_rl(config, data_dir, sft_adapter, output):
    if config["group_size"] < 2:
        raise ValueError("group_size must be >= 2")
    output = Path(output)
    if output.exists():
        raise ValueError(f"Output already exists: {output}. Choose a new run directory.")
    output.mkdir(parents=True)
    rows = read_rows(Path(data_dir) / "train.jsonl")
    random.Random(config["seed"] + 1).shuffle(rows)
    started = time.perf_counter()
    policy = Policy(config, sft_adapter=sft_adapter, training="rl")
    optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=config["rl_lr"])
    updates, skipped = 0, 0
    for step in range(config["rl_steps"]):
        row = rows[step % len(rows)]
        task = Task(**row["task"])
        trajectories = policy.generate_many(row["prompt"], count=config["group_size"], sample=True)
        metrics = [evaluate_score(t["text"], task, truncated=t["truncated"]) for t in trajectories]
        scores = [reward(m, config["reward"]) for m in metrics]
        diagnostics = {"reward_mean": sum(scores) / len(scores),
                       "success_rate": sum(m["all_satisfied"] for m in metrics) / len(metrics),
                       "valid_rate": sum(m["valid"] for m in metrics) / len(metrics),
                       "mean_completion_tokens": sum(t["tokens"] for t in trajectories) / len(trajectories)}
        advantages = group_advantages(torch.tensor(scores, device=policy.device))
        # Equal-reward groups have no relative learning signal. Report, don't hide.
        zero_variance = max(scores) - min(scores) < 1e-8
        for t, score, m in zip(trajectories, scores, metrics):
            with (output / "rollouts.jsonl").open("a") as handle:
                handle.write(json.dumps({"step": step + 1, "task_id": row["id"], "task": row["task"],
                             "text": t["text"], "truncated": t["truncated"], "reward": score, "metrics": m}) + "\n")
        if zero_variance:
            skipped += 1
            append_json(output / "train.jsonl", {"step": step + 1, "skipped_zero_variance": True,
                        "rewards": scores, "updates": updates, **diagnostics,
                        "seconds": time.perf_counter() - started})
            continue
        policy.model.eval()
        with torch.no_grad():
            old_logps = [policy.logps(t["ids"], t["prompt_length"]).detach() for t in trajectories]
            with policy.model.disable_adapter():
                reference_logps = [policy.logps(t["ids"], t["prompt_length"]).detach() for t in trajectories]
        policy.model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, kl_sum, clip_sum, ratio_sum = 0., 0., 0., 0.
        # Sequential microbatches share one optimizer update per on-policy group.
        for t, old, ref, advantage in zip(trajectories, old_logps, reference_logps, advantages):
            logps = policy.logps(t["ids"], t["prompt_length"])
            loss, stats = grpo_loss(logps, old, ref, advantage, clip=config["clip"], beta=config["beta"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite GRPO loss")
            (loss / len(trajectories)).backward()
            loss_sum += loss.item() / len(trajectories)
            kl_sum += stats["kl"] / len(trajectories)
            clip_sum += stats["clip_fraction"] / len(trajectories)
            ratio_sum += stats["ratio_mean"] / len(trajectories)
        norm = torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        updates += 1
        append_json(output / "train.jsonl", {"step": step + 1, "loss": loss_sum, "grad_norm": norm.item(),
                    "kl": kl_sum, "clip_fraction": clip_sum, "rewards": scores, "updates": updates,
                    "ratio_mean": ratio_sum, **diagnostics,
                    "skipped_zero_variance": False, "seconds": time.perf_counter() - started})
        policy.clear_cache()
    info = metadata(policy, "grpo", started, updates=updates, skipped_groups=skipped,
                    sft_adapter=str(Path(sft_adapter).resolve()),
                    note="Pilot only. A completed update does not establish improved held-out performance.")
    policy.save(output / "adapter", info)
    (output / "summary.json").write_text(json.dumps(info, indent=2) + "\n")
    return info
