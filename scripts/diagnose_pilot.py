"""Audit saved pilot evidence without inference or additional training."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from melody_rl.analysis import paired_comparison
from melody_rl.music import parse_score, pitch_name


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    run = ROOT / "runs/local-study"
    train = rows(run / "grpo/train.jsonl")
    rollouts = rows(run / "grpo/rollouts.jsonl")
    sampled = [c for row in rows(run / "bestof4/predictions.jsonl") for c in row["candidates"]]
    updates = [row for row in train if not row["skipped_zero_variance"]]
    greedy = rows(ROOT / "runs/diagnosis/sft-greedy/predictions.jsonl")
    prefixes = Counter(tuple(pitch_name(p) for bar in parse_score(r["text"]) for p in bar)[:2]
                       for r in greedy if r["metrics"]["valid"])
    result = {
        "training": {"groups": len(train), "updates": len(updates), "skipped_equal_reward": len(train) - len(updates),
                     "unique_training_tasks": len({r["task_id"] for r in rollouts}), "available_training_tasks": 174,
                     "sampled_completions": len(rollouts),
                     "successful_completions": sum(r["metrics"]["all_satisfied"] for r in rollouts),
                     "valid_completions": sum(r["metrics"]["valid"] for r in rollouts),
                     "min_grad_norm": min(r["grad_norm"] for r in updates),
                     "max_grad_norm": max(r["grad_norm"] for r in updates),
                     "last_kl": updates[-1]["kl"]},
        "sft_stochastic_pilot_validation": {
            "tasks": 24, "completions": len(sampled),
            "successes": sum(c["metrics"]["all_satisfied"] for c in sampled),
            "failed_constraints": dict(Counter(k for c in sampled for k, v in c["metrics"]["checks"].items() if not v))},
        "greedy_prefix_concentration": {"most_common_two_note_prefix": list(prefixes.most_common(1)[0][0]),
                                        "count": prefixes.most_common(1)[0][1], "tasks": len(greedy)},
        "full_validation_greedy": paired_comparison(
            rows(ROOT / "runs/diagnosis/sft-greedy/predictions.jsonl"),
            rows(ROOT / "runs/diagnosis/pilot-grpo-greedy/predictions.jsonl")),
        "interpretation": [
            "Increasing evaluation size measures uncertainty, not a stronger learning update.",
            "28 updates over 32 of 174 training tasks is a short feasibility run.",
            "Only 4 of 32 groups have zero reward variance, so total lack of reward signal is not supported.",
            "Nonzero finite gradients and KL indicate updates, but do not prove the implementation has no bugs.",
            "Stochastic SFT success is 30/96, distinct from greedy 21/24 and selected best-of-4 18/24.",
            "The graded reward does not penalize repetition. Better compliance need not mean better music."
        ]}
    dest = ROOT / "results/diagnosis/pilot-audit.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2) + "\n")
    print(dest)


if __name__ == "__main__":
    main()
