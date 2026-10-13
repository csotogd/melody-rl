"""Export all fixed-policy results and paired comparisons from completed evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from melody_rl.analysis import paired_comparison, task_metrics
from melody_rl.music import parse_score, pitch_name


def read_evaluation(directory):
    summary = json.loads((directory / "summary.json").read_text())
    rows = [json.loads(s) for s in (directory / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == summary["n"] and summary["config"]["reward"] == "graded"
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    layouts = {
        "validation": {"sft": ("sft-eval", "bestof4"), "graded": ("grpo-eval", "grpo-sampled"),
                       "diverse": ("diverse-eval", "diverse-sampled")},
        "test_id": {policy: (f"test-id-{folder}-greedy", f"test-id-{folder}-sampled")
                    for policy, folder in (("sft", "sft"), ("graded", "grpo"), ("diverse", "diverse"))}}
    result = {"note": "All policies fixed before the first ID test evaluation. One training seed. Diversity and repetition are proxies, not musical quality.",
              "splits": {}, "training": {}}
    task_ids = {}
    for split, layout in layouts.items():
        result["splits"][split] = {}
        loaded = {}
        for policy, (greedy_dir, sampled_dir) in layout.items():
            gs, greedy = read_evaluation(args.run / greedy_dir)
            ss, sampled = read_evaluation(args.run / sampled_dir)
            assert gs["split"] == ss["split"] == split
            assert [r["id"] for r in greedy] == [r["id"] for r in sampled]
            assert gs["best_of"] == 1 and ss["best_of"] == 4
            metrics = [task_metrics(r) for r in sampled]
            repetitions = [m["candidate_repetition"] for m in metrics if m["candidate_repetition"] is not None]
            candidates = [c for r in sampled for c in r["candidates"]]
            prefixes = Counter(tuple(pitch_name(p) for bar in parse_score(r["text"]) for p in bar)[:2]
                               for r in greedy if r["metrics"]["valid"])
            prefix, count = prefixes.most_common(1)[0] if prefixes else ((), 0)
            result["splits"][split][policy] = {
                "tasks": len(greedy), "greedy_successes": gs["successes"],
                "greedy_repetition": gs["repeat_rate_valid_outputs"],
                "sampled_successes": sum(c["metrics"]["all_satisfied"] for c in candidates),
                "sampled_completions": len(candidates),
                "sampled_success_rate": sum(m["candidate_success"] for m in metrics) / len(metrics),
                "sampled_repetition_mean_per_task": sum(repetitions) / len(repetitions) if repetitions else None,
                "sampled_validity": sum(c["metrics"]["valid"] for c in candidates) / len(candidates),
                "best_of_four_successes": ss["successes"],
                "greedy_dominant_two_note_prefix": list(prefix), "greedy_dominant_prefix_count": count,
                "sampled_generated_tokens": ss["generated_tokens"], "sampled_generation_seconds": ss["generation_seconds"],
            }
            loaded[policy] = (greedy, sampled)
        task_ids[split] = {r["id"] for r in loaded["sft"][0]}
        for before, after, label in (("sft", "graded", ""), ("sft", "diverse", "diverse-")):
            for index, mode in enumerate(("greedy", "sampled")):
                name = f'{"test-id" if split == "test_id" else split}-{label}{mode}-comparison.json'
                comparison = paired_comparison(loaded[before][index], loaded[after][index])
                (args.run / name).write_text(json.dumps(comparison, indent=2) + "\n")
        comparison = paired_comparison(loaded["graded"][1], loaded["diverse"][1])
        name = f'{"test-id" if split == "test_id" else split}-reward-ablation-comparison.json'
        (args.run / name).write_text(json.dumps(comparison, indent=2) + "\n")
    assert not task_ids["validation"] & task_ids["test_id"]
    for policy, folder in (("graded", "grpo"), ("diverse", "diverse")):
        summary = json.loads((args.run / folder / "summary.json").read_text())
        result["training"][policy] = {key: summary[key] for key in
                                      ("config", "updates", "skipped_groups", "seconds", "device", "paid_compute_eur")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
