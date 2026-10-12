import json
import math
from pathlib import Path
import random
import time

from .data import read_rows, teacher
from .music import Task, evaluate_score, reward, write_midi, write_wav


def summarize(records):
    n = len(records)
    successes = sum(r["metrics"]["all_satisfied"] for r in records)
    p = successes / n if n else 0
    z = 1.96
    denominator = 1 + z * z / max(1, n)
    center = (p + z * z / (2 * max(1, n))) / denominator
    margin = z * math.sqrt(p * (1 - p) / max(1, n) + z * z / (4 * max(1, n) ** 2)) / denominator
    valid = [r for r in records if r["metrics"]["valid"]]
    return {"n": n, "successes": successes, "all_satisfied_rate": p,
            "wilson_95_interval": [max(0, center - margin), min(1, center + margin)] if n else None,
            "valid_rate": len(valid) / n if n else 0,
            "mean_partial_all_outputs": sum(r["metrics"]["partial"] for r in records) / n if n else 0,
            "repeat_rate_valid_outputs": sum(r["metrics"]["repeat_rate"] for r in valid) / len(valid) if valid else None,
            "unique_output_fraction_valid": len(set(r["text"] for r in valid)) / len(valid) if valid else None,
            "truncated_rate": sum(r["truncated"] for r in records) / n if n else 0,
            "constraint_rates_all_outputs": {k: sum(r["metrics"]["checks"][k] for r in records) / n if n else 0
                                             for k in ("bars", "meter", "key", "range", "final", "leap")}}


def evaluate(config, data_dir, output, *, split="validation", sft_adapter=None, rl_adapter=None,
             rule_based=False, best_of=1, limit=None):
    output = Path(output)
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    if best_of < 1:
        raise ValueError("best_of must be positive")
    rows = read_rows(Path(data_dir) / f"{split}.jsonl")
    limit = config["eval_limit"] if limit is None else limit
    if limit > 0:
        rows = rows[:limit]
    output.mkdir(parents=True)
    policy = None
    started = time.perf_counter()
    if not rule_based:
        from .model import Policy
        policy = Policy(config, sft_adapter=sft_adapter, rl_adapter=rl_adapter)
    records, generated_tokens, generation_seconds = [], 0, 0.
    for index, row in enumerate(rows):
        task = Task(**row["task"])
        candidates = []
        generations = policy.generate_many(row["prompt"], count=best_of, sample=best_of > 1) if policy else None
        for candidate in range(best_of):
            if rule_based:
                generation = {"text": teacher(task, random.Random(config["seed"] + index * best_of + candidate)),
                              "truncated": False, "tokens": 0, "seconds": 0}
            else:
                generation = generations[candidate]
            metrics = evaluate_score(generation["text"], task, truncated=generation["truncated"])
            candidates.append({"text": generation["text"], "truncated": generation["truncated"], "metrics": metrics})
            generated_tokens += generation["tokens"]
            generation_seconds += generation["seconds"]
        selected = max(candidates, key=lambda item: reward(item["metrics"], config["reward"]))
        record = {"id": row["id"], "task": row["task"], "prompt": row["prompt"], **selected, "candidates": candidates}
        records.append(record)
        with (output / "predictions.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        if selected["metrics"]["valid"]:
            write_midi(selected["text"], output / "audio" / f"{index:03d}.mid")
            write_wav(selected["text"], output / "audio" / f"{index:03d}.wav")
        print(json.dumps({"item": index + 1, "id": task.id, "text": selected["text"],
                          "valid": selected["metrics"]["valid"], "success": selected["metrics"]["all_satisfied"]}), flush=True)
    summary = {**summarize(records), "split": split, "method": "rules" if rule_based else "grpo" if rl_adapter else "sft" if sft_adapter else "base",
               "config": config, "sft_adapter": str(sft_adapter) if sft_adapter else None,
               "rl_adapter": str(rl_adapter) if rl_adapter else None,
               "resolved_revision": policy.resolved_revision if policy else None,
               "best_of": best_of, "generated_tokens": generated_tokens, "generation_seconds": generation_seconds,
               "tokens_per_second": generated_tokens / generation_seconds if generation_seconds else None,
               "seconds": time.perf_counter() - started,
               "note": "Validation pilot; not a final test result. Diversity proxies are not musical quality." if split == "validation" else "Reserved test split; record any use for model selection."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
