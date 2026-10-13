"""Paired comparisons, with the task (not each correlated completion) as the unit."""
import random
import math


def task_metrics(record):
    candidates = record["candidates"]
    if not candidates:
        raise ValueError("An evaluation task must contain candidates")
    n = len(candidates)
    valid = [c for c in candidates if c["metrics"]["valid"]]
    return {
        "candidate_success": sum(c["metrics"]["all_satisfied"] for c in candidates) / n,
        "selected_success": float(record["metrics"]["all_satisfied"]),
        "candidate_validity": len(valid) / n,
        # Keep undefined repetition explicit rather than rewarding invalid output.
        "candidate_repetition": (sum(c["metrics"]["repeat_rate"] for c in valid) / len(valid)
                                 if valid else None),
        **{f"candidate_{key}": sum(c["metrics"]["checks"][key] for c in candidates) / n
           for key in ("bars", "meter", "key", "range", "final", "leap")},
    }


def paired_comparison(before, after, *, seed=42, resamples=10000):
    if not before or resamples < 1:
        raise ValueError("Nonempty records and positive resamples are required")
    ids = [r["id"] for r in before]
    if len(set(ids)) != len(ids) or ids != [r["id"] for r in after]:
        raise ValueError("Expected unique task IDs in identical order")
    if any(len(a["candidates"]) != len(b["candidates"]) for a, b in zip(before, after)):
        raise ValueError("Candidate budgets must match")
    a, b = [task_metrics(r) for r in before], [task_metrics(r) for r in after]
    result = {"tasks": len(ids), "candidates_per_policy": sum(len(r["candidates"]) for r in before),
              "bootstrap_seed": seed, "bootstrap_resamples": resamples,
              "interval_note": "Paired percentile bootstrap over tasks; conditional on one training seed. Repeated samples are not independent tasks. A collapsed interval with no discordant tasks does not establish equivalence.",
              "metrics": {}}
    improved = sum(not x["metrics"]["all_satisfied"] and y["metrics"]["all_satisfied"] for x, y in zip(before, after))
    regressed = sum(x["metrics"]["all_satisfied"] and not y["metrics"]["all_satisfied"] for x, y in zip(before, after))
    discordant = improved + regressed
    exact_p = (min(1., 2 * sum(math.comb(discordant, k) for k in range(min(improved, regressed) + 1)) / 2 ** discordant)
               if discordant else 1.)
    result["paired_binary_selected_success"] = {
        "improved": improved, "regressed": regressed, "unchanged": len(ids) - discordant,
        "exact_mcnemar_p_two_sided": exact_p,
        "note": "Secondary binary endpoint; unadjusted exact paired test. Few discordant tasks can make percentile bootstrap intervals overconfident; do not infer significance from those intervals alone.",
    }
    for key in a[0]:
        pairs = [(x[key], y[key]) for x, y in zip(a, b) if x[key] is not None and y[key] is not None]
        if not pairs:
            result["metrics"][key] = None
            continue
        differences = [y - x for x, y in pairs]
        n = len(pairs)
        rng = random.Random(seed)
        samples = sorted(sum(rng.choices(differences, k=n)) / n for _ in range(resamples))
        result["metrics"][key] = {"paired_tasks": n,
            "before": sum(x for x, _ in pairs) / n, "after": sum(y for _, y in pairs) / n,
            "difference": sum(differences) / n,
            "difference_ci95": [samples[int(.025 * (resamples - 1))], samples[int(.975 * (resamples - 1))]]}
    return result
