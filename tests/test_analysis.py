import copy

import pytest

from melody_rl.analysis import paired_comparison, task_metrics


def record(identifier, flags):
    candidates = [{"metrics": {"all_satisfied": flag, "valid": True, "repeat_rate": 0.,
                               "checks": dict.fromkeys(("bars", "meter", "key", "range", "final", "leap"), flag)}}
                  for flag in flags]
    return {"id": identifier, "candidates": candidates, "metrics": candidates[-1]["metrics"]}


def test_candidates_do_not_become_independent_tasks():
    before = [record("a", [False, False, False, False]), record("b", [True, True, True, True])]
    after = [record("a", [True, True, True, True]), record("b", [True, True, True, True])]
    comparison = paired_comparison(before, after, resamples=1000)
    assert comparison["tasks"] == 2
    assert comparison["candidates_per_policy"] == 8
    assert comparison["metrics"]["candidate_success"]["difference"] == .5
    assert comparison["metrics"]["candidate_success"]["difference_ci95"] == [0., 1.]


def test_sample_success_is_separate_from_selection():
    row = record("a", [False, False, False, True])
    assert task_metrics(row)["candidate_success"] == .25
    assert task_metrics(row)["selected_success"] == 1.
    same = paired_comparison([row], [copy.deepcopy(row)], resamples=100)
    assert same["metrics"]["candidate_success"]["difference_ci95"] == [0., 0.]


def test_unpaired_ids_or_unequal_budgets_rejected():
    a = record("a", [True])
    with pytest.raises(ValueError):
        paired_comparison([a], [record("b", [True])])
    with pytest.raises(ValueError):
        paired_comparison([a], [record("a", [True, False])])
