import random
import wave
import pytest
from melody_rl.music import Task, evaluate_score, parse_score, reward, write_midi, write_wav
from melody_rl.data import make_splits, teacher


def task():
    return Task("C", 3, 2, 60, 72, 60, 7)


@pytest.mark.parametrize("text", ["", "C4 |", "C4 || C4", "```C4 D4 E4```", "C4 D4 E4 explanation", "H4", "B#4"])
def test_parser_rejects_reward_hacks(text):
    assert not evaluate_score(text, task())["valid"]


def test_checks_and_truncation():
    text = "C4 D4 E4 | G4 E4 C4"
    assert evaluate_score(text, task())["all_satisfied"]
    assert reward(evaluate_score(text, task(), truncated=True)) == 0
    metrics = evaluate_score("C4 D4 F#4 | G4 E4 C4", task())
    assert metrics["valid"] and not metrics["checks"]["key"]
    assert not evaluate_score("C4 D4 E4 G4 E4 C4", task())["checks"]["meter"]


def test_constant_pitch_is_reported_as_degenerate_not_musical_quality():
    metrics = evaluate_score("C4 C4 C4 | C4 C4 C4", task())
    assert metrics["all_satisfied"]
    assert metrics["repeat_rate"] == 1 and metrics["diversity"] == 0
    assert reward(metrics, "graded") == 1
    assert reward(metrics, "diverse") < 1


def test_diversity_reward_keeps_constraint_compliance_as_priority():
    constant = evaluate_score("C4 C4 C4 | C4 C4 C4", task())
    varied = evaluate_score("C4 D4 E4 | G4 E4 C4", task())
    wrong_key = evaluate_score("C4 D4 F#4 | G4 E4 C4", task())
    assert reward(varied, "diverse") == pytest.approx(1.)
    assert reward(constant, "diverse") == pytest.approx(.9)
    assert reward(wrong_key, "diverse") <= .5625
    assert reward(varied, "diverse") > reward(constant, "diverse") > reward(wrong_key, "diverse")


def test_split_disjointness_and_actual_distribution_shift():
    splits = make_splits()
    all_ids = [t.id for tasks in splits.values() for t in tasks]
    assert len(all_ids) == len(set(all_ids))
    assert all(t.bars <= 2 and not (t.key == "G" and t.beats == 4) for t in splits["train"])
    assert all(t.key == "G" and t.beats == 4 for t in splits["test_composition"])
    assert all(t.bars == 3 for t in splits["test_length"])
    assert make_splits(42) == splits


def test_teacher_satisfies_every_feasible_task():
    for tasks in make_splits().values():
        for t in tasks:
            assert evaluate_score(teacher(t, random.Random(0)), t)["all_satisfied"]


def test_export(tmp_path):
    text = "C4 D4 E4 | G4 E4 C4"
    write_midi(text, tmp_path / "a.mid")
    write_wav(text, tmp_path / "a.wav")
    assert (tmp_path / "a.mid").read_bytes().startswith(b"MThd")
    with wave.open(str(tmp_path / "a.wav")) as audio:
        assert audio.getnframes() == 6 * 11025
