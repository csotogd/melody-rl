"""Synthetic task construction with disjoint constraint tuples, not just RNG seeds."""
from dataclasses import asdict
from itertools import product
import json
import random
from pathlib import Path

from .music import MAJOR, ROOTS, Task, evaluate_score, pitch_name


def teacher(task: Task, rng: random.Random) -> str:
    """A rule-based feasible sampler, also a mandatory non-neural baseline.

    Work backwards from the requested final note so the last leap is feasible.
    This is not a human-composed music corpus.
    """
    allowed = [p for p in range(task.low, task.high + 1) if (p - ROOTS[task.key]) % 12 in MAJOR]
    notes = [task.final]
    for _ in range(task.beats * task.bars - 1):
        choices = [p for p in allowed if 0 < abs(p - notes[-1]) <= task.max_leap]
        notes.append(rng.choice(choices or [notes[-1]]))
    notes.reverse()
    text = " | ".join(" ".join(pitch_name(p) for p in notes[i:i + task.beats])
                      for i in range(0, len(notes), task.beats))
    assert evaluate_score(text, task)["all_satisfied"]
    return text


def make_splits(seed: int = 42) -> dict[str, list[Task]]:
    candidates, composition, length = [], [], []
    for key, beats, bars, low, leap, final_pc in product(
        ROOTS, (3, 4), (1, 2, 3), (60, 62, 65), (5, 7, 12), (0, 4, 7)
    ):
        final = next(p for p in range(low, low + 13) if (p - ROOTS[key]) % 12 == final_pc)
        task = Task(key, beats, bars, low, low + 12, final, leap)
        if bars == 3:
            length.append(task)
        elif key == "G" and beats == 4:
            composition.append(task)
        else:
            candidates.append(task)
    rng = random.Random(seed)
    for group in (candidates, composition, length):
        rng.shuffle(group)
    # Test constraints never appear in SFT, RL, checkpoint selection or teacher data.
    return {"train": candidates[96:], "validation": candidates[:48], "test_id": candidates[48:96],
            "test_composition": composition, "test_length": length}


def write_data(directory: Path, seed: int = 42):
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": seed, "provenance": "Procedural tasks and rule-based quarter-note teacher; no scraped music."}
    for split, tasks in make_splits(seed).items():
        path = directory / f"{split}.jsonl"
        with path.open("w") as handle:
            for i, task in enumerate(tasks):
                row = {"id": task.id, "task": asdict(task), "prompt": task.prompt()}
                if split == "train":
                    row["target"] = teacher(task, random.Random(seed + i))
                handle.write(json.dumps(row) + "\n")
        manifest[split] = {"count": len(tasks), "ids": [task.id for task in tasks]}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {split: details["count"] for split, details in manifest.items() if isinstance(details, dict)}


def read_rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
