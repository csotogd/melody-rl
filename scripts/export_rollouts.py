"""Export predetermined training groups, including failures, as audible progress snapshots."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from melody_rl.music import write_midi, write_wav


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 8, 16, 32])
    args = parser.parse_args()
    groups = {}
    # Read only complete lines so a running trainer can keep appending safely.
    for line in args.rollouts.read_text().splitlines(keepends=True):
        if not line.endswith("\n"):
            continue
        row = json.loads(line)
        if row["step"] in args.steps:
            groups.setdefault(row["step"], []).append(row)
    for step, rows in groups.items():
        output = args.output / f"rl-step-{step:03d}"
        output.mkdir(parents=True, exist_ok=True)
        (output / "outputs.json").write_text(json.dumps({
            "note": "Four stochastic training-prompt samples collected BEFORE this group's update. These are not held-out evaluations or saved model checkpoints.",
            "selection": "Predetermined group number; all samples retained, including failures.",
            "step": step, "samples": rows,
        }, indent=2) + "\n")
        for index, row in enumerate(rows):
            if row["metrics"]["valid"]:
                write_wav(row["text"], output / f"sample-{index}.wav")
                write_midi(row["text"], output / f"sample-{index}.mid")
        print(f"Step {step}: exported {len(rows)} outputs to {output}")


if __name__ == "__main__":
    main()
